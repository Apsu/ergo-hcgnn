#!/usr/bin/env python3
"""
Fast dataset using pre-tokenized conversations
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FastConversationDataset(Dataset):
    """
    Fast dataset that loads pre-tokenized conversations
    """
    
    def __init__(self, 
                 tokenized_path: str,
                 split: str = 'train',
                 val_split: float = 0.1,
                 test_split: float = 0.1):
        """
        Args:
            tokenized_path: Path to pre-tokenized conversations
            split: One of 'train', 'val', 'test'
            val_split: Validation split ratio
            test_split: Test split ratio
        """
        # Load pre-tokenized data
        logger.info(f"Loading pre-tokenized data from {tokenized_path}")
        data = torch.load(tokenized_path)
        
        self.conversations = data['conversations']
        self.tokenizer_name = data['tokenizer_name']
        self.max_length = data['max_length']
        self.vocab_size = data['vocab_size']
        
        # Split data
        total = len(self.conversations)
        val_size = int(total * val_split)
        test_size = int(total * test_split)
        
        if split == 'test':
            self.conversations = self.conversations[:test_size]
        elif split == 'val':
            self.conversations = self.conversations[test_size:test_size + val_size]
        else:  # train
            self.conversations = self.conversations[test_size + val_size:]
        
        logger.info(f"Loaded {len(self.conversations)} {split} conversations")
        
        # Filter out very short conversations
        self.conversations = [c for c in self.conversations if len(c['messages']) >= 2]
        logger.info(f"After filtering: {len(self.conversations)} conversations")
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get pre-tokenized conversation
        
        Returns dictionary with:
            - token_ids_list: List of token tensors  
            - lengths: List of sequence lengths
            - num_messages: Number of messages
            - message_roles: Tensor of role indices
            - message_positions: Normalized positions
            - relevance_queries: List of query-context pairs for training
        """
        conv = self.conversations[idx]
        messages = conv['messages']
        
        # Convert to tensors (already tokenized!)
        token_ids_list = []
        lengths = []
        role_to_idx = {'user': 0, 'assistant': 1, 'system': 2, 'tool': 3}
        message_roles = []
        
        for msg in messages:
            # Already tokenized - just convert to tensor
            token_ids = torch.tensor(msg['token_ids'], dtype=torch.long)
            token_ids_list.append(token_ids)
            lengths.append(msg['length'])
            
            # Role
            role_idx = role_to_idx.get(msg['role'], 0)
            message_roles.append(role_idx)
        
        # Create relevance queries for training
        relevance_queries = []
        
        for i, msg in enumerate(messages):
            if i == 0:  # First message has no context
                continue
            
            # Create query for this message
            query_info = {
                'query_idx': i,
                'context_indices': list(range(i)),  # All previous messages
                'relevance_labels': [0] * i,  # Will be updated below
                'dependency_type': msg.get('dependency_type', 'none')
            }
            
            # Mark dependent messages as relevant
            depends_on = msg.get('depends_on_indices', [])
            for dep_idx in depends_on:
                if 0 <= dep_idx < i:
                    query_info['relevance_labels'][dep_idx] = 1
            
            # Only add queries that have at least one relevant message
            if sum(query_info['relevance_labels']) > 0:
                relevance_queries.append(query_info)
        
        # Compute normalized message positions
        num_messages = len(messages)
        message_positions = torch.linspace(0, 1, num_messages)
        
        return {
            'token_ids_list': token_ids_list,
            'lengths': lengths,
            'num_messages': num_messages,
            'message_roles': torch.tensor(message_roles, dtype=torch.long),
            'message_positions': message_positions,
            'relevance_queries': relevance_queries
        }


def collate_conversations(batch: List[Dict]) -> Dict:
    """
    Collate function for batching conversations
    """
    # Flatten all token sequences
    all_token_ids = []
    all_lengths = []
    all_roles = []
    all_positions = []
    all_queries = []
    
    batch_indices = []  # Track which conversation each item belongs to
    
    for batch_idx, item in enumerate(batch):
        for i in range(item['num_messages']):
            all_token_ids.append(item['token_ids_list'][i])
            all_lengths.append(item['lengths'][i])
            batch_indices.append(batch_idx)
        
        all_roles.append(item['message_roles'])
        all_positions.append(item['message_positions'])
        
        # Adjust query indices for batching
        offset = sum(batch[j]['num_messages'] for j in range(batch_idx))
        for query in item['relevance_queries']:
            adjusted_query = {
                'query_idx': query['query_idx'] + offset,
                'context_indices': [idx + offset for idx in query['context_indices']],
                'relevance_labels': query['relevance_labels'],
                'dependency_type': query['dependency_type']
            }
            all_queries.append(adjusted_query)
    
    # Pad token sequences
    max_length = max(len(seq) for seq in all_token_ids)
    padded_tokens = torch.zeros(len(all_token_ids), max_length, dtype=torch.long)
    
    for i, seq in enumerate(all_token_ids):
        padded_tokens[i, :len(seq)] = seq
    
    message_roles = torch.cat(all_roles)
    message_positions = torch.cat(all_positions)
    
    return {
        'token_ids': padded_tokens,
        'lengths': all_lengths,
        'message_roles': message_roles.float(),
        'message_positions': message_positions,
        'batch_indices': batch_indices,
        'num_messages_per_conv': [item['num_messages'] for item in batch],
        'relevance_queries': all_queries
    }