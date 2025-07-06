#!/usr/bin/env python3
"""
Dataset implementation for Hierarchical Conversation GNN V2
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    """
    Dataset for conversation data with dependency labels
    """
    
    def __init__(self,
                 conversations: List[Dict],
                 tokenizer: Optional[AutoTokenizer] = None,
                 max_length: int = 128,
                 max_messages: Optional[int] = None,
                 include_system_messages: bool = False,
                 pre_tokenized: bool = False):
        """
        Args:
            conversations: List of conversation dictionaries
            tokenizer: HuggingFace tokenizer (not needed if pre_tokenized=True)
            max_length: Maximum token length per message
            max_messages: Maximum messages per conversation (None for no limit)
            include_system_messages: Whether to include system messages
            pre_tokenized: Whether conversations are already tokenized
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_messages = max_messages
        self.include_system_messages = include_system_messages
        self.pre_tokenized = pre_tokenized
        
        if not pre_tokenized and tokenizer is None:
            raise ValueError("Tokenizer required when pre_tokenized=False")
        
        # Process and filter conversations
        self.conversations = []
        self.process_conversations(conversations)
        
        logger.info(f"Loaded {len(self.conversations)} conversations")
        
    def process_conversations(self, conversations: List[Dict]):
        """Process and validate conversations"""
        for conv in conversations:
            messages = conv.get('messages', conv)
            
            # Filter system messages if needed
            if not self.include_system_messages:
                messages = [m for m in messages if m['role'] != 'system']
            
            # Skip empty conversations
            if len(messages) < 2:
                continue
            
            # Truncate if needed
            if self.max_messages:
                messages = messages[:self.max_messages]
            
            # Validate dependency indices
            valid = True
            for i, msg in enumerate(messages):
                if msg.get('depends_on_indices'):
                    # Ensure all dependencies are valid
                    deps = [d for d in msg['depends_on_indices'] if 0 <= d < i]
                    if len(deps) != len(msg['depends_on_indices']):
                        logger.warning(f"Invalid dependencies in message {i}: {msg['depends_on_indices']}")
                        msg['depends_on_indices'] = deps
            
            if valid:
                processed_conv = {
                    'messages': messages,
                    'metadata': conv.get('metadata', {})
                }
                self.conversations.append(processed_conv)
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a conversation and prepare it for training
        
        Returns:
            Dict containing:
                - token_ids_list: List of token tensors
                - lengths: List of sequence lengths
                - num_messages: Number of messages
                - message_roles: Tensor of role indices
                - message_positions: Normalized positions
                - relevance_queries: List of query-context pairs for training
        """
        conv = self.conversations[idx]
        messages = conv['messages']
        
        # Tokenize messages
        token_ids_list = []
        lengths = []
        role_to_idx = {'user': 0, 'assistant': 1, 'system': 2, 'tool': 3}
        message_roles = []
        
        for msg in messages:
            if self.pre_tokenized:
                # Already tokenized
                token_ids = torch.tensor(msg['token_ids'], dtype=torch.long)
                token_ids_list.append(token_ids)
                lengths.append(msg['length'])
            else:
                # Tokenize on the fly
                tokens = self.tokenizer(
                    msg['text'],
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors='pt'
                )
                
                token_ids = tokens['input_ids'].squeeze()
                token_ids_list.append(token_ids)
                lengths.append(len(token_ids))
            
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
                'relevant_indices': msg.get('depends_on_indices', []),
                'dependency_type': msg.get('dependency_type', 'none'),
                'is_dependent': msg.get('is_context_dependent', False)
            }
            
            relevance_queries.append(query_info)
        
        # Prepare output
        return {
            'token_ids_list': token_ids_list,
            'lengths': lengths,
            'num_messages': len(messages),
            'message_roles': torch.tensor(message_roles, dtype=torch.long),
            'message_positions': torch.arange(len(messages), dtype=torch.float) / max(len(messages) - 1, 1),
            'relevance_queries': relevance_queries,
            'conversation_idx': idx
        }


def collate_conversations(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching conversations
    
    Handles variable-length sequences and creates batched graph structures
    """
    # Separate batch components
    all_token_ids = []
    all_lengths = []
    all_roles = []
    all_positions = []
    batch_indices = []
    
    for batch_idx, item in enumerate(batch):
        num_messages = item['num_messages']
        
        # Collect token data
        all_token_ids.extend(item['token_ids_list'])
        all_lengths.extend(item['lengths'])
        
        # Collect message data
        all_roles.append(item['message_roles'])
        all_positions.append(item['message_positions'])
        
        # Track which messages belong to which conversation
        batch_indices.extend([batch_idx] * num_messages)
    
    # Stack message attributes
    message_roles = torch.cat(all_roles)
    message_positions = torch.cat(all_positions)
    message_batch = torch.tensor(batch_indices, dtype=torch.long)
    
    # Create message node attributes (role, position)
    message_node_attr = torch.stack([
        message_roles.float(),
        message_positions
    ], dim=1)
    
    # Build message-level edges (temporal connections within each conversation)
    edge_list = []
    cum_nodes = 0
    
    for item in batch:
        num_messages = item['num_messages']
        
        # Add temporal edges for this conversation
        for i in range(num_messages - 1):
            # Forward edge
            edge_list.append([cum_nodes + i, cum_nodes + i + 1])
            # Backward edge
            edge_list.append([cum_nodes + i + 1, cum_nodes + i])
        
        cum_nodes += num_messages
    
    if edge_list:
        message_edge_index = torch.tensor(edge_list, dtype=torch.long).T
    else:
        message_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Prepare targets for loss computation
    targets = []
    for item in batch:
        targets.append({
            'relevance_queries': item['relevance_queries']
        })
    
    return {
        'token_ids_list': all_token_ids,
        'lengths': all_lengths,
        'num_messages_list': [item['num_messages'] for item in batch],
        'message_edge_index': message_edge_index,
        'message_node_attr': message_node_attr,
        'message_batch': message_batch,
        'targets': targets,
        'conversation_indices': [item['conversation_idx'] for item in batch]
    }


class ConversationSampler:
    """
    Custom sampler for curriculum learning and balanced sampling
    """
    
    def __init__(self,
                 dataset: ConversationDataset,
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 curriculum_scheduler: Optional['CurriculumScheduler'] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.curriculum_scheduler = curriculum_scheduler
        
    def __iter__(self):
        if self.curriculum_scheduler:
            # Use curriculum to select conversations
            indices = self.curriculum_scheduler.select_conversations(
                self.dataset.conversations,
                len(self.dataset)
            )
        else:
            indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            indices = np.random.permutation(indices).tolist()
        
        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        return iter(batches)
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size