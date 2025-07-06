#!/usr/bin/env python3
"""
Inference module for Hierarchical Conversation GNN V2
Provides clean API for model loading and context selection
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from transformers import AutoTokenizer

from models.hierarchical_gnn import HierarchicalConversationGNN


class ConversationContextSelector:
    """
    High-level interface for conversation context selection using Hierarchical GNN
    """
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: Optional[str] = None,
                 tokenizer_name: str = 'bert-base-uncased'):
        """
        Initialize the context selector
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            tokenizer_name: Name of HuggingFace tokenizer to use
        """
        self.device = self._setup_device(device)
        self.model, self.config = self._load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = self.config.get('max_seq_length', 512)
        
        # Role mapping
        self.role_map = {
            'user': 0,
            'assistant': 1, 
            'system': 2,
            'tool': 3
        }
        
        # Cache for performance
        self._embedding_cache = {}
        self._cache_enabled = True
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: Union[str, Path]) -> Tuple[HierarchicalConversationGNN, Dict]:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # Infer from state dict
            state_dict = checkpoint['model_state_dict']
            config = {
                'vocab_size': state_dict['token_embeddings.weight'].shape[0],
                'token_embedding_dim': state_dict['token_embeddings.weight'].shape[1],
                'hidden_dim': 256,
                'message_dim': 128,
                'num_heads': 4,
                'max_seq_length': state_dict['position_embeddings.weight'].shape[0]
            }
        
        # Initialize model
        model = HierarchicalConversationGNN(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def select_context(self,
                      messages: List[Dict[str, str]],
                      query: str,
                      max_context: int = 10,
                      temperature: float = 1.0,
                      min_score_threshold: Optional[float] = None,
                      dependency_type: Optional[str] = None) -> Dict[str, any]:
        """
        Select relevant context messages for a query
        
        Args:
            messages: List of conversation messages with 'role' and 'content'
            query: The query to find context for
            max_context: Maximum number of context messages to return
            temperature: Temperature for score scaling (higher = more random)
            min_score_threshold: Minimum score threshold for selection
            dependency_type: Optional dependency type for adaptive temperature
            
        Returns:
            Dictionary containing:
                - selected_indices: List of selected message indices
                - scores: All relevance scores
                - probabilities: Softmax probabilities
                - selected_messages: The selected messages
                - debug_info: Additional debugging information
        """
        if not messages:
            return {
                'selected_indices': [],
                'scores': [],
                'probabilities': [],
                'selected_messages': [],
                'debug_info': {}
            }
        
        # Prepare all messages including query
        all_messages = messages + [{"role": "user", "content": query}]
        
        # Get embeddings for all messages
        embeddings = self._get_message_embeddings(all_messages)
        
        # Score all messages against the query
        query_idx = len(messages)
        scores = self._score_messages(embeddings, query_idx, dependency_type)
        
        # Apply temperature
        scaled_scores = scores / temperature
        
        # Select top messages
        selected_indices, selected_scores = self._select_top_messages(
            scaled_scores, max_context, min_score_threshold
        )
        
        # Convert to probabilities
        probabilities = F.softmax(scaled_scores, dim=0)
        
        # Get selected messages
        selected_messages = [messages[i] for i in selected_indices]
        
        # Clear cache if it gets too large
        if len(self._embedding_cache) > 1000:
            self._embedding_cache.clear()
        
        return {
            'selected_indices': selected_indices,
            'scores': scores.tolist(),
            'probabilities': probabilities.tolist(),
            'selected_messages': selected_messages,
            'debug_info': {
                'embeddings': embeddings.cpu(),
                'query_embedding': embeddings[query_idx].cpu(),
                'temperature_used': temperature,
                'cache_size': len(self._embedding_cache)
            }
        }
    
    def _get_message_embeddings(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Get embeddings for all messages"""
        # Check if all messages are cached
        all_cached = True
        cached_embeddings = []
        
        if self._cache_enabled:
            for msg in messages:
                cache_key = (msg['role'], msg['content'])
                if cache_key in self._embedding_cache:
                    cached_embeddings.append(self._embedding_cache[cache_key].to(self.device))
                else:
                    all_cached = False
                    cached_embeddings.append(None)
        else:
            all_cached = False
        
        # If all cached, return them
        if all_cached:
            return torch.stack(cached_embeddings)
        
        # Prepare inputs for all messages (even cached ones for consistent graph structure)
        token_ids_list = []
        lengths = []
        segment_ids_list = []
        
        for msg in messages:
            # Tokenize
            encoded = self.tokenizer(
                msg['content'],
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            token_ids = encoded['input_ids'].squeeze(0).to(self.device)
            length = token_ids.size(0)
            
            token_ids_list.append(token_ids)
            lengths.append(length)
            
            # Create segment IDs for role
            role_id = self.role_map.get(msg['role'], 0)
            segment_ids = torch.full((length,), role_id, device=self.device)
            segment_ids_list.append(segment_ids)
        
        # Build message graph
        num_messages = len(messages)
        edge_list = []
        for i in range(num_messages - 1):
            edge_list.append([i, i + 1])  # Forward
            edge_list.append([i + 1, i])  # Backward
        
        if edge_list:
            message_edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).T
        else:
            message_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Create message node attributes
        message_node_attr = torch.zeros((num_messages, 2), device=self.device)
        for i, msg in enumerate(messages):
            message_node_attr[i, 0] = self.role_map.get(msg['role'], 0)
            message_node_attr[i, 1] = i / max(1, num_messages - 1)  # Normalized position
        
        # Get embeddings from model
        with torch.no_grad():
            embeddings = self.model(
                token_ids_list=token_ids_list,
                lengths=lengths,
                message_edge_index=message_edge_index,
                message_node_attr=message_node_attr,
                segment_ids_list=segment_ids_list
            )
        
        # Cache embeddings
        if self._cache_enabled:
            for i, msg in enumerate(messages):
                cache_key = (msg['role'], msg['content'])
                self._embedding_cache[cache_key] = embeddings[i].cpu()
        
        return embeddings
    
    def _score_messages(self, 
                       embeddings: torch.Tensor,
                       query_idx: int,
                       dependency_type: Optional[str] = None) -> torch.Tensor:
        """Score all messages against the query"""
        query_emb = embeddings[query_idx]
        num_context = query_idx
        
        # Map dependency type to index if provided
        dep_type_map = {
            'continuation': 0,
            'clarification': 1,
            'correction': 2,
            'follow_up': 3,
            'topic_reference': 4,
            'pronoun_reference': 5,
            'example_request': 6,
            'disagreement': 7
        }
        dep_type_idx = dep_type_map.get(dependency_type) if dependency_type else None
        
        scores = []
        for i in range(num_context):
            score = self.model.score_relevance(
                query_emb.unsqueeze(0),
                embeddings[i].unsqueeze(0),
                query_idx=query_idx,
                context_idx=i,
                dependency_type_idx=dep_type_idx
            )
            scores.append(score.item())
        
        return torch.tensor(scores, device=self.device)
    
    def _select_top_messages(self,
                           scores: torch.Tensor,
                           max_context: int,
                           min_score_threshold: Optional[float] = None) -> Tuple[List[int], List[float]]:
        """Select top scoring messages"""
        # Apply threshold if specified
        if min_score_threshold is not None:
            valid_mask = scores > min_score_threshold
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                return [], []
            
            valid_scores = scores[valid_indices]
            k = min(max_context, len(valid_indices))
            
            top_scores, top_indices_in_valid = torch.topk(valid_scores, k)
            top_indices = valid_indices[top_indices_in_valid]
        else:
            k = min(max_context, len(scores))
            top_scores, top_indices = torch.topk(scores, k)
        
        return top_indices.tolist(), top_scores.tolist()
    
    def get_token_importance(self, 
                           message: Dict[str, str],
                           message_idx: int) -> Optional[torch.Tensor]:
        """
        Get token importance scores for a specific message
        
        Args:
            message: Message dict with 'role' and 'content'
            message_idx: Index of the message in the last forward pass
            
        Returns:
            Token importance scores or None if not available
        """
        return self.model.get_token_importance(message_idx)
    
    def batch_select_context(self,
                           conversations: List[List[Dict[str, str]]],
                           queries: List[str],
                           **kwargs) -> List[Dict[str, any]]:
        """
        Batch process multiple conversations
        
        Args:
            conversations: List of conversation histories
            queries: List of queries (one per conversation)
            **kwargs: Additional arguments passed to select_context
            
        Returns:
            List of selection results
        """
        results = []
        for conv, query in zip(conversations, queries):
            result = self.select_context(conv, query, **kwargs)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable embedding caching"""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()


class StreamingContextSelector(ConversationContextSelector):
    """
    Extended selector with streaming support for real-time applications
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_buffer = []
        self.embedding_buffer = None
    
    def add_message(self, message: Dict[str, str]):
        """Add a message to the conversation buffer"""
        self.conversation_buffer.append(message)
        
        # Invalidate embedding buffer when conversation changes
        self.embedding_buffer = None
    
    def select_context_streaming(self, 
                                query: str,
                                **kwargs) -> Dict[str, any]:
        """
        Select context using the buffered conversation
        
        Args:
            query: The query to find context for
            **kwargs: Additional arguments passed to select_context
            
        Returns:
            Selection results
        """
        return self.select_context(
            self.conversation_buffer,
            query,
            **kwargs
        )
    
    def clear_conversation(self):
        """Clear the conversation buffer"""
        self.conversation_buffer = []
        self.embedding_buffer = None


def create_context_selector(model_path: Union[str, Path],
                          streaming: bool = False,
                          **kwargs) -> Union[ConversationContextSelector, StreamingContextSelector]:
    """
    Factory function to create appropriate context selector
    
    Args:
        model_path: Path to model checkpoint
        streaming: Whether to use streaming selector
        **kwargs: Additional arguments for selector initialization
        
    Returns:
        Context selector instance
    """
    if streaming:
        return StreamingContextSelector(model_path, **kwargs)
    else:
        return ConversationContextSelector(model_path, **kwargs)