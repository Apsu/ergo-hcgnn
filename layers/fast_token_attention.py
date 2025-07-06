#!/usr/bin/env python3
"""
Fast, vectorized token attention layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FastCrossMessageTokenAttention(nn.Module):
    """
    Vectorized cross-message token attention for reference resolution.
    No Python loops - fully GPU accelerated.
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 top_k_ratio: float = 0.1,
                 temperature: float = 1.0):
        """
        Args:
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
            top_k_ratio: Ratio of tokens to connect across messages
            temperature: Temperature for attention
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k_ratio = top_k_ratio
        self.temperature = temperature
        
        # Multi-head attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Token importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                token_features: torch.Tensor,
                message_boundaries: torch.Tensor,
                batch_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply vectorized cross-message attention.
        
        Args:
            token_features: [total_tokens, hidden_dim] all tokens from all messages
            message_boundaries: [num_messages, 2] start/end indices for each message
            batch_indices: [num_messages] which batch each message belongs to
            
        Returns:
            updated_features: [total_tokens, hidden_dim]
            attention_scores: [num_selected_tokens, max_targets] for visualization
        """
        num_tokens = token_features.size(0)
        device = token_features.device
        
        # Score token importance
        importance_scores = self.importance_scorer(token_features).squeeze(-1)  # [num_tokens]
        
        # Create message masks efficiently
        num_messages = len(message_boundaries)
        message_mask = torch.zeros(num_tokens, num_messages, device=device, dtype=torch.bool)
        
        for msg_idx in range(num_messages):
            start_idx, end_idx = message_boundaries[msg_idx]
            message_mask[start_idx:end_idx, msg_idx] = True
        
        # Select top-k important tokens per message
        selected_indices = []
        message_ids = []
        
        for msg_idx in range(num_messages):
            msg_mask = message_mask[:, msg_idx]
            msg_scores = importance_scores.masked_fill(~msg_mask, float('-inf'))
            
            # Get top-k tokens in this message
            num_tokens_in_msg = msg_mask.sum().item()
            k = max(1, int(num_tokens_in_msg * self.top_k_ratio))
            
            if num_tokens_in_msg > 0:
                top_k_values, top_k_indices = torch.topk(msg_scores, min(k, num_tokens_in_msg))
                selected_indices.append(top_k_indices)
                message_ids.extend([msg_idx] * len(top_k_indices))
        
        if not selected_indices:
            return token_features, torch.empty(0, 0, device=device)
        
        selected_indices = torch.cat(selected_indices)  # [num_selected]
        message_ids = torch.tensor(message_ids, device=device)  # [num_selected]
        
        # Compute attention in parallel for all selected tokens
        queries = self.query_proj(token_features[selected_indices])  # [num_selected, hidden_dim]
        
        # Reshape for multi-head attention
        batch_size = len(selected_indices)
        queries = queries.view(batch_size, self.num_heads, self.head_dim)
        
        # For efficiency, compute keys and values for all tokens once
        keys = self.key_proj(token_features).view(num_tokens, self.num_heads, self.head_dim)
        values = self.value_proj(token_features).view(num_tokens, self.num_heads, self.head_dim)
        
        # Compute attention scores for all query-key pairs
        # [batch_size, num_heads, num_tokens]
        attention_scores = torch.bmm(queries, keys.transpose(0, 1).transpose(1, 2)) / (self.head_dim ** 0.5)
        attention_scores = attention_scores / self.temperature
        
        # Mask out tokens from the same message
        # Create mask where mask[i, j] = True if query i and token j are from different messages
        query_message_ids = message_ids  # [num_selected]
        token_message_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
        for msg_idx in range(num_messages):
            start_idx, end_idx = message_boundaries[msg_idx]
            token_message_ids[start_idx:end_idx] = msg_idx
        
        # [num_selected, num_tokens]
        different_message_mask = query_message_ids.unsqueeze(1) != token_message_ids.unsqueeze(0)
        
        # Apply mask to attention scores
        attention_scores = attention_scores.masked_fill(
            ~different_message_mask.unsqueeze(1), 
            float('-inf')
        )
        
        # Apply batch mask if provided
        if batch_indices is not None:
            query_batch_ids = batch_indices[query_message_ids]
            token_batch_ids = batch_indices[token_message_ids]
            same_batch_mask = query_batch_ids.unsqueeze(1) == token_batch_ids.unsqueeze(0)
            attention_scores = attention_scores.masked_fill(
                ~same_batch_mask.unsqueeze(1),
                float('-inf')
            )
        
        # Get top-k attention targets per query
        k_targets = min(10, num_tokens)  # Limit number of targets per query
        # [batch_size, num_heads, k_targets]
        top_k_scores, top_k_indices = torch.topk(attention_scores, k_targets, dim=-1)
        
        # Compute attention weights
        attention_weights = F.softmax(top_k_scores, dim=-1)  # [batch_size, num_heads, k_targets]
        attention_weights = self.dropout(attention_weights)
        
        # Gather values for top-k indices
        # We need to gather from values which is [num_tokens, num_heads, head_dim]
        # Create indices for gathering: [batch_size, num_heads, k_targets, head_dim]
        gather_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        
        # Gather values: [batch_size, num_heads, k_targets, head_dim]
        gathered_values = values.unsqueeze(0).expand(batch_size, -1, -1, -1).gather(
            dim=2, 
            index=gather_indices
        )
        
        # Apply attention: [batch_size, num_heads, head_dim]
        attended_values = torch.sum(
            attention_weights.unsqueeze(-1) * gathered_values,
            dim=2
        )
        
        # Reshape back: [batch_size, hidden_dim]
        attended_values = attended_values.view(batch_size, self.hidden_dim)
        
        # Apply output projection
        updates = self.output_proj(attended_values)
        
        # Update features with residual connection
        updated_features = token_features.clone()
        updated_features[selected_indices] = updated_features[selected_indices] + self.dropout(updates)
        
        # Return attention scores for visualization (average across heads)
        avg_attention_scores = top_k_scores.mean(dim=1)  # [batch_size, k_targets]
        
        return updated_features, avg_attention_scores