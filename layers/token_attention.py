#!/usr/bin/env python3
"""
Learnable Token-to-Message Attention Layer
Replaces fixed pooling with learned attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenToMessageAttention(nn.Module):
    """
    Attention-based aggregation of token features into message embeddings.
    Learns which tokens are important for message-level understanding.
    """
    
    def __init__(self, 
                 token_dim: int,
                 message_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_positional_weights: bool = True):
        """
        Args:
            token_dim: Dimension of token embeddings
            message_dim: Dimension of output message embeddings
            num_heads: Number of attention heads for multi-aspect aggregation
            dropout: Dropout rate
            use_positional_weights: Whether to use position-based attention bias
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.message_dim = message_dim
        self.num_heads = num_heads
        self.head_dim = message_dim // num_heads
        assert message_dim % num_heads == 0, "message_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.query_proj = nn.Linear(token_dim, message_dim)
        self.key_proj = nn.Linear(token_dim, message_dim)
        self.value_proj = nn.Linear(token_dim, message_dim)
        
        # Learnable CLS-like query for aggregation
        self.message_query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))
        
        # Optional positional attention bias
        self.use_positional_weights = use_positional_weights
        if use_positional_weights:
            self.position_bias_proj = nn.Linear(1, num_heads)
        
        # Output projection
        self.output_proj = nn.Linear(message_dim, message_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(message_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism to blend with pooling
        self.attention_gate = nn.Sequential(
            nn.Linear(message_dim * 3, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, 
                token_embeddings: torch.Tensor,
                token_mask: Optional[torch.Tensor] = None,
                token_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to aggregate token embeddings into message embedding.
        
        Args:
            token_embeddings: [batch_size, seq_len, token_dim]
            token_mask: [batch_size, seq_len] boolean mask (True = valid token)
            token_positions: [batch_size, seq_len] normalized positions (0-1)
            
        Returns:
            message_embedding: [batch_size, message_dim]
            attention_weights: [batch_size, num_heads, seq_len] for visualization
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Project to multi-head format
        # [batch_size, seq_len, num_heads, head_dim]
        keys = self.key_proj(token_embeddings).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_proj(token_embeddings).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch_size, num_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Expand message query for batch: [batch_size, num_heads, 1, head_dim]
        message_query = self.message_query.expand(batch_size, -1, -1).unsqueeze(2)
        
        # Compute attention scores: [batch_size, num_heads, 1, seq_len]
        scores = torch.matmul(message_query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Add positional bias if enabled
        if self.use_positional_weights and token_positions is not None:
            # Give more weight to certain positions (e.g., start/end of message)
            pos_bias = self.position_bias_proj(token_positions.unsqueeze(-1))  # [batch, seq_len, num_heads]
            pos_bias = pos_bias.transpose(1, 2).unsqueeze(2)  # [batch, num_heads, 1, seq_len]
            scores = scores + pos_bias
        
        # Apply mask if provided
        if token_mask is not None:
            mask = token_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, 1, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values: [batch, num_heads, 1, head_dim]
        attended_values = torch.matmul(attention_weights, values)
        
        # Reshape to [batch, message_dim]
        attended_values = attended_values.squeeze(2).transpose(1, 2).reshape(batch_size, self.message_dim)
        
        # Apply output projection and layer norm
        attention_output = self.layer_norm(self.output_proj(attended_values))
        
        # Also compute traditional pooling for gating
        if token_mask is not None:
            masked_embeddings = token_embeddings * token_mask.unsqueeze(-1)
            sum_embeddings = masked_embeddings.sum(dim=1)
            mean_pool = sum_embeddings / token_mask.sum(dim=1, keepdim=True).clamp(min=1)
            max_pool = masked_embeddings.max(dim=1)[0]
        else:
            mean_pool = token_embeddings.mean(dim=1)
            max_pool = token_embeddings.max(dim=1)[0]
        
        # Project pooling to message dim
        mean_pool = self.layer_norm(self.output_proj(self.query_proj(mean_pool)))
        max_pool = self.layer_norm(self.output_proj(self.query_proj(max_pool)))
        
        # Learn how to blend attention with pooling
        concat_features = torch.cat([attention_output, mean_pool, max_pool], dim=-1)
        gates = self.attention_gate(concat_features)  # [batch, 3]
        
        # Weighted combination
        message_embedding = (
            gates[:, 0:1] * attention_output + 
            gates[:, 1:2] * mean_pool + 
            gates[:, 2:3] * max_pool
        )
        
        return message_embedding, attention_weights.squeeze(2)


class CrossMessageTokenAttention(nn.Module):
    """
    Sparse cross-message token attention for reference resolution.
    Connects important tokens across messages (e.g., pronouns to antecedents).
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
        self.top_k_ratio = top_k_ratio
        self.temperature = temperature
        
        # Attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Token importance scorer (which tokens should look across messages)
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
                message_boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sparse cross-message attention (vectorized implementation).
        
        Args:
            token_features: [total_tokens, hidden_dim] all tokens from all messages
            message_boundaries: [num_messages, 2] start/end indices for each message
            
        Returns:
            updated_features: [total_tokens, hidden_dim]
            cross_attention_edges: [2, num_edges] edges that were created
        """
        device = token_features.device
        num_messages = len(message_boundaries)
        
        # Score token importance
        importance_scores = self.importance_scorer(token_features).squeeze(-1)
        
        # Build message masks efficiently
        message_lengths = [end - start for start, end in message_boundaries]
        max_msg_len = max(message_lengths)
        
        # Collect all important tokens at once
        all_important_indices = []
        all_message_ids = []
        
        for msg_idx in range(num_messages):
            start_idx, end_idx = message_boundaries[msg_idx]
            if end_idx > start_idx:
                msg_scores = importance_scores[start_idx:end_idx]
                k = max(1, int((end_idx - start_idx) * self.top_k_ratio))
                k = min(k, end_idx - start_idx)
                
                if k > 0:
                    top_k_values, top_k_indices = torch.topk(msg_scores, k)
                    global_indices = top_k_indices + start_idx
                    all_important_indices.append(global_indices)
                    all_message_ids.extend([msg_idx] * k)
        
        if not all_important_indices:
            return token_features, torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Concatenate all important indices
        important_indices = torch.cat(all_important_indices)
        message_ids = torch.tensor(all_message_ids, device=device)
        
        # Project all tokens once
        all_keys = self.key_proj(token_features)
        all_values = self.value_proj(token_features)
        query_tokens = self.query_proj(token_features[important_indices])
        
        # Create token-to-message mapping
        token_to_message = torch.zeros(token_features.size(0), dtype=torch.long, device=device)
        for msg_idx, (start, end) in enumerate(message_boundaries):
            token_to_message[start:end] = msg_idx
        
        # Compute all attention scores at once
        # [num_important, hidden_dim] x [hidden_dim, total_tokens] -> [num_important, total_tokens]
        attention_scores = torch.matmul(query_tokens, all_keys.T) / (self.hidden_dim ** 0.5)
        attention_scores = attention_scores / self.temperature
        
        # Mask same-message tokens
        query_messages = message_ids.unsqueeze(1)  # [num_important, 1]
        token_messages = token_to_message.unsqueeze(0)  # [1, total_tokens]
        same_message_mask = query_messages == token_messages  # [num_important, total_tokens]
        attention_scores.masked_fill_(same_message_mask, float('-inf'))
        
        # Get top-k connections per query token
        k_connections = min(5, token_features.size(0) // num_messages)
        top_scores, top_indices = torch.topk(attention_scores, k_connections, dim=1)
        
        # Apply softmax only on top-k
        attention_weights = F.softmax(top_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Gather values and compute updates
        # [num_important, k_connections, hidden_dim]
        gathered_values = all_values[top_indices]
        
        # [num_important, k_connections, 1] * [num_important, k_connections, hidden_dim]
        weighted_values = attention_weights.unsqueeze(-1) * gathered_values
        
        # Sum over k_connections: [num_important, hidden_dim]
        attended_values = weighted_values.sum(dim=1)
        
        # Apply output projection
        updates = self.output_proj(attended_values)
        
        # Update features with residual connection
        updated_features = token_features.clone()
        updated_features[important_indices] += self.dropout(updates)
        
        # Create sparse edges for visualization
        edge_threshold = 0.1
        edge_mask = attention_weights > edge_threshold
        edges = []
        
        for i in range(len(important_indices)):
            src_idx = important_indices[i].item()
            for j in range(k_connections):
                if edge_mask[i, j]:
                    tgt_idx = top_indices[i, j].item()
                    edges.append([src_idx, tgt_idx])
        
        if edges:
            edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).T
        else:
            edge_tensor = torch.empty((2, 0), dtype=torch.long, device=device)
            
        return updated_features, edge_tensor