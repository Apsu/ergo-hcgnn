#!/usr/bin/env python3
"""
Graph Neural Network layers for token and message processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from typing import Optional, Tuple


class ImprovedGATConv(MessagePassing):
    """
    Improved GAT layer with edge features and residual connections
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 4,
                 dropout: float = 0.1,
                 edge_dim: Optional[int] = None,
                 residual: bool = True,
                 normalize: bool = True):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.residual = residual
        self.normalize = normalize
        
        # Multi-head projections
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # Edge feature projection
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        else:
            self.lin_edge = None
            
        # Output projection
        self.lin_out = nn.Linear(heads * out_channels, out_channels)
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Residual connection
        if residual:
            self.lin_residual = nn.Linear(in_channels, out_channels)
        
        # Normalization
        if normalize:
            self.norm = nn.LayerNorm(out_channels)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        H, C = self.heads, self.out_channels
        
        # Add self-loops
        num_nodes = x.size(0)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_nodes)
        
        # Linear transformations
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        
        # Propagate
        out = self.propagate(edge_index, query=query, key=key, value=value,
                           edge_attr=edge_attr, size=None)
        
        # Concatenate heads and project
        out = out.view(-1, H * C)
        out = self.lin_out(out)
        
        # Residual connection
        if self.residual:
            res = self.lin_residual(x)
            out = out + res
            
        # Normalization
        if self.normalize:
            out = self.norm(out)
            
        return out
        
    def message(self, query_i: torch.Tensor, key_j: torch.Tensor, value_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor], index: torch.Tensor,
                ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor:
        """Compute messages"""
        # Compute attention scores
        alpha = (query_i * self.att_src).sum(dim=-1) + (key_j * self.att_dst).sum(dim=-1)
        
        # Add edge features if available
        if self.lin_edge is not None and edge_attr is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_feat * self.att_dst).sum(dim=-1)
        
        # Softmax
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Return weighted values
        return value_j * alpha.unsqueeze(-1)


class TokenGraphBuilder(nn.Module):
    """
    Builds token-level graphs with multiple edge types
    """
    
    def __init__(self,
                 window_sizes: list = [1, 2, 3],
                 add_syntax_edges: bool = False,
                 add_similarity_edges: bool = False,
                 similarity_threshold: float = 0.8):
        super().__init__()
        
        self.window_sizes = window_sizes
        self.add_syntax_edges = add_syntax_edges
        self.add_similarity_edges = add_similarity_edges
        self.similarity_threshold = similarity_threshold
        
        # Edge type embeddings
        self.num_edge_types = len(window_sizes) + 2  # +2 for syntax and similarity
        self.edge_embeddings = nn.Embedding(self.num_edge_types, 64)
        
    def forward(self, 
                token_ids: torch.Tensor,
                token_embeddings: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build token graph
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            token_embeddings: Optional embeddings for similarity edges
            
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 64]
        """
        batch_size, seq_len = token_ids.shape
        edges = []
        edge_types = []
        
        # Add edges for each window size
        for edge_type, window in enumerate(self.window_sizes):
            for i in range(seq_len - window):
                edges.append([i, i + window])
                edges.append([i + window, i])  # Bidirectional
                edge_types.extend([edge_type, edge_type])
        
        # Add self-loops
        for i in range(seq_len):
            edges.append([i, i])
            edge_types.append(0)  # Self-loop uses type 0
        
        # Convert to tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).T.to(token_ids.device)
            edge_types = torch.tensor(edge_types, dtype=torch.long).to(token_ids.device)
            edge_attr = self.edge_embeddings(edge_types)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(token_ids.device)
            edge_attr = torch.empty((0, 64), dtype=torch.float).to(token_ids.device)
            
        return edge_index, edge_attr


class MessageGraphBuilder(nn.Module):
    """
    Builds message-level graphs with temporal and optional semantic edges
    """
    
    def __init__(self,
                 use_semantic_edges: bool = False,
                 semantic_threshold: float = 0.7,
                 max_semantic_edges: int = 5):
        super().__init__()
        
        self.use_semantic_edges = use_semantic_edges
        self.semantic_threshold = semantic_threshold
        self.max_semantic_edges = max_semantic_edges
        
        # Edge type embeddings (temporal, semantic)
        self.edge_embeddings = nn.Embedding(2, 64)
        
    def forward(self,
                message_embeddings: torch.Tensor,
                add_backward_edges: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build message graph
        
        Args:
            message_embeddings: [num_messages, embedding_dim]
            add_backward_edges: Whether to add backward temporal edges
            
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 64]
        """
        num_messages = message_embeddings.size(0)
        edges = []
        edge_types = []
        
        # Temporal edges
        for i in range(num_messages - 1):
            edges.append([i, i + 1])
            edge_types.append(0)  # Temporal forward
            
            if add_backward_edges:
                edges.append([i + 1, i])
                edge_types.append(0)  # Temporal backward
        
        # Semantic edges if enabled
        if self.use_semantic_edges and num_messages > 2:
            # Compute pairwise similarities
            embeddings_norm = F.normalize(message_embeddings, p=2, dim=1)
            similarities = torch.matmul(embeddings_norm, embeddings_norm.T)
            
            # Mask out self and adjacent messages
            mask = torch.ones_like(similarities, dtype=torch.bool)
            mask.fill_diagonal_(False)
            for i in range(num_messages - 1):
                mask[i, i + 1] = False
                mask[i + 1, i] = False
            
            # Find high similarity pairs
            similarities = similarities * mask
            
            for i in range(num_messages):
                # Get top-k similar messages for each message
                if i > 0:  # Only look backward
                    sim_scores = similarities[i, :i]
                    if sim_scores.numel() > 0:
                        k = min(self.max_semantic_edges, sim_scores.numel())
                        top_k_values, top_k_indices = torch.topk(sim_scores, k)
                        
                        for val, idx in zip(top_k_values, top_k_indices):
                            if val > self.semantic_threshold:
                                edges.append([idx.item(), i])
                                edge_types.append(1)  # Semantic edge
        
        # Convert to tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).T.to(message_embeddings.device)
            edge_types = torch.tensor(edge_types, dtype=torch.long).to(message_embeddings.device)
            edge_attr = self.edge_embeddings(edge_types)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(message_embeddings.device)
            edge_attr = torch.empty((0, 64), dtype=torch.float).to(message_embeddings.device)
            
        return edge_index, edge_attr