#!/usr/bin/env python3
"""
Hierarchical Conversation GNN Model V2
Complete implementation with architectural improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple

from layers.token_attention import TokenToMessageAttention, CrossMessageTokenAttention
from layers.graph_layers import ImprovedGATConv, TokenGraphBuilder, MessageGraphBuilder


class HierarchicalConversationGNN(nn.Module):
    """
    Two-level hierarchical GNN for conversation understanding
    with improved token-to-message attention and cross-message connections
    """
    
    def __init__(self,
                 vocab_size: int = 30522,
                 token_embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 message_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 max_seq_length: int = 512,
                 num_token_gat_layers: int = 2,
                 num_message_gat_layers: int = 2,
                 use_cross_message_attention: bool = True,
                 use_semantic_edges: bool = True,
                 window_sizes: List[int] = [1, 2, 3],
                 num_dependency_types: int = 8):
        """
        Args:
            vocab_size: Size of token vocabulary
            token_embedding_dim: Dimension of learned token embeddings
            hidden_dim: Hidden dimension for GNN layers
            message_dim: Final message embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length for position embeddings
            num_token_gat_layers: Number of token-level GAT layers
            num_message_gat_layers: Number of message-level GAT layers
            use_cross_message_attention: Whether to use cross-message token attention
            use_semantic_edges: Whether to add semantic edges in message graph
            window_sizes: Window sizes for token graph connections
            num_dependency_types: Number of dependency types for adaptive temperature
        """
        super().__init__()
        
        self.token_embedding_dim = token_embedding_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.dropout = dropout
        self.use_cross_message_attention = use_cross_message_attention
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, token_embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, token_embedding_dim)
        
        # Learnable segment embeddings for role information
        self.segment_embeddings = nn.Embedding(4, token_embedding_dim)  # user, assistant, system, tool
        
        # Layer normalization and dropout
        self.embedding_layer_norm = nn.LayerNorm(token_embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Token graph builder
        self.token_graph_builder = TokenGraphBuilder(
            window_sizes=window_sizes,
            add_syntax_edges=False,  # Could enable with dependency parser
            add_similarity_edges=False  # Could enable for semantic connections
        )
        
        # Token-level GAT layers
        self.token_gat_layers = nn.ModuleList()
        in_dim = token_embedding_dim
        for i in range(num_token_gat_layers):
            out_dim = hidden_dim if i < num_token_gat_layers - 1 else hidden_dim
            self.token_gat_layers.append(
                ImprovedGATConv(
                    in_dim, out_dim, 
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=64,  # From graph builder
                    residual=(i > 0),  # Residual from second layer
                    normalize=True
                )
            )
            in_dim = out_dim
        
        # Cross-message token attention
        if use_cross_message_attention:
            self.cross_message_attention = CrossMessageTokenAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                top_k_ratio=0.1,
                temperature=1.0
            )
        
        # Token to message attention (replaces pooling)
        self.token_to_message = TokenToMessageAttention(
            token_dim=hidden_dim,
            message_dim=message_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_weights=True
        )
        
        # Message graph builder
        self.message_graph_builder = MessageGraphBuilder(
            use_semantic_edges=use_semantic_edges,
            semantic_threshold=0.7,
            max_semantic_edges=5
        )
        
        # Message-level processing
        self.message_encoder = nn.Sequential(
            nn.Linear(message_dim + 2, message_dim),  # +2 for role and position
            nn.LayerNorm(message_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Message-level GAT layers
        self.message_gat_layers = nn.ModuleList()
        in_dim = message_dim
        for i in range(num_message_gat_layers):
            self.message_gat_layers.append(
                ImprovedGATConv(
                    in_dim, message_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=64,
                    residual=(i > 0),
                    normalize=True
                )
            )
        
        # Relevance scoring with position awareness
        self.relevance_scorer = nn.Sequential(
            nn.Linear(message_dim * 2 + 64, hidden_dim),  # +64 for position encoding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Relative position encoding for scoring
        self.max_relative_position = 32
        self.relative_position_embedding = nn.Embedding(
            2 * self.max_relative_position + 1, 64
        )
        
        # Learnable temperature parameters per dependency type
        self.dependency_temperatures = nn.Parameter(torch.ones(num_dependency_types) * 1.5)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self,
                token_ids_list: List[torch.Tensor],
                lengths: List[int],
                message_edge_index: torch.Tensor,
                message_node_attr: torch.Tensor,
                segment_ids_list: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through hierarchical GNN
        
        Args:
            token_ids_list: List of token ID tensors for each message
            lengths: List of actual lengths for each message
            message_edge_index: Message-level graph edges [2, num_edges]
            message_node_attr: Message attributes (role, position) [num_messages, 2]
            segment_ids_list: Optional segment IDs for role information
            
        Returns:
            message_embeddings: [num_messages, message_dim]
        """
        # Process tokens for each message and build graphs
        token_graphs = []
        message_boundaries = []
        current_pos = 0
        
        for i, (token_ids, length) in enumerate(zip(token_ids_list, lengths)):
            # Create token embeddings
            token_emb = self.token_embeddings(token_ids[:length])
            
            # Add position embeddings
            positions = torch.arange(length, device=token_ids.device)
            pos_emb = self.position_embeddings(positions)
            
            # Add segment embeddings if provided
            if segment_ids_list is not None:
                seg_emb = self.segment_embeddings(segment_ids_list[i][:length])
                embeddings = token_emb + pos_emb + seg_emb
            else:
                # Use role from message attributes
                role = int(message_node_attr[i, 0].item())
                seg_emb = self.segment_embeddings(torch.full((length,), role, device=token_ids.device))
                embeddings = token_emb + pos_emb + seg_emb
            
            # Layer norm and dropout
            embeddings = self.embedding_layer_norm(embeddings)
            embeddings = self.embedding_dropout(embeddings)
            
            # Build token graph
            edge_index, edge_attr = self.token_graph_builder(token_ids[:length].unsqueeze(0))
            
            # Create PyG data object
            token_graph = Data(
                x=embeddings,
                edge_index=edge_index,
                edge_attr=edge_attr,
                length=length,
                message_idx=i
            )
            token_graphs.append(token_graph)
            
            # Track boundaries for cross-message attention
            message_boundaries.append([current_pos, current_pos + length])
            current_pos += length
        
        # Batch token graphs
        token_batch = Batch.from_data_list(token_graphs)
        
        # Process through token GAT layers
        x = token_batch.x
        for gat_layer in self.token_gat_layers:
            x = gat_layer(x, token_batch.edge_index, token_batch.edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply cross-message attention if enabled
        if self.use_cross_message_attention and len(message_boundaries) > 1:
            boundaries_tensor = torch.tensor(message_boundaries, device=x.device)
            x, cross_edges = self.cross_message_attention(x, boundaries_tensor)
        
        # Convert tokens to messages using attention
        message_embeddings = []
        token_attention_weights = []
        
        for i, graph in enumerate(token_graphs):
            # Get tokens for this message
            mask = token_batch.batch == i
            message_tokens = x[mask]
            
            # Create position information for attention
            positions = torch.arange(graph.length, device=x.device).float() / graph.length
            
            # Apply token-to-message attention
            message_emb, attention_weights = self.token_to_message(
                message_tokens.unsqueeze(0),
                token_mask=torch.ones(1, graph.length, device=x.device, dtype=torch.bool),
                token_positions=positions.unsqueeze(0)
            )
            
            message_embeddings.append(message_emb.squeeze(0))
            token_attention_weights.append(attention_weights)
        
        # Stack message embeddings
        message_embeddings = torch.stack(message_embeddings)
        
        # Add message-level features
        message_features = torch.cat([message_embeddings, message_node_attr], dim=-1)
        message_features = self.message_encoder(message_features)
        
        # Build message graph with optional semantic edges
        msg_edge_index, msg_edge_attr = self.message_graph_builder(
            message_features, add_backward_edges=True
        )
        
        # Process through message GAT layers
        x = message_features
        for gat_layer in self.message_gat_layers:
            x = gat_layer(x, msg_edge_index, msg_edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store attention weights for visualization
        self.last_token_attention_weights = token_attention_weights
        
        return x
    
    def score_relevance(self,
                       query_embedding: torch.Tensor,
                       context_embedding: torch.Tensor,
                       query_idx: Optional[int] = None,
                       context_idx: Optional[int] = None,
                       dependency_type_idx: Optional[int] = None) -> torch.Tensor:
        """
        Score relevance between query and context messages
        
        Args:
            query_embedding: [1, message_dim] or [batch, message_dim]
            context_embedding: [1, message_dim] or [batch, message_dim]
            query_idx: Position index of query message
            context_idx: Position index of context message
            dependency_type_idx: Index of dependency type for temperature
            
        Returns:
            relevance_score: [1] or [batch, 1]
        """
        # Get relative position encoding
        if query_idx is not None and context_idx is not None:
            relative_pos = query_idx - context_idx
            relative_pos = max(-self.max_relative_position,
                             min(self.max_relative_position, relative_pos))
            relative_pos_idx = relative_pos + self.max_relative_position
            
            pos_encoding = self.relative_position_embedding(
                torch.tensor([relative_pos_idx], device=query_embedding.device)
            )
            
            # Expand for batch if needed
            if query_embedding.dim() == 2 and query_embedding.size(0) > 1:
                pos_encoding = pos_encoding.expand(query_embedding.size(0), -1)
        else:
            # Zero position encoding if not provided
            batch_size = query_embedding.size(0) if query_embedding.dim() == 2 else 1
            pos_encoding = torch.zeros(batch_size, 64, device=query_embedding.device)
        
        # Concatenate features
        combined = torch.cat([query_embedding, context_embedding, pos_encoding], dim=-1)
        
        # Score relevance
        score = self.relevance_scorer(combined)
        
        # Apply temperature if dependency type provided
        if dependency_type_idx is not None and dependency_type_idx < len(self.dependency_temperatures):
            temperature = F.softplus(self.dependency_temperatures[dependency_type_idx])
        else:
            temperature = 1.5
            
        return score / temperature
    
    def get_token_importance(self, message_idx: int) -> torch.Tensor:
        """
        Get token importance scores from last forward pass
        
        Args:
            message_idx: Index of message
            
        Returns:
            importance_scores: [num_tokens] averaged across heads
        """
        if hasattr(self, 'last_token_attention_weights') and message_idx < len(self.last_token_attention_weights):
            # Average across attention heads
            return self.last_token_attention_weights[message_idx].mean(dim=0)
        else:
            return None