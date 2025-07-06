#!/usr/bin/env python3
"""
Visualization utilities for model analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_attention_heatmap(attention_weights: torch.Tensor,
                          query_labels: List[str],
                          context_labels: List[str],
                          save_path: Optional[Path] = None,
                          title: str = "Attention Weights"):
    """
    Plot attention weight heatmap
    
    Args:
        attention_weights: [num_queries, num_contexts] attention matrix
        query_labels: Labels for queries (y-axis)
        context_labels: Labels for contexts (x-axis)
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy if needed
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=context_labels,
        yticklabels=query_labels,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        annot=attention_weights.shape[0] <= 20,  # Annotate if small
        fmt='.2f'
    )
    
    plt.title(title)
    plt.xlabel('Context Messages')
    plt.ylabel('Query Messages')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_token_importance(token_scores: Dict[int, torch.Tensor],
                         token_texts: Dict[int, List[str]],
                         save_path: Optional[Path] = None,
                         top_k: int = 10):
    """
    Plot most important tokens per message
    
    Args:
        token_scores: Dict mapping message idx to token importance scores
        token_texts: Dict mapping message idx to token texts
        save_path: Path to save figure
        top_k: Show top k tokens per message
    """
    num_messages = len(token_scores)
    fig, axes = plt.subplots(
        (num_messages + 2) // 3, 3, 
        figsize=(15, 4 * ((num_messages + 2) // 3))
    )
    axes = axes.flatten() if num_messages > 1 else [axes]
    
    for idx, (msg_idx, scores) in enumerate(token_scores.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        tokens = token_texts[msg_idx]
        
        # Get top tokens
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # Handle multi-dimensional scores by flattening or taking mean
        if scores.ndim > 1:
            scores = scores.mean(axis=0) if scores.ndim == 2 else scores.flatten()
        
        # Ensure we don't exceed available tokens
        actual_top_k = min(top_k, len(tokens), len(scores))
        
        top_indices = np.argsort(scores)[-actual_top_k:][::-1]
        top_tokens = [tokens[i] for i in top_indices if i < len(tokens)]
        top_scores = [scores[i] for i in top_indices if i < len(tokens)]
        
        # Plot
        ax.barh(range(len(top_tokens)), top_scores)
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Message {msg_idx} - Top Tokens')
        ax.invert_yaxis()
    
    # Hide unused subplots
    for idx in range(len(token_scores), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_conversation_graph(edge_index: torch.Tensor,
                          node_labels: List[str],
                          edge_weights: Optional[torch.Tensor] = None,
                          node_colors: Optional[List[str]] = None,
                          save_path: Optional[Path] = None,
                          title: str = "Conversation Graph"):
    """
    Visualize conversation graph structure
    
    Args:
        edge_index: [2, num_edges] edge indices
        node_labels: Labels for each node
        edge_weights: Optional edge weights for visualization
        node_colors: Optional colors for nodes
        save_path: Path to save figure
        title: Plot title
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX required for graph visualization")
        return
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label)
    
    # Add edges
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        weight = edge_weights[i].item() if edge_weights is not None else 1.0
        G.add_edge(src, dst, weight=weight)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    node_colors = node_colors or ['lightblue'] * len(node_labels)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.9)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges] if edge_weights is not None else None
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          width=weights, alpha=0.6, arrowsize=20)
    
    # Draw labels
    labels = {i: f"{i}: {label[:20]}..." if len(label) > 20 else f"{i}: {label}" 
              for i, label in enumerate(node_labels)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_training_curves(history: Dict[str, List[float]],
                        save_path: Optional[Path] = None):
    """
    Plot training history curves
    
    Args:
        history: Dictionary of metric histories
        save_path: Path to save figure
    """
    # Determine layout
    metrics = list(history.keys())
    num_metrics = len(metrics)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (metric, values) in enumerate(history.items()):
        ax = axes[idx]
        
        # Plot curve
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker='o', markersize=4)
        
        # Formatting
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Add trend line for loss metrics
        if 'loss' in metric:
            z = np.polyfit(epochs, values, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "r--", alpha=0.5, label=f'Trend: {z[0]:.4f}x')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Training History', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_retrieval_metrics(metrics: Dict[str, float],
                          by_type: Optional[Dict[str, Dict[str, float]]] = None,
                          save_path: Optional[Path] = None):
    """
    Plot retrieval evaluation metrics
    
    Args:
        metrics: Overall metrics dictionary
        by_type: Metrics broken down by dependency type
        save_path: Path to save figure
    """
    if by_type:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None
    
    # Overall metrics bar plot
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax1.bar(metric_names, metric_values)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Retrieval Metrics')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Metrics by dependency type
    if by_type and ax2:
        # Prepare data
        dep_types = list(by_type.keys())
        metrics_to_plot = ['precision@5', 'recall@5', 'mrr']
        
        x = np.arange(len(dep_types))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [by_type[dt].get(metric, 0) for dt in dep_types]
            ax2.bar(x + i * width - width, values, width, label=metric)
        
        ax2.set_xlabel('Dependency Type')
        ax2.set_ylabel('Score')
        ax2.set_title('Metrics by Dependency Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dep_types, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()