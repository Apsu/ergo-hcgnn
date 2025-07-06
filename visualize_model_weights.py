#!/usr/bin/env python3
"""
Visualize model weight distributions and learned representations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict

from models.hierarchical_gnn import HierarchicalConversationGNN


def visualize_weight_distributions(model, output_dir):
    """Visualize weight distributions across different layers"""
    output_dir.mkdir(exist_ok=True)
    
    # Collect weights by layer type
    weights_by_type = defaultdict(list)
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            # Categorize by layer type
            if 'token_embed' in name:
                layer_type = 'Token Embeddings'
            elif 'position_embed' in name:
                layer_type = 'Position Embeddings'
            elif 'role_embed' in name:
                layer_type = 'Role Embeddings'
            elif 'gat' in name.lower() or 'attention' in name.lower():
                layer_type = 'Attention Layers'
            elif 'linear' in name or 'fc' in name:
                layer_type = 'Linear Layers'
            elif 'norm' in name:
                layer_type = 'Normalization'
            else:
                layer_type = 'Other'
            
            weights_by_type[layer_type].append({
                'name': name,
                'weights': param.detach().cpu().numpy().flatten()
            })
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (layer_type, weights_list) in enumerate(weights_by_type.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Combine all weights for this layer type
        all_weights = np.concatenate([w['weights'] for w in weights_list])
        
        # Plot histogram
        ax.hist(all_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f'{layer_type}\n({len(weights_list)} layers, {len(all_weights):,} params)')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        
        # Add statistics
        stats_text = f'μ={np.mean(all_weights):.3f}\nσ={np.std(all_weights):.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(weights_by_type), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Weight Distribution Analysis by Layer Type', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_distributions.png', dpi=150)
    plt.close()
    
    # Create detailed weight magnitude analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layer_names = []
    weight_magnitudes = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            layer_names.append(name.replace('model.', ''))
            weight_magnitudes.append(np.abs(param.detach().cpu().numpy()).mean())
    
    # Sort by magnitude
    sorted_indices = np.argsort(weight_magnitudes)[::-1]
    layer_names = [layer_names[i] for i in sorted_indices]
    weight_magnitudes = [weight_magnitudes[i] for i in sorted_indices]
    
    # Plot top 20 layers
    top_n = min(20, len(layer_names))
    y_pos = np.arange(top_n)
    
    ax.barh(y_pos, weight_magnitudes[:top_n], color='green', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names[:top_n])
    ax.set_xlabel('Average Weight Magnitude')
    ax.set_title('Top 20 Layers by Average Weight Magnitude', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_magnitudes.png', dpi=150)
    plt.close()


def visualize_embedding_relationships(model, output_dir):
    """Visualize relationships between different embedding types"""
    output_dir.mkdir(exist_ok=True)
    
    # Get embeddings
    token_emb = model.token_embeddings.weight.detach().cpu().numpy()
    pos_emb = model.position_embeddings.weight.detach().cpu().numpy()
    
    # Role embeddings if they exist
    role_emb = None
    if hasattr(model, 'role_embeddings'):
        role_emb = model.role_embeddings.weight.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Token embedding similarity matrix (sample)
    ax = axes[0, 0]
    sample_size = min(100, token_emb.shape[0])
    sample_tokens = token_emb[:sample_size]
    token_sim = np.corrcoef(sample_tokens)
    
    sns.heatmap(token_sim, cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'label': 'Correlation'})
    ax.set_title(f'Token Embedding Similarities (first {sample_size})')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Token Index')
    
    # Position embedding patterns
    ax = axes[0, 1]
    pos_sim = np.corrcoef(pos_emb.T)  # Correlation between dimensions
    
    sns.heatmap(pos_sim, cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Position Embedding Dimension Correlations')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Embedding Dimension')
    
    # Embedding dimension statistics
    ax = axes[1, 0]
    token_std = np.std(token_emb, axis=0)
    pos_std = np.std(pos_emb, axis=0)
    
    x = np.arange(len(token_std))
    ax.plot(x, token_std, label='Token Embeddings', alpha=0.7)
    ax.plot(x, pos_std, label='Position Embeddings', alpha=0.7)
    if role_emb is not None:
        role_std = np.std(role_emb, axis=0)
        ax.plot(x, role_std, label='Role Embeddings', alpha=0.7)
    
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Variance by Embedding Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average embedding magnitudes
    ax = axes[1, 1]
    embeddings_info = [
        ('Token', np.linalg.norm(token_emb, axis=1).mean()),
        ('Position', np.linalg.norm(pos_emb, axis=1).mean())
    ]
    if role_emb is not None:
        embeddings_info.append(('Role', np.linalg.norm(role_emb, axis=1).mean()))
    
    names, magnitudes = zip(*embeddings_info)
    ax.bar(names, magnitudes, color=['blue', 'green', 'red'][:len(names)], alpha=0.7)
    ax.set_ylabel('Average L2 Norm')
    ax.set_title('Average Embedding Magnitudes')
    
    plt.suptitle('Embedding Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_analysis.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model weights and representations')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='model_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get model configuration
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
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
    model.eval()
    
    output_dir = Path(args.output_dir)
    
    print("\nVisualizing weight distributions...")
    visualize_weight_distributions(model, output_dir)
    
    print("\nVisualizing embedding relationships...")
    visualize_embedding_relationships(model, output_dir)
    
    print(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()