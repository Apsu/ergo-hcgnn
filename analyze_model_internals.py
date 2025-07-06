#!/usr/bin/env python3
"""
Analyze and visualize model internals for Hierarchical Conversation GNN V2
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F

from models.hierarchical_gnn import HierarchicalConversationGNN


def analyze_token_embeddings(model, vocab_size_limit=1000, output_dir=Path("model_analysis")):
    """Analyze learned token embeddings"""
    output_dir.mkdir(exist_ok=True)
    
    # Get token embeddings
    token_embeddings = model.token_embeddings.weight.detach().cpu().numpy()
    
    # Sample for visualization if vocab is large
    if token_embeddings.shape[0] > vocab_size_limit:
        indices = np.random.choice(token_embeddings.shape[0], vocab_size_limit, replace=False)
        token_embeddings = token_embeddings[indices]
    
    print(f"Analyzing {token_embeddings.shape[0]} token embeddings of dimension {token_embeddings.shape[1]}")
    
    # Compute similarity matrix
    similarities = np.corrcoef(token_embeddings)
    
    # Plot similarity heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarities[:100, :100], cmap='coolwarm', center=0, 
                square=True, cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Token Embedding Similarity Matrix (First 100 tokens)')
    plt.tight_layout()
    plt.savefig(output_dir / 'token_similarity_matrix.png', dpi=150)
    plt.close()
    
    # t-SNE visualization
    print("Computing t-SNE for token embeddings...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(token_embeddings)-1))
    embeddings_2d = tsne.fit_transform(token_embeddings)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=range(len(embeddings_2d)), cmap='viridis', 
                         alpha=0.6, s=20)
    plt.colorbar(scatter, label='Token Index')
    plt.title('Token Embeddings t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'token_embeddings_tsne.png', dpi=150)
    plt.close()


def analyze_attention_heads(model, output_dir=Path("model_analysis")):
    """Analyze attention head specialization"""
    output_dir.mkdir(exist_ok=True)
    
    # Check if model has GAT layers
    gat_layers = []
    for name, module in model.named_modules():
        if 'GAT' in type(module).__name__ or 'gat' in name.lower():
            gat_layers.append((name, module))
    
    if not gat_layers:
        print("No GAT layers found in model")
        return
    
    print(f"Found {len(gat_layers)} GAT layers")
    
    # Analyze attention patterns for each layer (limit to first 5 for visualization)
    num_layers_to_plot = min(5, len(gat_layers))
    fig, axes = plt.subplots(num_layers_to_plot, 2, figsize=(12, 4*num_layers_to_plot))
    if num_layers_to_plot == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (layer_name, layer) in enumerate(gat_layers[:num_layers_to_plot]):
        # Try to get attention weights if available
        if hasattr(layer, 'att_src') and layer.att_src is not None:
            att_weights = layer.att_src.detach().cpu().numpy()
            
            # Handle different weight shapes
            if att_weights.ndim == 3:
                # If 3D, take the first head or average across heads
                att_weights = att_weights[0] if att_weights.shape[0] == 1 else att_weights.mean(axis=0)
            elif att_weights.ndim == 1:
                # If 1D, reshape to 2D
                att_weights = att_weights.reshape(-1, 1)
            
            # Plot weight distribution
            ax = axes[idx, 0]
            ax.hist(att_weights.flatten(), bins=50, alpha=0.7, color='blue')
            ax.set_title(f'{layer_name} - Weight Distribution')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            
            # Plot weight matrix heatmap (if 2D)
            ax = axes[idx, 1]
            if att_weights.ndim == 2 and att_weights.shape[1] > 1:
                sns.heatmap(att_weights[:min(50, att_weights.shape[0])], 
                           cmap='coolwarm', center=0, ax=ax,
                           cbar_kws={'label': 'Weight Value'})
                ax.set_title(f'{layer_name} - Weight Matrix')
            else:
                # If 1D or single column, plot as line
                ax.plot(att_weights[:min(100, len(att_weights))].flatten())
                ax.set_title(f'{layer_name} - Weight Values')
                ax.set_xlabel('Index')
                ax.set_ylabel('Weight Value')
    
    plt.suptitle('GAT Layer Attention Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_head_analysis.png', dpi=150)
    plt.close()


def analyze_message_embeddings(model, sample_messages, output_dir=Path("model_analysis")):
    """Analyze how different message types are embedded"""
    output_dir.mkdir(exist_ok=True)
    
    # Create sample embeddings for different message types
    message_types = {
        'question': ["What is the weather like?", "How can I help you?", "Where is the nearest restaurant?"],
        'answer': ["The weather is sunny today.", "I can help you with that.", "The restaurant is two blocks away."],
        'statement': ["I think that's correct.", "This seems interesting.", "The project is complete."],
        'clarification': ["What do you mean by that?", "Could you explain further?", "I don't understand."]
    }
    
    embeddings_by_type = {}
    model.eval()
    
    with torch.no_grad():
        for msg_type, messages in message_types.items():
            type_embeddings = []
            for msg in messages:
                # Simple tokenization (in practice, use the actual tokenizer)
                tokens = msg.lower().split()
                vocab_size = model.token_embeddings.weight.shape[0]
                token_ids = torch.randint(0, vocab_size, (len(tokens),))
                
                # Get token embeddings
                token_embs = model.token_embeddings(token_ids)
                # Simple mean pooling
                msg_embedding = token_embs.mean(dim=0)
                type_embeddings.append(msg_embedding.cpu().numpy())
            
            embeddings_by_type[msg_type] = np.array(type_embeddings)
    
    # Visualize embeddings by type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compute centroids
    centroids = {}
    all_embeddings = []
    all_labels = []
    
    for msg_type, embeddings in embeddings_by_type.items():
        centroids[msg_type] = embeddings.mean(axis=0)
        all_embeddings.extend(embeddings)
        all_labels.extend([msg_type] * len(embeddings))
    
    all_embeddings = np.array(all_embeddings)
    
    # PCA visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    centroids_2d = pca.transform(np.array(list(centroids.values())))
    
    colors = {'question': 'blue', 'answer': 'green', 'statement': 'orange', 'clarification': 'red'}
    
    for i, (msg_type, color) in enumerate(colors.items()):
        mask = np.array(all_labels) == msg_type
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=color, label=msg_type, alpha=0.6, s=100)
    
    # Plot centroids
    for i, (msg_type, color) in enumerate(colors.items()):
        ax1.scatter(centroids_2d[i, 0], centroids_2d[i, 1], 
                   c=color, marker='*', s=500, edgecolors='black', linewidth=2)
    
    ax1.set_title('Message Type Embeddings (PCA)')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.legend()
    
    # Distance matrix between centroids
    num_types = len(centroids)
    distance_matrix = np.zeros((num_types, num_types))
    type_names = list(centroids.keys())
    
    for i, type1 in enumerate(type_names):
        for j, type2 in enumerate(type_names):
            distance_matrix[i, j] = np.linalg.norm(centroids[type1] - centroids[type2])
    
    sns.heatmap(distance_matrix, xticklabels=type_names, yticklabels=type_names,
                annot=True, fmt='.2f', cmap='viridis', ax=ax2,
                cbar_kws={'label': 'Euclidean Distance'})
    ax2.set_title('Distance Between Message Type Centroids')
    
    plt.suptitle('Message Type Embedding Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'message_type_analysis.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze Hierarchical Conversation GNN V2 internals')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='model_analysis', help='Output directory')
    parser.add_argument('--vocab-limit', type=int, default=1000, help='Limit for vocab analysis')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get model configuration
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
    model.eval()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\nAnalyzing token embeddings...")
    analyze_token_embeddings(model, args.vocab_limit, output_dir)
    
    print("\nAnalyzing attention heads...")
    analyze_attention_heads(model, output_dir)
    
    print("\nAnalyzing message embeddings...")
    analyze_message_embeddings(model, None, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()