#!/usr/bin/env python3
"""
Advanced visualization utilities for Hierarchical Conversation GNN V2
Provides sophisticated analysis plots and model introspection tools
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def plot_embedding_analysis(embeddings: torch.Tensor,
                          metadata: List[Dict[str, Any]],
                          save_path: Optional[Path] = None,
                          title: str = "Message Embedding Analysis"):
    """
    Create comprehensive embedding space visualizations
    
    Args:
        embeddings: [num_messages, embedding_dim] tensor of message embeddings
        metadata: List of dicts with keys: role, dependency_type, position, text_length
        save_path: Path to save figure
        title: Overall figure title
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    
    # Dimensionality reduction
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), 
                random_state=42)
    embeddings_2d_tsne = tsne.fit_transform(embeddings)
    
    print("Computing PCA...")
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. t-SNE colored by role
    ax1 = fig.add_subplot(gs[0, 0])
    roles = [m.get('role', 'unknown') for m in metadata]
    role_colors = {'user': '#3498db', 'assistant': '#e74c3c', 'system': '#2ecc71'}
    colors = [role_colors.get(r, '#95a5a6') for r in roles]
    
    scatter = ax1.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                         c=colors, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax1.set_title('t-SNE - Colored by Role', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Add legend
    handles = [mpatches.Patch(color=color, label=role) 
              for role, color in role_colors.items()]
    ax1.legend(handles=handles, loc='best', frameon=True, fancybox=True)
    
    # 2. t-SNE colored by dependency type
    ax2 = fig.add_subplot(gs[0, 1])
    dep_types = list(set(m.get('dependency_type', 'none') for m in metadata))
    dep_colors = plt.cm.tab10(np.linspace(0, 1, len(dep_types)))
    dep_color_map = dict(zip(dep_types, dep_colors))
    colors = [dep_color_map[m.get('dependency_type', 'none')] for m in metadata]
    
    ax2.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
               c=colors, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax2.set_title('t-SNE - Colored by Dependency Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    # 3. t-SNE colored by position
    ax3 = fig.add_subplot(gs[0, 2])
    positions = [m.get('position', i) for i, m in enumerate(metadata)]
    scatter = ax3.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                         c=positions, cmap='viridis', alpha=0.7, s=50,
                         edgecolors='white', linewidth=0.5)
    ax3.set_title('t-SNE - Colored by Position', fontsize=12, fontweight='bold')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax3, label='Message Position')
    
    # 4. PCA with explained variance
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1],
               c=[role_colors.get(r, '#95a5a6') for r in roles],
               alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax4.set_title(f'PCA - Explained Variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    
    # 5. Density plot
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hexbin(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], 
               gridsize=20, cmap='YlOrRd', mincnt=1)
    ax5.set_title('Embedding Density (t-SNE)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('t-SNE 1')
    ax5.set_ylabel('t-SNE 2')
    
    # 6. Text length analysis
    ax6 = fig.add_subplot(gs[1, 2])
    text_lengths = [m.get('text_length', 0) for m in metadata]
    scatter = ax6.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                         c=text_lengths, cmap='plasma', alpha=0.7, s=50,
                         edgecolors='white', linewidth=0.5)
    ax6.set_title('t-SNE - Colored by Text Length', fontsize=12, fontweight='bold')
    ax6.set_xlabel('t-SNE 1')
    ax6.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax6, label='Text Length (tokens)')
    
    # 7. Dependency distance analysis
    ax7 = fig.add_subplot(gs[2, :])
    dep_distances = []
    for i, m in enumerate(metadata):
        if 'depends_on_indices' in m and m['depends_on_indices']:
            for dep_idx in m['depends_on_indices']:
                dep_distances.append(i - dep_idx)
    
    if dep_distances:
        ax7.hist(dep_distances, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax7.axvline(np.mean(dep_distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(dep_distances):.1f}')
        ax7.set_title('Dependency Distance Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Distance (number of messages)')
        ax7.set_ylabel('Count')
        ax7.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_attention_flow(attention_weights: torch.Tensor,
                       message_labels: List[str],
                       save_path: Optional[Path] = None,
                       title: str = "Attention Flow Analysis"):
    """
    Create sophisticated attention flow visualization
    
    Args:
        attention_weights: [num_queries, num_contexts] attention matrix
        message_labels: Labels for each message
        save_path: Path to save figure
        title: Plot title
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Standard heatmap with annotations
    ax1 = axes[0, 0]
    mask = np.triu(np.ones_like(attention_weights), k=1)
    sns.heatmap(attention_weights, 
                xticklabels=message_labels,
                yticklabels=message_labels,
                cmap='YlOrRd',
                mask=mask,
                annot=attention_weights < 0.1,
                fmt='.2f',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax1)
    ax1.set_title('Attention Weights Heatmap', fontweight='bold')
    ax1.set_xlabel('Context Messages')
    ax1.set_ylabel('Query Messages')
    
    # 2. Attention flow diagram
    ax2 = axes[0, 1]
    ax2.set_xlim(-0.5, len(message_labels) - 0.5)
    ax2.set_ylim(-0.5, len(message_labels) - 0.5)
    
    # Draw connections
    for i in range(len(message_labels)):
        for j in range(i):
            weight = attention_weights[i, j]
            if weight > 0.1:  # Only show significant connections
                ax2.plot([j, i], [j, i], 
                        alpha=min(weight * 2, 1.0),
                        linewidth=weight * 5,
                        color='#e74c3c')
    
    # Draw nodes
    for i, label in enumerate(message_labels):
        ax2.scatter(i, i, s=300, c='#3498db', edgecolors='black', linewidth=2, zorder=5)
        ax2.text(i, i, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_title('Attention Flow Network', fontweight='bold')
    ax2.set_xlabel('Message Index')
    ax2.set_ylabel('Message Index')
    ax2.invert_yaxis()
    
    # 3. Attention distribution per query
    ax3 = axes[1, 0]
    for i in range(1, len(message_labels)):
        weights = attention_weights[i, :i]
        if len(weights) > 0:
            ax3.plot(range(len(weights)), weights, 
                    marker='o', label=f'Query {i}', alpha=0.7)
    
    ax3.set_title('Attention Distribution per Query', fontweight='bold')
    ax3.set_xlabel('Context Position')
    ax3.set_ylabel('Attention Weight')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Attention statistics
    ax4 = axes[1, 1]
    
    # Calculate statistics
    avg_attention_per_position = []
    for j in range(len(message_labels)):
        weights = [attention_weights[i, j] for i in range(j+1, len(message_labels))]
        if weights:
            avg_attention_per_position.append(np.mean(weights))
        else:
            avg_attention_per_position.append(0)
    
    x = range(len(avg_attention_per_position))
    ax4.bar(x, avg_attention_per_position, color='#3498db', alpha=0.7)
    ax4.set_title('Average Attention Received by Position', fontweight='bold')
    ax4.set_xlabel('Message Position')
    ax4.set_ylabel('Average Attention')
    ax4.set_xticks(x)
    ax4.set_xticklabels(message_labels, rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_error_analysis_detailed(error_cases: List[Dict],
                               save_path: Optional[Path] = None,
                               title: str = "Detailed Error Analysis"):
    """
    Create detailed error analysis visualizations
    
    Args:
        error_cases: List of error case dictionaries
        save_path: Path to save figure
        title: Plot title
    """
    if not error_cases:
        print("No error cases to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Error types distribution
    ax1 = axes[0, 0]
    dep_types = [case.get('dependency_type', 'unknown') for case in error_cases]
    dep_type_counts = Counter(dep_types)
    
    types, counts = zip(*dep_type_counts.most_common())
    y_pos = np.arange(len(types))
    ax1.barh(y_pos, counts, color='#e74c3c', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(types)
    ax1.set_xlabel('Error Count')
    ax1.set_title('Errors by Dependency Type', fontweight='bold')
    
    # 2. False positive/negative analysis
    ax2 = axes[0, 1]
    fp_counts = [len(case.get('false_positives', [])) for case in error_cases]
    fn_counts = [len(case.get('false_negatives', [])) for case in error_cases]
    
    x = np.arange(len(error_cases))
    width = 0.35
    ax2.bar(x - width/2, fp_counts, width, label='False Positives', color='#e74c3c', alpha=0.7)
    ax2.bar(x + width/2, fn_counts, width, label='False Negatives', color='#3498db', alpha=0.7)
    ax2.set_xlabel('Error Case')
    ax2.set_ylabel('Count')
    ax2.set_title('False Positives vs False Negatives', fontweight='bold')
    ax2.legend()
    
    # 3. Score distribution analysis
    ax3 = axes[1, 0]
    all_scores = []
    score_labels = []
    
    for case in error_cases[:10]:  # Limit to first 10 cases
        scores = case.get('scores', {})
        if scores:
            case_scores = list(scores.values())
            all_scores.extend(case_scores)
            score_labels.extend([f"Case {error_cases.index(case)}" for _ in case_scores])
    
    if all_scores:
        df = pd.DataFrame({'scores': all_scores, 'case': score_labels})
        df.boxplot(column='scores', by='case', ax=ax3)
        ax3.set_title('Score Distribution by Error Case', fontweight='bold')
        ax3.set_xlabel('Error Case')
        ax3.set_ylabel('Relevance Score')
    
    # 4. Missed reasons analysis
    ax4 = axes[1, 1]
    missed_reasons = []
    for case in error_cases:
        reasons = case.get('missed_reasons', [])
        for reason in reasons:
            if 'Low score' in reason:
                missed_reasons.append('Low Score')
            elif 'Long range' in reason:
                missed_reasons.append('Long Range')
            else:
                missed_reasons.append('Other')
    
    reason_counts = Counter(missed_reasons)
    if reason_counts:
        labels, sizes = zip(*reason_counts.items())
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(labels)]
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Reasons for Missing Dependencies', fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_performance_by_conversation_length(results_by_length: Dict[int, Dict],
                                          save_path: Optional[Path] = None,
                                          title: str = "Performance vs Conversation Length"):
    """
    Plot how model performance varies with conversation length
    
    Args:
        results_by_length: Dict mapping conversation length to metrics
        save_path: Path to save figure
        title: Plot title
    """
    if not results_by_length:
        print("No results to visualize")
        return
    
    # Prepare data
    lengths = sorted(results_by_length.keys())
    metrics_to_plot = ['precision@1', 'recall@5', 'mrr', 'map']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for metric in metrics_to_plot:
        values = [results_by_length[l].get(metric, 0) for l in lengths]
        ax.plot(lengths, values, marker='o', label=metric.upper(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Conversation Length (messages)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add trend lines
    from scipy import stats
    for i, metric in enumerate(metrics_to_plot):
        values = [results_by_length[l].get(metric, 0) for l in lengths]
        slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, values)
        line = slope * np.array(lengths) + intercept
        ax.plot(lengths, line, '--', alpha=0.5, 
               label=f'{metric} trend (r²={r_value**2:.3f})')
    
    ax.legend(loc='best', frameon=True, fancybox=True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def create_comprehensive_report(all_results: Dict[str, Any],
                              output_dir: Path):
    """
    Create a comprehensive visual report combining all analyses
    
    Args:
        all_results: Dictionary containing all evaluation results
        output_dir: Directory to save the report
    """
    # Create a multi-page report
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = output_dir / 'comprehensive_evaluation_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'Hierarchical Conversation GNN V2', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, 'Comprehensive Evaluation Report', 
                ha='center', va='center', fontsize=18)
        fig.text(0.5, 0.5, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 
                ha='center', va='center', fontsize=12)
        
        # Add summary metrics
        overall_metrics = all_results.get('overall', {})
        summary_text = f"""
Key Metrics:
• MRR: {overall_metrics.get('mrr', 0):.3f}
• MAP: {overall_metrics.get('map', 0):.3f}
• Precision@1: {overall_metrics.get('precision@1', 0):.3f}
• Recall@5: {overall_metrics.get('recall@5', 0):.3f}
        """
        fig.text(0.5, 0.3, summary_text, ha='center', va='center', fontsize=14)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Performance by dependency type
        if 'by_dependency_type' in all_results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            dep_data = all_results['by_dependency_type']
            # Create mapping of original keys to display labels
            dep_type_mapping = [(dt, str(dt) if dt is not None else 'unknown') 
                               for dt in dep_data.keys()]
            original_types = [dt[0] for dt in dep_type_mapping]
            display_types = [dt[1] for dt in dep_type_mapping]
            
            # Precision@1
            ax = axes[0, 0]
            values = [dep_data[dt].get('precision@1', 0) for dt in original_types]
            ax.bar(display_types, values, color='#3498db', alpha=0.7)
            ax.set_title('Precision@1 by Dependency Type', fontweight='bold')
            ax.set_ylabel('Precision@1')
            ax.tick_params(axis='x', rotation=45)
            
            # Recall@5
            ax = axes[0, 1]
            values = [dep_data[dt].get('recall@5', 0) for dt in original_types]
            ax.bar(display_types, values, color='#2ecc71', alpha=0.7)
            ax.set_title('Recall@5 by Dependency Type', fontweight='bold')
            ax.set_ylabel('Recall@5')
            ax.tick_params(axis='x', rotation=45)
            
            # MRR
            ax = axes[1, 0]
            values = [dep_data[dt].get('mrr', 0) for dt in original_types]
            ax.bar(display_types, values, color='#e74c3c', alpha=0.7)
            ax.set_title('MRR by Dependency Type', fontweight='bold')
            ax.set_ylabel('MRR')
            ax.tick_params(axis='x', rotation=45)
            
            # Combined metrics
            ax = axes[1, 1]
            metrics = ['precision@1', 'recall@5', 'mrr']
            x = np.arange(len(display_types))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [dep_data[dt].get(metric, 0) for dt in original_types]
                ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.7)
            
            ax.set_xlabel('Dependency Type')
            ax.set_ylabel('Score')
            ax.set_title('All Metrics by Dependency Type', fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(display_types, rotation=45, ha='right')
            ax.legend()
            
            plt.suptitle('Performance Analysis by Dependency Type', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Comprehensive report saved to: {pdf_path}")