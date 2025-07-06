#!/usr/bin/env python3
"""
Evaluation script for Hierarchical Conversation GNN V2
"""

import argparse
import json
import logging
import torch
from pathlib import Path
from datetime import datetime

from models.hierarchical_gnn import HierarchicalConversationGNN
from utils.evaluation import ConversationEvaluator
from utils.visualization import (
    plot_retrieval_metrics, 
    plot_attention_heatmap,
    plot_token_importance
)
from utils.advanced_visualization import (
    plot_embedding_analysis,
    plot_attention_flow,
    plot_error_analysis_detailed,
    plot_performance_by_conversation_length,
    create_comprehensive_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = 'cuda') -> HierarchicalConversationGNN:
    """Load model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Try to infer from state dict
        state_dict = checkpoint['model_state_dict']
        config = {
            'vocab_size': state_dict['token_embeddings.weight'].shape[0],
            'token_embedding_dim': state_dict['token_embeddings.weight'].shape[1],
            'hidden_dim': 256,  # Default
            'message_dim': 128,  # Default
            'num_heads': 4,
            'max_seq_length': state_dict['position_embeddings.weight'].shape[0]
        }
    
    # Initialize model
    model = HierarchicalConversationGNN(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def evaluate_model(model, test_conversations, output_dir, device='cuda'):
    """Run comprehensive evaluation"""
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(model, device)
    
    # 1. Retrieval quality evaluation
    logger.info("Evaluating retrieval quality...")
    retrieval_results = evaluator.evaluate_retrieval(
        test_conversations,
        k_values=[1, 3, 5, 10],
        by_dependency_type=True
    )
    
    # Log results
    logger.info("\nRetrieval Results:")
    logger.info(f"  Evaluated on {retrieval_results['num_evaluated']} queries")
    for metric, value in retrieval_results['overall'].items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # Save detailed results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(retrieval_results, f, indent=2)
    
    # Plot retrieval metrics
    plot_retrieval_metrics(
        retrieval_results['overall'],
        retrieval_results.get('by_dependency_type'),
        save_path=output_dir / 'retrieval_metrics.png'
    )
    
    # 2. Efficiency evaluation
    logger.info("\nEvaluating efficiency...")
    efficiency_results = evaluator.evaluate_efficiency(
        test_conversations[:100],  # Sample
        batch_sizes=[1, 4, 8, 16]
    )
    
    logger.info("Efficiency Results:")
    for batch_info, metrics in efficiency_results.items():
        logger.info(f"  {batch_info}:")
        for metric, value in metrics.items():
            logger.info(f"    {metric}: {value:.3f}")
    
    # 3. Error analysis
    logger.info("\nAnalyzing errors...")
    error_cases = evaluator.analyze_errors(test_conversations, top_k=10)
    
    # Save error analysis
    error_path = output_dir / 'error_analysis.json'
    with open(error_path, 'w') as f:
        json.dump(error_cases, f, indent=2)
    
    logger.info(f"Found {len(error_cases)} error cases")
    if error_cases:
        logger.info("Top error case:")
        case = error_cases[0]
        logger.info(f"  Query: {case['query_text']}")
        logger.info(f"  True deps: {case['true_dependencies']}")
        logger.info(f"  Predicted: {case['predicted']}")
        logger.info(f"  False negatives: {case['false_negatives']}")
        
        # Create detailed error visualization
        plot_error_analysis_detailed(
            error_cases,
            save_path=output_dir / 'error_analysis_detailed.png',
            title='Detailed Error Analysis'
        )
    
    # 4. Performance by conversation length
    logger.info("\nAnalyzing performance by conversation length...")
    length_results = {}
    length_bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100)]
    
    for min_len, max_len in length_bins:
        conversations_in_bin = [
            conv for conv in test_conversations
            if min_len <= len(conv.get('messages', conv)) < max_len
        ]
        
        if conversations_in_bin:
            bin_results = evaluator.evaluate_retrieval(
                conversations_in_bin,
                k_values=[1, 5],
                by_dependency_type=False
            )
            length_results[f"{min_len}-{max_len}"] = bin_results['overall']
    
    # Create comprehensive report
    logger.info("\nGenerating comprehensive report...")
    create_comprehensive_report(retrieval_results, output_dir)
    
    # 5. Visualize sample conversations
    logger.info("\nCreating visualizations...")
    visualize_samples(model, test_conversations[:5], output_dir, device)
    
    return retrieval_results


def visualize_samples(model, conversations, output_dir, device='cuda'):
    """Create visualizations for sample conversations"""
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Collect embeddings and metadata for advanced visualization
    all_embeddings = []
    all_metadata = []
    
    for i, conv in enumerate(conversations):
        messages = conv.get('messages', conv)
        if len(messages) < 5:
            continue
        
        # Get embeddings and attention
        from utils.evaluation import ConversationEvaluator
        evaluator = ConversationEvaluator(model, device)
        embeddings = evaluator._get_message_embeddings(messages)
        
        # Collect for global analysis
        for j, (emb, msg) in enumerate(zip(embeddings, messages)):
            all_embeddings.append(emb)
            all_metadata.append({
                'role': msg['role'],
                'dependency_type': msg.get('dependency_type', 'none'),
                'position': j,
                'text_length': len(msg['text'].split()),
                'depends_on_indices': msg.get('depends_on_indices', []),
                'conversation_id': i
            })
        
        # Create attention matrix
        num_messages = len(messages)
        attention_matrix = torch.zeros(num_messages, num_messages)
        
        with torch.no_grad():
            for query_idx in range(1, num_messages):
                scores = evaluator._score_all_pairs(embeddings, query_idx)
                if len(scores) > 0:
                    weights = torch.softmax(scores, dim=0)
                    attention_matrix[query_idx, :query_idx] = weights
        
        # Create labels
        labels = [f"{j}:{m['role'][:3]}" for j, m in enumerate(messages)]
        
        # Plot standard attention heatmap
        plot_attention_heatmap(
            attention_matrix.cpu(),
            labels,
            labels,
            save_path=vis_dir / f'attention_conv_{i}.png',
            title=f'Conversation {i} Attention Pattern'
        )
        
        # Plot advanced attention flow
        plot_attention_flow(
            attention_matrix,
            labels,
            save_path=vis_dir / f'attention_flow_conv_{i}.png',
            title=f'Conversation {i} Attention Flow Analysis'
        )
        
        # Get token importance if available
        if hasattr(model, 'last_token_attention_weights'):
            token_scores = {}
            token_texts = {}
            
            for msg_idx in range(min(3, num_messages)):
                importance = model.get_token_importance(msg_idx)
                if importance is not None:
                    token_scores[msg_idx] = importance
                    # Get tokens (simplified)
                    tokens = messages[msg_idx]['text'].split()[:20]
                    token_texts[msg_idx] = tokens
            
            if token_scores:
                plot_token_importance(
                    token_scores,
                    token_texts,
                    save_path=vis_dir / f'token_importance_conv_{i}.png'
                )
    
    # Create global embedding analysis
    if all_embeddings:
        all_embeddings_tensor = torch.stack(all_embeddings)
        plot_embedding_analysis(
            all_embeddings_tensor,
            all_metadata,
            save_path=vis_dir / 'global_embedding_analysis.png',
            title='Global Message Embedding Analysis'
        )


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hierarchical Conversation GNN V2')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test conversations JSON')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--max-conversations', type=int, default=None,
                       help='Maximum conversations to evaluate')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_conversations = json.load(f)
    
    if args.max_conversations:
        test_conversations = test_conversations[:args.max_conversations]
    
    logger.info(f"Loaded {len(test_conversations)} conversations")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Run evaluation
    results = evaluate_model(model, test_conversations, output_dir, args.device)
    
    # Create summary report
    summary = {
        'checkpoint': args.checkpoint,
        'test_data': args.test_data,
        'num_conversations': len(test_conversations),
        'timestamp': datetime.now().isoformat(),
        'overall_metrics': results['overall'],
        'device': args.device
    }
    
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nEvaluation complete! Results saved to {output_dir}")
    logger.info("\nKey metrics:")
    logger.info(f"  MRR: {results['overall']['mrr']:.3f}")
    logger.info(f"  MAP: {results['overall']['map']:.3f}")
    logger.info(f"  Recall@5: {results['overall']['recall@5']:.3f}")
    logger.info(f"  F1@5: {results['overall']['f1@5']:.3f}")


if __name__ == "__main__":
    main()