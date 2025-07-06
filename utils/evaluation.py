#!/usr/bin/env python3
"""
Evaluation utilities for Hierarchical Conversation GNN V2
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ConversationEvaluator:
    """
    Comprehensive evaluation for conversation context retrieval
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate_retrieval(self, 
                          conversations: List[Dict],
                          k_values: List[int] = [1, 3, 5, 10],
                          by_dependency_type: bool = True) -> Dict:
        """
        Evaluate retrieval quality with multiple metrics
        
        Args:
            conversations: List of conversations to evaluate
            k_values: K values for precision/recall@k
            by_dependency_type: Whether to break down metrics by dependency type
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Initialize metrics
        metrics = {
            f'precision@{k}': [] for k in k_values
        }
        metrics.update({
            f'recall@{k}': [] for k in k_values
        })
        metrics['mrr'] = []  # Mean Reciprocal Rank
        metrics['map'] = []  # Mean Average Precision
        metrics['ndcg'] = []  # Normalized Discounted Cumulative Gain
        
        # Metrics by dependency type
        if by_dependency_type:
            metrics_by_type = defaultdict(lambda: defaultdict(list))
        
        # Process each conversation
        for conv in conversations:
            messages = conv.get('messages', conv)
            
            if len(messages) < 2:
                continue
            
            # Build conversation in model
            embeddings = self._get_message_embeddings(messages)
            
            # Evaluate each message with dependencies
            for i in range(1, len(messages)):
                msg = messages[i]
                if not msg.get('depends_on_indices'):
                    continue
                
                true_deps = set(msg['depends_on_indices'])
                dep_type = msg.get('dependency_type', 'unknown')
                
                # Get model predictions
                scores = self._score_all_pairs(embeddings, i)
                ranked_indices = torch.argsort(scores, descending=True).tolist()
                
                # Compute metrics at each k
                for k in k_values:
                    if k > len(ranked_indices):
                        continue
                    
                    pred_at_k = set(ranked_indices[:k])
                    
                    precision = len(pred_at_k & true_deps) / k if k > 0 else 0
                    recall = len(pred_at_k & true_deps) / len(true_deps) if true_deps else 0
                    
                    metrics[f'precision@{k}'].append(precision)
                    metrics[f'recall@{k}'].append(recall)
                    
                    if by_dependency_type:
                        metrics_by_type[dep_type][f'precision@{k}'].append(precision)
                        metrics_by_type[dep_type][f'recall@{k}'].append(recall)
                
                # MRR
                for rank, idx in enumerate(ranked_indices, 1):
                    if idx in true_deps:
                        metrics['mrr'].append(1.0 / rank)
                        if by_dependency_type:
                            metrics_by_type[dep_type]['mrr'].append(1.0 / rank)
                        break
                else:
                    metrics['mrr'].append(0.0)
                    if by_dependency_type:
                        metrics_by_type[dep_type]['mrr'].append(0.0)
                
                # MAP
                precisions = []
                num_relevant = 0
                for rank, idx in enumerate(ranked_indices, 1):
                    if idx in true_deps:
                        num_relevant += 1
                        precisions.append(num_relevant / rank)
                
                ap = np.mean(precisions) if precisions else 0.0
                metrics['map'].append(ap)
                if by_dependency_type:
                    metrics_by_type[dep_type]['map'].append(ap)
                
                # NDCG
                relevance_scores = [1.0 if idx in true_deps else 0.0 for idx in ranked_indices]
                ndcg = self._compute_ndcg(relevance_scores, len(true_deps))
                metrics['ndcg'].append(ndcg)
                if by_dependency_type:
                    metrics_by_type[dep_type]['ndcg'].append(ndcg)
        
        # Aggregate metrics
        results = {
            'overall': {k: np.mean(v) if v else 0.0 for k, v in metrics.items()},
            'num_evaluated': len(metrics['mrr'])
        }
        
        if by_dependency_type:
            results['by_dependency_type'] = {
                dep_type: {k: np.mean(v) if v else 0.0 for k, v in type_metrics.items()}
                for dep_type, type_metrics in metrics_by_type.items()
            }
        
        # Add F1 scores
        for k in k_values:
            p = results['overall'][f'precision@{k}']
            r = results['overall'][f'recall@{k}']
            results['overall'][f'f1@{k}'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        return results
    
    def evaluate_efficiency(self, conversations: List[Dict], 
                          batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict:
        """
        Evaluate model efficiency metrics
        
        Returns:
            Dictionary with timing and memory statistics
        """
        import time
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(conversations):
                continue
            
            # Sample conversations
            sample = conversations[:batch_size]
            
            # Warmup
            for _ in range(3):
                _ = self._get_message_embeddings(sample[0]['messages'])
            
            # Time embedding generation
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for conv in sample:
                _ = self._get_message_embeddings(conv['messages'])
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            
            results[f'batch_{batch_size}'] = {
                'total_time': elapsed,
                'per_conversation': elapsed / batch_size,
                'conversations_per_second': batch_size / elapsed
            }
            
            # Memory usage
            if torch.cuda.is_available():
                results[f'batch_{batch_size}']['peak_memory_mb'] = \
                    torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return results
    
    def analyze_errors(self, conversations: List[Dict], 
                      top_k: int = 5) -> List[Dict]:
        """
        Analyze common error patterns
        
        Returns:
            List of error cases with analysis
        """
        error_cases = []
        
        for conv in conversations:
            messages = conv.get('messages', conv)
            if len(messages) < 2:
                continue
            
            embeddings = self._get_message_embeddings(messages)
            
            for i in range(1, len(messages)):
                msg = messages[i]
                if not msg.get('depends_on_indices'):
                    continue
                
                true_deps = set(msg['depends_on_indices'])
                scores = self._score_all_pairs(embeddings, i)
                ranked_indices = torch.argsort(scores, descending=True).tolist()
                
                predicted = set(ranked_indices[:len(true_deps)])
                
                false_positives = predicted - true_deps
                false_negatives = true_deps - predicted
                
                if false_positives or false_negatives:
                    error_case = {
                        'query_text': msg['text'][:100] + '...',
                        'query_idx': i,
                        'true_dependencies': list(true_deps),
                        'predicted': list(predicted),
                        'false_positives': list(false_positives),
                        'false_negatives': list(false_negatives),
                        'dependency_type': msg.get('dependency_type', 'unknown'),
                        'scores': {idx: scores[idx].item() for idx in range(i)}
                    }
                    
                    # Analyze why errors occurred
                    if false_negatives:
                        error_case['missed_reasons'] = []
                        for fn_idx in false_negatives:
                            distance = i - fn_idx
                            if distance > 10:
                                error_case['missed_reasons'].append(f"Long range: {distance}")
                            
                            # Check if low score
                            if fn_idx < len(scores):
                                score = scores[fn_idx].item()
                                rank = ranked_indices.index(fn_idx) + 1
                                error_case['missed_reasons'].append(
                                    f"Low score: {score:.3f} (rank {rank})"
                                )
                    
                    error_cases.append(error_case)
        
        # Sort by error severity
        error_cases.sort(key=lambda x: len(x['false_negatives']), reverse=True)
        
        return error_cases[:top_k]
    
    def _get_message_embeddings(self, messages: List[Dict]) -> torch.Tensor:
        """Get message embeddings from model"""
        from transformers import AutoTokenizer
        
        # Use default tokenizer if not provided
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Prepare inputs
        token_ids_list = []
        lengths = []
        
        for msg in messages:
            tokens = self.tokenizer(
                msg['text'],
                max_length=128,
                truncation=True,
                return_tensors='pt'
            )
            token_ids = tokens['input_ids'].squeeze().to(self.device)
            token_ids_list.append(token_ids)
            lengths.append(len(token_ids))
        
        # Create message graph
        num_messages = len(messages)
        edge_list = []
        for i in range(num_messages - 1):
            edge_list.extend([[i, i+1], [i+1, i]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list).T.to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        
        # Node attributes
        role_map = {'user': 0, 'assistant': 1, 'system': 2, 'tool': 3}
        roles = torch.tensor([role_map.get(m['role'], 0) for m in messages], 
                           dtype=torch.float).unsqueeze(1).to(self.device)
        positions = torch.arange(num_messages, dtype=torch.float).unsqueeze(1).to(self.device) / max(num_messages - 1, 1)
        node_attr = torch.cat([roles, positions], dim=1)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(
                token_ids_list,
                lengths,
                edge_index,
                node_attr
            )
        
        return embeddings
    
    def _score_all_pairs(self, embeddings: torch.Tensor, query_idx: int) -> torch.Tensor:
        """Score all context messages for a query"""
        scores = []
        query_emb = embeddings[query_idx:query_idx+1]
        
        with torch.no_grad():
            for ctx_idx in range(query_idx):
                score = self.model.score_relevance(
                    query_emb,
                    embeddings[ctx_idx:ctx_idx+1],
                    query_idx=query_idx,
                    context_idx=ctx_idx
                )
                scores.append(score.squeeze())
        
        return torch.stack(scores) if scores else torch.tensor([])
    
    def _compute_ndcg(self, relevance_scores: List[float], num_relevant: int) -> float:
        """Compute Normalized Discounted Cumulative Gain"""
        if not relevance_scores or num_relevant == 0:
            return 0.0
        
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # Ideal DCG
        ideal_relevance = [1.0] * num_relevant + [0.0] * (len(relevance_scores) - num_relevant)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0