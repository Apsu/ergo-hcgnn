#!/usr/bin/env python3
"""
Advanced trainer for Hierarchical Conversation GNN V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from models.hierarchical_gnn import HierarchicalConversationGNN
from data.dataset import ConversationDataset, collate_conversations
from training.losses import DependencyAwareLoss, AdaptiveLossWeighting, ContrastiveLoss, ListNetRankingLoss, MarginRankingLoss
from training.curriculum import CurriculumScheduler, HardExampleMiner, CurriculumConfig

logger = logging.getLogger(__name__)


class HierarchicalGNNTrainer:
    """
    Complete training pipeline for Hierarchical Conversation GNN
    """
    
    def __init__(self,
                 model: HierarchicalConversationGNN,
                 train_dataset: ConversationDataset,
                 val_dataset: ConversationDataset,
                 output_dir: str = 'checkpoints',
                 learning_rate: float = 1e-4,
                 batch_size: int = 8,
                 accumulation_steps: int = 1,
                 num_epochs: int = 30,
                 device: str = 'cuda',
                 num_workers: int = 4,
                 use_curriculum: bool = True,
                 use_adaptive_loss: bool = True,
                 checkpoint_every: int = 5,
                 patience: int = 5,
                 **kwargs):
        """
        Initialize trainer with model and datasets
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.checkpoint_every = checkpoint_every
        self.patience = patience
        
        # Loss weights
        self.relevance_weight = kwargs.get('relevance_weight', 1.0)
        self.contrastive_weight = kwargs.get('contrastive_weight', 0.1)
        self.ranking_weight = kwargs.get('ranking_weight', 0.3)
        self.margin_weight = kwargs.get('margin_weight', 0.1)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=kwargs.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        # Initialize separate loss functions
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07)
        self.ranking_loss_fn = ListNetRankingLoss(temperature=1.0)
        self.margin_loss_fn = MarginRankingLoss(margin=0.5)
        
        # Adaptive loss weighting
        self.use_adaptive_loss = use_adaptive_loss
        if use_adaptive_loss:
            self.adaptive_loss_weighter = AdaptiveLossWeighting(num_losses=4).to(device)
            self.optimizer.add_param_group({
                'params': self.adaptive_loss_weighter.parameters(),
                'lr': learning_rate
            })
        
        # Curriculum learning
        self.use_curriculum = use_curriculum
        if use_curriculum:
            curriculum_config = CurriculumConfig(**kwargs.get('curriculum_config', {}))
            self.curriculum_scheduler = CurriculumScheduler(curriculum_config, num_epochs)
            self.hard_example_miner = HardExampleMiner()
        else:
            self.curriculum_scheduler = None
            self.hard_example_miner = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_conversations,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_conversations,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        self.current_epoch = 0
        
        # Log configuration
        self._log_configuration()
    
    def _log_configuration(self):
        """Log training configuration"""
        config = {
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'accumulation_steps': self.accumulation_steps,
            'effective_batch_size': self.batch_size * self.accumulation_steps,
            'num_epochs': self.num_epochs,
            'device': str(self.device),
            'use_curriculum': self.use_curriculum,
            'use_adaptive_loss': self.use_adaptive_loss,
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Training Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    def compute_relevance_loss(self, attention_weights: torch.Tensor, 
                              target_weights: torch.Tensor,
                              dependency_types: Optional[List[str]] = None) -> torch.Tensor:
        """Compute KL divergence loss between predicted and target attention distributions"""
        # KL divergence loss
        log_attention = torch.log(attention_weights + 1e-10)
        kl_loss = F.kl_div(log_attention, target_weights, reduction='none').sum(dim=-1)
        
        # Apply dependency-specific weights if provided
        if dependency_types is not None:
            dep_weights = {
                'continuation': 1.2,
                'pronoun_reference': 1.5,
                'clarification': 1.3,
                'topic_reference': 1.0,
                'agreement': 1.1,
                'follow_up': 1.1,
                'example_request': 1.0,
                'none': 0.8
            }
            weights = torch.tensor(
                [dep_weights.get(dt, 1.0) for dt in dependency_types],
                device=attention_weights.device
            )
            kl_loss = kl_loss * weights
        
        return kl_loss.mean()
    
    def compute_batch_loss(self, batch: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Compute all losses for a batch
        
        Returns:
            losses: Dict of loss tensors
            metrics: Dict of scalar metrics for logging
        """
        # Forward pass through model
        message_embeddings = self.model(
            batch['token_ids_list'],
            batch['lengths'],
            batch['message_edge_index'],
            batch['message_node_attr']
        )
        
        # Initialize losses and metrics
        all_losses = defaultdict(list)
        metrics = defaultdict(float)
        
        # Collect data for batch-level losses
        all_embeddings = []
        all_dependency_labels = []
        all_relevance_scores = []
        all_relevance_labels = []
        
        # Process each conversation for query-specific losses
        for conv_idx, (targets, num_messages) in enumerate(
            zip(batch['targets'], batch['num_messages_list'])
        ):
            # Get embeddings for this conversation
            mask = batch['message_batch'] == conv_idx
            conv_embeddings = message_embeddings[mask]
            
            # Collect embeddings for contrastive loss
            all_embeddings.append(conv_embeddings)
            
            # Create dependency labels for this conversation (simplified)
            conv_dep_labels = torch.zeros(num_messages, device=self.device, dtype=torch.long)
            
            if 'relevance_queries' not in targets:
                all_dependency_labels.append(conv_dep_labels)
                continue
            
            # Process each query in the conversation
            for query_info in targets['relevance_queries']:
                query_idx = query_info['query_idx']
                context_indices = query_info['context_indices']
                relevant_indices = query_info.get('relevant_indices', [])
                dependency_type = query_info.get('dependency_type', 'none')
                
                if query_idx >= num_messages or not context_indices:
                    continue
                
                # Update dependency label for query message
                dep_type_map = {
                    'continuation': 0,
                    'pronoun_reference': 1,
                    'agreement': 2,
                    'topic_reference': 3,
                    'clarification': 4,
                    'follow_up': 5,
                    'example_request': 6,
                    'none': 7
                }
                conv_dep_labels[query_idx] = dep_type_map.get(dependency_type, 7)
                
                # Score all context messages
                query_emb = conv_embeddings[query_idx:query_idx+1]
                scores = []
                
                for ctx_idx in context_indices:
                    score = self.model.score_relevance(
                        query_emb,
                        conv_embeddings[ctx_idx:ctx_idx+1],
                        query_idx=query_idx,
                        context_idx=ctx_idx
                    )
                    scores.append(score)
                
                if not scores:
                    continue
                
                scores = torch.cat(scores)
                
                # Compute attention weights
                attention_weights = F.softmax(scores.squeeze(-1), dim=0)
                
                # Create target distribution
                if relevant_indices:
                    target_weights = torch.zeros(len(context_indices), device=self.device)
                    for rel_idx in relevant_indices:
                        if rel_idx in context_indices:
                            pos = context_indices.index(rel_idx)
                            target_weights[pos] = 1.0
                    
                    # Normalize with label smoothing
                    smoothing = 0.1
                    if target_weights.sum() > 0:
                        target_weights = (1 - smoothing) * target_weights / target_weights.sum() + \
                                       smoothing / len(target_weights)
                    else:
                        target_weights = torch.ones_like(target_weights) / len(target_weights)
                else:
                    # Use recency bias for non-dependent messages
                    distances = torch.arange(len(scores), 0, -1, device=self.device, dtype=torch.float)
                    target_weights = torch.exp(-0.1 * distances)
                    target_weights = target_weights / target_weights.sum()
                
                # 1. Relevance loss (KL divergence)
                relevance_loss = self.compute_relevance_loss(
                    attention_weights.unsqueeze(0),
                    target_weights.unsqueeze(0),
                    [dependency_type]
                )
                all_losses['relevance'].append(relevance_loss)
                
                # Collect scores and labels for ranking losses
                relevance_labels = torch.zeros(len(context_indices), device=self.device)
                for rel_idx in relevant_indices:
                    if rel_idx in context_indices:
                        pos = context_indices.index(rel_idx)
                        relevance_labels[pos] = 1.0
                
                all_relevance_scores.append(scores.squeeze(-1))
                all_relevance_labels.append(relevance_labels)
                
                # Track metrics
                metrics['num_queries'] += 1
                
                # Compute accuracy metrics
                predicted_idx = attention_weights.argmax().item()
                if predicted_idx in [context_indices.index(ri) for ri in relevant_indices if ri in context_indices]:
                    metrics['correct_top1'] += 1
                
                # Top-k accuracy
                k = min(5, len(attention_weights))
                top_k_indices = torch.topk(attention_weights, k).indices.tolist()
                relevant_positions = [context_indices.index(ri) for ri in relevant_indices if ri in context_indices]
                if any(idx in relevant_positions for idx in top_k_indices):
                    metrics['correct_top5'] += 1
            
            all_dependency_labels.append(conv_dep_labels)
        
        # 2. Contrastive loss (batch-level)
        if self.contrastive_weight > 0 and all_embeddings:
            # Concatenate all embeddings
            batch_embeddings = torch.cat(all_embeddings, dim=0)
            batch_dep_labels = torch.cat(all_dependency_labels, dim=0)
            
            # Only compute if we have multiple dependency types
            if len(torch.unique(batch_dep_labels)) > 1:
                contrastive_loss = self.contrastive_loss_fn(batch_embeddings, batch_dep_labels)
                all_losses['contrastive'] = [contrastive_loss]
        
        # 3. Ranking losses (batch-level)
        if all_relevance_scores:
            # Process each query's scores separately since they may have different lengths
            ranking_losses = []
            margin_losses = []
            
            for scores, labels in zip(all_relevance_scores, all_relevance_labels):
                if len(scores) > 0:
                    # Add batch dimension
                    scores_batch = scores.unsqueeze(0)
                    labels_batch = labels.unsqueeze(0)
                    
                    if self.ranking_weight > 0:
                        ranking_losses.append(self.ranking_loss_fn(scores_batch, labels_batch))
                    
                    if self.margin_weight > 0:
                        margin_losses.append(self.margin_loss_fn(scores_batch, labels_batch))
            
            if ranking_losses:
                all_losses['ranking'] = ranking_losses
            if margin_losses:
                all_losses['margin'] = margin_losses
        
        # Aggregate losses
        final_losses = {}
        for loss_name, loss_list in all_losses.items():
            if loss_list:
                stacked = torch.stack(loss_list)
                if torch.isnan(stacked).any():
                    print(f"WARNING: NaN in {loss_name} loss")
                    final_losses[loss_name] = torch.tensor(0.0, device=self.device)
                else:
                    final_losses[loss_name] = stacked.mean()
            else:
                final_losses[loss_name] = torch.tensor(0.0, device=self.device)
        
        # Apply loss weights
        weighted_losses = {
            'relevance': final_losses.get('relevance', torch.tensor(0.0, device=self.device)) * self.relevance_weight,
            'contrastive': final_losses.get('contrastive', torch.tensor(0.0, device=self.device)) * self.contrastive_weight,
            'ranking': final_losses.get('ranking', torch.tensor(0.0, device=self.device)) * self.ranking_weight,
            'margin': final_losses.get('margin', torch.tensor(0.0, device=self.device)) * self.margin_weight
        }
        
        # Apply adaptive loss weighting if enabled
        if self.use_adaptive_loss:
            loss_values = [weighted_losses[name] for name in ['relevance', 'contrastive', 'ranking', 'margin']]
            final_losses['total'] = self.adaptive_loss_weighter(loss_values)
        else:
            final_losses['total'] = sum(weighted_losses.values())
        
        # Add weighted losses to final losses for logging
        final_losses.update(weighted_losses)
        
        # Compute accuracy metrics
        if metrics['num_queries'] > 0:
            metrics['accuracy_top1'] = metrics['correct_top1'] / metrics['num_queries']
            metrics['accuracy_top5'] = metrics['correct_top5'] / metrics['num_queries']
        else:
            metrics['accuracy_top1'] = 0.0
            metrics['accuracy_top5'] = 0.0
        
        return final_losses, metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # For curriculum learning
        if self.curriculum_scheduler:
            limits = self.curriculum_scheduler.get_current_limits()
            logger.info(f"Curriculum limits: {limits}")
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        # Training loop
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Compute losses
            losses, metrics = self.compute_batch_loss(batch)
            
            # Scale loss for accumulation
            scaled_loss = losses['total'] / self.accumulation_steps
            scaled_loss.backward()
            
            # Track losses and metrics
            for name, value in losses.items():
                epoch_losses[name] += value.item()
            for name, value in metrics.items():
                epoch_metrics[name] += value
            
            num_batches += 1
            
            # Update hard example miner
            if self.hard_example_miner and 'conversation_indices' in batch:
                conv_losses = [losses['total'].item()] * len(batch['conversation_indices'])
                self.hard_example_miner.update(batch['conversation_indices'], conv_losses)
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'acc@1': metrics.get('accuracy_top1', 0),
                'acc@5': metrics.get('accuracy_top5', 0)
            })
        
        # Final gradient step if needed
        if (batch_idx + 1) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Average losses and metrics
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = self._batch_to_device(batch)
                
                losses, metrics = self.compute_batch_loss(batch)
                
                for name, value in losses.items():
                    epoch_losses[name] += value.item()
                for name, value in metrics.items():
                    epoch_metrics[name] += value
                
                num_batches += 1
        
        # Average losses and metrics
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def train(self) -> Dict[str, List[float]]:
        """Complete training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_stats = self.train_epoch()
            
            # Validate
            val_stats = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_stats['total'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch statistics
            logger.info(f"\nEpoch {epoch}/{self.num_epochs - 1}")
            logger.info(f"  Learning rate: {current_lr:.2e}")
            logger.info(f"  Train - Loss: {train_stats['total']:.4f}, Acc@1: {train_stats.get('accuracy_top1', 0):.3f}, Acc@5: {train_stats.get('accuracy_top5', 0):.3f}")
            logger.info(f"  Val   - Loss: {val_stats['total']:.4f}, Acc@1: {val_stats.get('accuracy_top1', 0):.3f}, Acc@5: {val_stats.get('accuracy_top5', 0):.3f}")
            
            # Track history
            for key, value in train_stats.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_stats.items():
                self.training_history[f'val_{key}'].append(value)
            self.training_history['learning_rate'].append(current_lr)
            
            # Checkpointing
            if val_stats['total'] < self.best_val_loss:
                self.best_val_loss = val_stats['total']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt', epoch, val_stats)
                logger.info(f"  Saved best model (val loss: {val_stats['total']:.4f})")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement ({self.patience_counter}/{self.patience})")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_stats)
            
            # Update curriculum
            if self.curriculum_scheduler:
                self.curriculum_scheduler.step()
        
        # Save final model and training history
        self._save_checkpoint('final_model.pt', self.current_epoch, val_stats)
        self._save_training_history()
        
        logger.info("Training complete!")
        return self.training_history
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        batch['message_edge_index'] = batch['message_edge_index'].to(self.device)
        batch['message_node_attr'] = batch['message_node_attr'].to(self.device)
        batch['message_batch'] = batch['message_batch'].to(self.device)
        
        # Move token IDs
        batch['token_ids_list'] = [t.to(self.device) for t in batch['token_ids_list']]
        
        return batch
    
    def _save_checkpoint(self, filename: str, epoch: int, val_stats: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'val_stats': val_stats,
            'training_history': dict(self.training_history),
            'model_config': {
                'vocab_size': self.model.token_embeddings.num_embeddings,
                'token_embedding_dim': self.model.token_embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'message_dim': self.model.message_dim,
                'num_heads': 4,  # Could be stored in model
                'max_seq_length': self.model.position_embeddings.num_embeddings
            }
        }
        
        if self.use_adaptive_loss:
            checkpoint['adaptive_loss_state_dict'] = self.adaptive_loss_weighter.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(dict(self.training_history), f, indent=2)
        
        # Also create plots
        self._plot_training_history()
    
    def _plot_training_history(self):
        """Create training history plots"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss curves
            ax = axes[0, 0]
            ax.plot(self.training_history['train_total'], label='Train')
            ax.plot(self.training_history['val_total'], label='Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Total Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Accuracy curves
            ax = axes[0, 1]
            if 'train_accuracy_top1' in self.training_history:
                ax.plot(self.training_history['train_accuracy_top1'], label='Train Acc@1')
                ax.plot(self.training_history['val_accuracy_top1'], label='Val Acc@1')
                ax.plot(self.training_history['train_accuracy_top5'], label='Train Acc@5', linestyle='--')
                ax.plot(self.training_history['val_accuracy_top5'], label='Val Acc@5', linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Learning rate
            ax = axes[1, 0]
            ax.plot(self.training_history['learning_rate'])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Individual losses
            ax = axes[1, 1]
            for loss_name in ['relevance', 'contrastive', 'ranking', 'margin']:
                if f'train_{loss_name}' in self.training_history:
                    ax.plot(self.training_history[f'train_{loss_name}'], label=loss_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Individual Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'training_history.png', dpi=150)
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")