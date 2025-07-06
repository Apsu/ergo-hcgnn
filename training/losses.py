#!/usr/bin/env python3
"""
Loss functions for hierarchical conversation GNN training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for learning better message representations
    Groups messages by dependency type or role
    """
    
    def __init__(self, temperature: float = 0.07, use_hard_negatives: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
    def forward(self, 
                embeddings: torch.Tensor, 
                labels: torch.Tensor,
                hard_negatives_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: [N, D] message embeddings
            labels: [N] dependency type labels or role labels
            hard_negatives_mask: [N, N] boolean mask for hard negatives
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).float()
        
        # Remove diagonal
        positive_mask.fill_diagonal_(0)
        
        # Hard negative weighting if provided
        if self.use_hard_negatives and hard_negatives_mask is not None:
            # Give more weight to hard negatives
            negative_mask = 1 - positive_mask
            hard_negative_weight = 2.0
            weights = torch.where(
                hard_negatives_mask & negative_mask.bool(),
                torch.tensor(hard_negative_weight, device=embeddings.device),
                torch.tensor(1.0, device=embeddings.device)
            )
            sim_matrix = sim_matrix * weights
        
        # For numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        
        # Compute log probabilities with better numerical stability
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # Avoid division by zero
        num_positives = positive_mask.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / num_positives
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ListNetRankingLoss(nn.Module):
    """
    ListNet ranking loss for better ordering of context messages
    Learns to rank relevant messages higher than irrelevant ones
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, 
                scores: torch.Tensor, 
                relevance_labels: torch.Tensor,
                position_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_items] predicted relevance scores
            relevance_labels: [batch_size, num_items] ground truth relevance (1 for relevant)
            position_bias: [batch_size, num_items] position-based bias to add
        """
        batch_size, num_items = scores.shape
        
        # Add position bias if provided
        if position_bias is not None:
            scores = scores + position_bias
        
        # Apply temperature
        scores = scores / self.temperature
        
        # Convert scores to probabilities
        pred_probs = F.softmax(scores, dim=-1)
        
        # Create target distribution
        # For each batch, normalize relevance labels to create probability distribution
        relevance_sum = relevance_labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target_probs = relevance_labels / relevance_sum
        
        # KL divergence between predicted and target distributions
        # Use log_softmax for numerical stability
        log_pred_probs = F.log_softmax(scores, dim=-1)
        loss = F.kl_div(log_pred_probs, target_probs, reduction='batchmean')
        
        return loss


class MarginRankingLoss(nn.Module):
    """
    Margin-based ranking loss that ensures relevant messages
    score higher than irrelevant ones by a margin
    """
    
    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self,
                scores: torch.Tensor,
                relevance_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_items] predicted scores
            relevance_labels: [batch_size, num_items] binary relevance
        """
        batch_size, num_items = scores.shape
        device = scores.device
        
        # Create masks for relevant and irrelevant items
        relevant_mask = relevance_labels > 0
        irrelevant_mask = relevance_labels == 0
        
        # Check if we have both relevant and irrelevant items
        has_relevant = relevant_mask.any(dim=1)
        has_irrelevant = irrelevant_mask.any(dim=1)
        has_both = has_relevant & has_irrelevant
        
        if not has_both.any():
            return torch.tensor(0.0, device=device)
        
        # Expand scores for pairwise comparisons
        relevant_scores = scores.unsqueeze(2)
        irrelevant_scores = scores.unsqueeze(1)
        
        # Compute all pairwise differences
        score_diffs = relevant_scores - irrelevant_scores
        
        # Apply margin and ReLU
        losses = F.relu(self.margin - score_diffs)
        
        # Mask out invalid pairs (only keep pairs where we have relevant-irrelevant)
        valid_pairs = (relevant_mask.unsqueeze(2) & irrelevant_mask.unsqueeze(1))
        losses = losses * valid_pairs.float()
        
        # Count valid pairs
        total_pairs = valid_pairs.sum()
        
        if total_pairs > 0:
            if self.reduction == 'mean':
                return losses.sum() / total_pairs
            else:
                return losses.sum()
        else:
            return torch.tensor(0.0, device=device)


class DependencyAwareLoss(nn.Module):
    """
    Weighted combination of losses with dependency-type-specific weighting
    """
    
    def __init__(self,
                 relevance_weight: float = 1.0,
                 contrastive_weight: float = 0.1,
                 ranking_weight: float = 0.3,
                 margin_weight: float = 0.1,
                 dependency_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.relevance_weight = relevance_weight
        self.contrastive_weight = contrastive_weight
        self.ranking_weight = ranking_weight
        self.margin_weight = margin_weight
        
        # Dependency-specific loss weights
        self.dependency_weights = dependency_weights or {
            'continuation': 1.2,
            'pronoun_reference': 1.5,
            'clarification': 1.3,
            'topic_reference': 1.0,
            'agreement': 1.1,
            'follow_up': 1.1,
            'example_request': 1.0,
            'none': 0.8
        }
        
        # Initialize loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.ranking_loss = ListNetRankingLoss(temperature=1.0)
        self.margin_loss = MarginRankingLoss(margin=0.5)
        
    def compute_relevance_loss(self,
                              attention_weights: torch.Tensor,
                              target_weights: torch.Tensor,
                              dependency_types: Optional[List[str]] = None) -> torch.Tensor:
        """
        Compute KL divergence loss between predicted and target attention distributions
        
        Args:
            attention_weights: [batch_size, num_context] predicted attention
            target_weights: [batch_size, num_context] target distribution
            dependency_types: List of dependency types for weighting
        """
        # KL divergence loss
        log_attention = torch.log(attention_weights + 1e-10)
        kl_loss = F.kl_div(log_attention, target_weights, reduction='none').sum(dim=-1)
        
        # Apply dependency-specific weights if provided
        if dependency_types is not None:
            weights = torch.tensor(
                [self.dependency_weights.get(dt, 1.0) for dt in dependency_types],
                device=attention_weights.device
            )
            kl_loss = kl_loss * weights
        
        return kl_loss.mean()
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                message_embeddings: torch.Tensor,
                dependency_types: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        
        Args:
            predictions: Dict with 'attention_weights', 'relevance_scores'
            targets: Dict with 'target_weights', 'relevance_labels', 'dependency_labels'
            message_embeddings: Message embeddings for contrastive loss
            dependency_types: List of dependency types
            
        Returns:
            Dict of individual losses and total loss
        """
        losses = {}
        
        # 1. Relevance loss (KL divergence)
        if 'attention_weights' in predictions and 'target_weights' in targets:
            relevance_loss = self.compute_relevance_loss(
                predictions['attention_weights'],
                targets['target_weights'],
                dependency_types
            )
            losses['relevance'] = relevance_loss * self.relevance_weight
        
        # 2. Contrastive loss
        if self.contrastive_weight > 0 and 'dependency_labels' in targets:
            contrastive_loss = self.contrastive_loss(
                message_embeddings,
                targets['dependency_labels']
            )
            losses['contrastive'] = contrastive_loss * self.contrastive_weight
        
        # 3. Ranking losses
        if 'relevance_scores' in predictions and 'relevance_labels' in targets:
            if self.ranking_weight > 0:
                ranking_loss = self.ranking_loss(
                    predictions['relevance_scores'],
                    targets['relevance_labels']
                )
                losses['ranking'] = ranking_loss * self.ranking_weight
            
            if self.margin_weight > 0:
                margin_loss = self.margin_loss(
                    predictions['relevance_scores'],
                    targets['relevance_labels']
                )
                losses['margin'] = margin_loss * self.margin_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class AdaptiveLossWeighting(nn.Module):
    """
    Learns to weight multiple losses based on their uncertainties
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(self, num_losses: int = 4):
        super().__init__()
        # Log variance parameters for each loss
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine losses with learned uncertainty weights
        
        Args:
            losses: List of loss tensors
            
        Returns:
            Weighted total loss
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss