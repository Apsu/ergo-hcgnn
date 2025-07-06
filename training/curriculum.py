#!/usr/bin/env python3
"""
Curriculum learning strategies for hierarchical conversation GNN
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    # Conversation length curriculum
    initial_max_length: int = 5
    final_max_length: int = 50
    length_growth_rate: float = 2.0  # Increase by 2 messages per epoch
    
    # Dependency distance curriculum
    initial_max_dep_distance: int = 2
    final_max_dep_distance: int = 20
    dep_distance_growth_rate: float = 1.0
    
    # Complexity scoring weights
    length_weight: float = 0.3
    dep_distance_weight: float = 0.4
    num_dependencies_weight: float = 0.3
    
    # Sampling strategy
    use_temperature_sampling: bool = True
    initial_temperature: float = 2.0
    final_temperature: float = 0.5
    temperature_decay_rate: float = 0.95
    
    # Hard example mining
    mine_hard_examples: bool = True
    hard_example_ratio: float = 0.2
    hard_example_start_epoch: int = 5


class CurriculumScheduler:
    """
    Manages curriculum learning schedule for training
    """
    
    def __init__(self, config: CurriculumConfig, total_epochs: int):
        self.config = config
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.difficulty_history = []
        
    def get_current_limits(self) -> Dict[str, float]:
        """Get current curriculum limits based on epoch"""
        progress = min(self.current_epoch / max(self.total_epochs - 1, 1), 1.0)
        
        # Linear growth for limits
        current_max_length = self.config.initial_max_length + \
            (self.config.final_max_length - self.config.initial_max_length) * progress
            
        current_max_dep_distance = self.config.initial_max_dep_distance + \
            (self.config.final_max_dep_distance - self.config.initial_max_dep_distance) * progress
        
        # Exponential decay for temperature
        current_temperature = self.config.initial_temperature * \
            (self.config.temperature_decay_rate ** self.current_epoch)
        current_temperature = max(current_temperature, self.config.final_temperature)
        
        return {
            'max_length': int(current_max_length),
            'max_dep_distance': int(current_max_dep_distance),
            'temperature': current_temperature,
            'mine_hard_examples': self.config.mine_hard_examples and 
                                 self.current_epoch >= self.config.hard_example_start_epoch
        }
    
    def compute_complexity_score(self, conversation: Dict) -> float:
        """
        Compute complexity score for a conversation
        
        Args:
            conversation: Conversation with messages and metadata
            
        Returns:
            Complexity score (0-1)
        """
        messages = conversation.get('messages', conversation)
        
        # Length complexity
        length = len(messages)
        length_score = min(length / self.config.final_max_length, 1.0)
        
        # Dependency distance complexity
        max_dep_distance = 0
        total_dependencies = 0
        
        for msg in messages:
            if msg.get('depends_on_indices'):
                total_dependencies += len(msg['depends_on_indices'])
                msg_idx = messages.index(msg)
                for dep_idx in msg['depends_on_indices']:
                    max_dep_distance = max(max_dep_distance, msg_idx - dep_idx)
        
        dep_distance_score = min(max_dep_distance / self.config.final_max_dep_distance, 1.0)
        num_dep_score = min(total_dependencies / max(length, 1), 1.0)
        
        # Weighted combination
        complexity = (
            self.config.length_weight * length_score +
            self.config.dep_distance_weight * dep_distance_score +
            self.config.num_dependencies_weight * num_dep_score
        )
        
        return complexity
    
    def select_conversations(self, 
                           conversations: List[Dict],
                           batch_size: int,
                           hard_examples: Optional[List[int]] = None) -> List[int]:
        """
        Select conversations for current epoch based on curriculum
        
        Args:
            conversations: All available conversations
            batch_size: Number of conversations to select
            hard_examples: Indices of hard examples from previous epoch
            
        Returns:
            Selected conversation indices
        """
        limits = self.get_current_limits()
        
        # Filter conversations by current limits
        valid_indices = []
        complexity_scores = []
        
        for i, conv in enumerate(conversations):
            messages = conv.get('messages', conv)
            
            # Check length
            if len(messages) > limits['max_length']:
                continue
            
            # Check dependency distances
            valid = True
            for msg in messages:
                if msg.get('depends_on_indices'):
                    msg_idx = messages.index(msg)
                    for dep_idx in msg['depends_on_indices']:
                        if msg_idx - dep_idx > limits['max_dep_distance']:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                valid_indices.append(i)
                complexity_scores.append(self.compute_complexity_score(conv))
        
        # Handle hard example mining
        if limits['mine_hard_examples'] and hard_examples:
            num_hard = int(batch_size * self.config.hard_example_ratio)
            hard_valid = [idx for idx in hard_examples if idx in valid_indices]
            selected_hard = np.random.choice(
                hard_valid, 
                size=min(num_hard, len(hard_valid)), 
                replace=False
            ).tolist() if hard_valid else []
            
            # Remove hard examples from regular pool
            regular_indices = [idx for idx in valid_indices if idx not in selected_hard]
            regular_scores = [score for idx, score in zip(valid_indices, complexity_scores) 
                            if idx not in selected_hard]
        else:
            selected_hard = []
            regular_indices = valid_indices
            regular_scores = complexity_scores
        
        # Sample remaining from regular pool
        num_regular = batch_size - len(selected_hard)
        
        if self.config.use_temperature_sampling and regular_scores:
            # Temperature-based sampling favoring appropriate difficulty
            scores_array = np.array(regular_scores)
            
            # Convert to probabilities with temperature
            probabilities = np.exp(scores_array / limits['temperature'])
            probabilities = probabilities / probabilities.sum()
            
            selected_regular = np.random.choice(
                regular_indices,
                size=min(num_regular, len(regular_indices)),
                replace=False,
                p=probabilities
            ).tolist()
        else:
            # Uniform sampling
            selected_regular = np.random.choice(
                regular_indices,
                size=min(num_regular, len(regular_indices)),
                replace=False
            ).tolist()
        
        selected = selected_hard + selected_regular
        
        # Record difficulty statistics
        selected_complexities = [complexity_scores[valid_indices.index(idx)] 
                               for idx in selected if idx in valid_indices]
        self.difficulty_history.append({
            'epoch': self.current_epoch,
            'mean_complexity': np.mean(selected_complexities) if selected_complexities else 0,
            'num_selected': len(selected),
            'num_hard_examples': len(selected_hard)
        })
        
        return selected
    
    def step(self):
        """Advance to next epoch"""
        self.current_epoch += 1
    
    def get_statistics(self) -> Dict:
        """Get curriculum statistics"""
        return {
            'current_epoch': self.current_epoch,
            'current_limits': self.get_current_limits(),
            'difficulty_history': self.difficulty_history
        }


class HardExampleMiner:
    """
    Identifies hard examples based on model performance
    """
    
    def __init__(self, 
                 loss_history_size: int = 100,
                 hard_threshold_percentile: float = 75.0):
        self.loss_history_size = loss_history_size
        self.hard_threshold_percentile = hard_threshold_percentile
        self.loss_history = {}
        
    def update(self, indices: List[int], losses: List[float]):
        """Update loss history for conversations"""
        for idx, loss in zip(indices, losses):
            if idx not in self.loss_history:
                self.loss_history[idx] = []
            
            self.loss_history[idx].append(loss)
            
            # Keep only recent history
            if len(self.loss_history[idx]) > self.loss_history_size:
                self.loss_history[idx].pop(0)
    
    def get_hard_examples(self, min_history: int = 5) -> List[int]:
        """
        Get indices of hard examples based on loss history
        
        Args:
            min_history: Minimum number of loss values required
            
        Returns:
            List of hard example indices
        """
        # Compute average loss for each conversation
        avg_losses = {}
        for idx, losses in self.loss_history.items():
            if len(losses) >= min_history:
                avg_losses[idx] = np.mean(losses[-min_history:])
        
        if not avg_losses:
            return []
        
        # Find threshold for hard examples
        losses_array = np.array(list(avg_losses.values()))
        threshold = np.percentile(losses_array, self.hard_threshold_percentile)
        
        # Return indices with loss above threshold
        hard_indices = [idx for idx, loss in avg_losses.items() if loss >= threshold]
        
        return hard_indices
    
    def get_statistics(self) -> Dict:
        """Get mining statistics"""
        if not self.loss_history:
            return {'num_tracked': 0, 'num_hard': 0}
        
        avg_losses = {idx: np.mean(losses) for idx, losses in self.loss_history.items() 
                     if losses}
        
        return {
            'num_tracked': len(self.loss_history),
            'num_hard': len(self.get_hard_examples()),
            'mean_loss': np.mean(list(avg_losses.values())) if avg_losses else 0,
            'std_loss': np.std(list(avg_losses.values())) if avg_losses else 0
        }