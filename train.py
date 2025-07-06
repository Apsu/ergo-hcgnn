#!/usr/bin/env python3
"""
Main training script for Hierarchical Conversation GNN V2
"""

import argparse
import json
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

from models.hierarchical_gnn import HierarchicalConversationGNN
from data.dataset import ConversationDataset
from training.trainer import HierarchicalGNNTrainer
from training.curriculum import CurriculumConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_conversations(data_paths: list, pre_tokenized: bool = False) -> tuple:
    """Load conversations from multiple files
    
    Returns:
        (conversations, vocab_size) - vocab_size is None for non-tokenized data
    """
    all_conversations = []
    vocab_size = None
    
    for path in data_paths:
        logger.info(f"Loading conversations from {path}")
        if pre_tokenized:
            # Load pre-tokenized data
            data = torch.load(path)
            conversations = data['conversations']
            if vocab_size is None and 'vocab_size' in data:
                vocab_size = data['vocab_size']
            all_conversations.extend(conversations)
        else:
            # Load raw JSON
            with open(path, 'r') as f:
                conversations = json.load(f)
                all_conversations.extend(conversations)
    
    logger.info(f"Loaded {len(all_conversations)} total conversations")
    return all_conversations, vocab_size


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Conversation GNN V2')
    
    # Data arguments
    parser.add_argument('--data-paths', nargs='+', required=True,
                       help='Paths to conversation JSON files')
    parser.add_argument('--pre-tokenized', action='store_true',
                       help='Whether data is pre-tokenized (.pt files)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--max-messages', type=int, default=None,
                       help='Maximum messages per conversation')
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=30522,
                       help='Vocabulary size')
    parser.add_argument('--token-embedding-dim', type=int, default=128,
                       help='Token embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--message-dim', type=int, default=128,
                       help='Message embedding dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--max-seq-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--num-token-gat-layers', type=int, default=2,
                       help='Number of token GAT layers')
    parser.add_argument('--num-message-gat-layers', type=int, default=2,
                       help='Number of message GAT layers')
    parser.add_argument('--use-cross-message-attention', action='store_true',
                       help='Use cross-message token attention')
    parser.add_argument('--use-semantic-edges', action='store_true',
                       help='Add semantic edges in message graph')
    parser.add_argument('--window-sizes', type=int, nargs='+', default=[1, 2, 3],
                       help='Window sizes for token connections')
    
    # Training arguments
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                       help='Tokenizer to use')
    parser.add_argument('--max-length', type=int, default=128,
                       help='Maximum token length per message')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers (0 for main process)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Loss weights
    parser.add_argument('--relevance-weight', type=float, default=1.0,
                       help='Weight for relevance loss')
    parser.add_argument('--contrastive-weight', type=float, default=0.1,
                       help='Weight for contrastive loss')
    parser.add_argument('--ranking-weight', type=float, default=0.3,
                       help='Weight for ranking loss')
    parser.add_argument('--margin-weight', type=float, default=0.1,
                       help='Weight for margin ranking loss')
    
    # Curriculum learning
    parser.add_argument('--use-curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--use-adaptive-loss', action='store_true',
                       help='Use adaptive loss weighting')
    parser.add_argument('--curriculum-initial-length', type=int, default=5,
                       help='Initial max conversation length for curriculum')
    parser.add_argument('--curriculum-final-length', type=int, default=50,
                       help='Final max conversation length for curriculum')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=5,
                       help='Checkpoint frequency')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved configuration to {config_path}")
    
    # Load conversations
    conversations, data_vocab_size = load_conversations(args.data_paths, args.pre_tokenized)
    
    # Shuffle and split data
    np.random.shuffle(conversations)
    
    val_size = int(len(conversations) * args.val_split)
    test_size = int(len(conversations) * args.test_split)
    
    test_conversations = conversations[:test_size]
    val_conversations = conversations[test_size:test_size + val_size]
    train_conversations = conversations[test_size + val_size:]
    
    logger.info(f"Data split - Train: {len(train_conversations)}, Val: {len(val_conversations)}, Test: {len(test_conversations)}")
    
    # Save test set for later evaluation
    if test_conversations:
        if args.pre_tokenized:
            test_path = output_dir / 'test_conversations.pt'
            torch.save({'conversations': test_conversations}, test_path)
        else:
            test_path = output_dir / 'test_conversations.json'
            with open(test_path, 'w') as f:
                json.dump(test_conversations, f, indent=2)
        logger.info(f"Saved test set to {test_path}")
    
    # Initialize tokenizer (only needed if not pre-tokenized)
    tokenizer = None
    vocab_size = args.vocab_size
    
    if not args.pre_tokenized:
        logger.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_size = tokenizer.vocab_size
    else:
        # For pre-tokenized data, use vocab size from data file
        if data_vocab_size is not None:
            vocab_size = data_vocab_size
            logger.info(f"Using vocab_size from pre-tokenized data: {vocab_size}")
        else:
            logger.info(f"Using default vocab_size: {vocab_size}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ConversationDataset(
        train_conversations,
        tokenizer,
        max_length=args.max_length,
        max_messages=args.max_messages,
        include_system_messages=True,
        pre_tokenized=args.pre_tokenized
    )
    
    val_dataset = ConversationDataset(
        val_conversations,
        tokenizer,
        max_length=args.max_length,
        max_messages=args.max_messages,
        include_system_messages=True,
        pre_tokenized=args.pre_tokenized
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = HierarchicalConversationGNN(
        vocab_size=vocab_size,
        token_embedding_dim=args.token_embedding_dim,
        hidden_dim=args.hidden_dim,
        message_dim=args.message_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        num_token_gat_layers=args.num_token_gat_layers,
        num_message_gat_layers=args.num_message_gat_layers,
        use_cross_message_attention=args.use_cross_message_attention,
        use_semantic_edges=args.use_semantic_edges,
        window_sizes=args.window_sizes
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from:
        logger.info(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Setup curriculum config
    curriculum_config = None
    if args.use_curriculum:
        curriculum_config = {
            'initial_max_length': args.curriculum_initial_length,
            'final_max_length': args.curriculum_final_length,
            'initial_max_dep_distance': 2,
            'final_max_dep_distance': 20,
            'use_temperature_sampling': True,
            'mine_hard_examples': True
        }
    
    # Initialize trainer
    trainer = HierarchicalGNNTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        num_epochs=args.num_epochs,
        device=args.device,
        num_workers=args.num_workers,
        use_curriculum=args.use_curriculum,
        use_adaptive_loss=args.use_adaptive_loss,
        checkpoint_every=args.checkpoint_every,
        patience=args.patience,
        relevance_weight=args.relevance_weight,
        contrastive_weight=args.contrastive_weight,
        ranking_weight=args.ranking_weight,
        margin_weight=args.margin_weight,
        weight_decay=args.weight_decay,
        curriculum_config=curriculum_config
    )
    
    # Load optimizer state if resuming
    if args.resume_from:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = start_epoch
        if 'training_history' in checkpoint:
            trainer.training_history.update(checkpoint['training_history'])
    
    # Train
    logger.info("\nStarting training...")
    training_history = trainer.train()
    
    logger.info("\nTraining complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    
    # Save final summary
    summary = {
        'best_val_loss': trainer.best_val_loss,
        'final_epoch': trainer.current_epoch,
        'training_time': datetime.now().isoformat(),
        'model_params': sum(p.numel() for p in model.parameters()),
        'args': vars(args)
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
