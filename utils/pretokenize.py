#!/usr/bin/env python3
"""Pre-tokenize conversation data for faster training"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pretokenize_conversations(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 512
):
    """Pre-tokenize all conversations and save to disk"""
    
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    logger.info(f"Loading conversations from {input_path}")
    with open(input_path, 'r') as f:
        conversations = json.load(f)
    
    logger.info(f"Pre-tokenizing {len(conversations)} conversations...")
    tokenized_conversations = []
    
    for conv_idx, conv in enumerate(tqdm(conversations, desc="Tokenizing")):
        messages = conv['messages']
        
        # Tokenize all messages
        tokenized_messages = []
        for msg in messages:
            # Tokenize
            tokens = tokenizer(
                msg['text'],
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # Store tokenized data with original message info
            tokenized_msg = {
                'role': msg['role'],
                'token_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'length': len(tokens['input_ids']),
                'is_context_dependent': msg.get('is_context_dependent', False),
                'depends_on_indices': msg.get('depends_on_indices', []),
                'dependency_type': msg.get('dependency_type', None)
            }
            tokenized_messages.append(tokenized_msg)
        
        # Create tokenized conversation
        tokenized_conv = {
            'messages': tokenized_messages,
            'conversation_patterns': conv.get('conversation_patterns', []),
            'config': conv.get('config', {}),
            'conversation_idx': conv_idx
        }
        tokenized_conversations.append(tokenized_conv)
        
        if (conv_idx + 1) % 1000 == 0:
            logger.info(f"Processed {conv_idx + 1}/{len(conversations)} conversations")
    
    # Save tokenized data
    logger.info(f"Saving tokenized data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use torch.save for efficient storage
    torch.save({
        'conversations': tokenized_conversations,
        'tokenizer_name': tokenizer_name,
        'max_length': max_length,
        'vocab_size': tokenizer.vocab_size
    }, output_path)
    
    # Print statistics
    total_messages = sum(len(conv['messages']) for conv in tokenized_conversations)
    avg_tokens_per_msg = sum(
        msg['length'] for conv in tokenized_conversations 
        for msg in conv['messages']
    ) / total_messages
    
    logger.info(f"""
Tokenization complete:
  Conversations: {len(tokenized_conversations)}
  Total messages: {total_messages}
  Avg tokens per message: {avg_tokens_per_msg:.1f}
  Output file: {output_path}
  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB
""")


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize conversation data')
    parser.add_argument('--input', type=str, default='datasets/processed/conversations.json',
                       help='Input conversations file')
    parser.add_argument('--output', type=str, default='datasets/processed/conversations_tokenized.pt',
                       help='Output tokenized file')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                       help='Tokenizer to use')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum token length per message')
    
    args = parser.parse_args()
    
    pretokenize_conversations(
        Path(args.input),
        Path(args.output),
        args.tokenizer,
        args.max_length
    )


if __name__ == "__main__":
    main()