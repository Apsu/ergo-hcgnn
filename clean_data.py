#!/usr/bin/env python3
"""
Clean and validate conversation data
Removes invalid dependencies and saves to processed directory
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_and_clean_dependencies(messages: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Validate and clean dependency indices in messages
    
    Returns:
        Cleaned messages and statistics about fixes
    """
    stats = {
        'total_messages': 0,
        'messages_with_deps': 0,
        'self_references_removed': 0,
        'future_references_removed': 0,
        'negative_references_removed': 0,
        'total_deps_removed': 0,
        'messages_fixed': 0
    }
    
    cleaned_messages = []
    
    for i, msg in enumerate(messages):
        stats['total_messages'] += 1
        cleaned_msg = msg.copy()
        
        if 'depends_on_indices' in msg and msg['depends_on_indices']:
            stats['messages_with_deps'] += 1
            original_deps = msg['depends_on_indices']
            
            # Filter out invalid dependencies
            valid_deps = []
            removed_deps = []
            
            for dep in original_deps:
                if dep == i:
                    # Self-reference
                    removed_deps.append((dep, 'self-reference'))
                    stats['self_references_removed'] += 1
                elif dep >= i:
                    # Future reference
                    removed_deps.append((dep, 'future-reference'))
                    stats['future_references_removed'] += 1
                elif dep < 0:
                    # Negative index
                    removed_deps.append((dep, 'negative-index'))
                    stats['negative_references_removed'] += 1
                else:
                    # Valid dependency
                    valid_deps.append(dep)
            
            # Update message with cleaned dependencies
            cleaned_msg['depends_on_indices'] = valid_deps
            
            # If we removed any dependencies, update stats
            if len(removed_deps) > 0:
                stats['total_deps_removed'] += len(removed_deps)
                stats['messages_fixed'] += 1
                
                # Log details for first few fixes
                if stats['messages_fixed'] <= 5:
                    logger.info(f"Fixed message {i}: removed {removed_deps} from {original_deps}")
            
            # If all dependencies were invalid, mark as non-dependent
            if len(valid_deps) == 0 and len(original_deps) > 0:
                cleaned_msg['is_context_dependent'] = False
                cleaned_msg['dependency_type'] = 'none'
        
        cleaned_messages.append(cleaned_msg)
    
    return cleaned_messages, stats


def clean_conversations(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Clean all conversations in a file
    
    Returns:
        Aggregated statistics
    """
    logger.info(f"Loading conversations from {input_path}")
    
    with open(input_path, 'r') as f:
        conversations = json.load(f)
    
    logger.info(f"Loaded {len(conversations)} conversations")
    
    # Aggregate statistics
    total_stats = {
        'total_conversations': len(conversations),
        'total_messages': 0,
        'messages_with_deps': 0,
        'self_references_removed': 0,
        'future_references_removed': 0,
        'negative_references_removed': 0,
        'total_deps_removed': 0,
        'messages_fixed': 0,
        'conversations_modified': 0
    }
    
    cleaned_conversations = []
    
    for i, conv in enumerate(conversations):
        # Get messages
        messages = conv.get('messages', conv)
        
        # Clean dependencies
        cleaned_messages, stats = validate_and_clean_dependencies(messages)
        
        # Update aggregate statistics
        for key in stats:
            if key in total_stats:
                total_stats[key] += stats[key]
        
        # Check if conversation was modified
        if stats['messages_fixed'] > 0:
            total_stats['conversations_modified'] += 1
        
        # Create cleaned conversation
        if isinstance(conv, dict) and 'messages' in conv:
            cleaned_conv = conv.copy()
            cleaned_conv['messages'] = cleaned_messages
        else:
            # If conv is just a list of messages
            cleaned_conv = {'messages': cleaned_messages}
        
        cleaned_conversations.append(cleaned_conv)
        
        # Progress update
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(conversations)} conversations")
    
    # Save cleaned conversations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(cleaned_conversations, f, indent=2)
    
    logger.info(f"Saved cleaned conversations to {output_path}")
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(description='Clean and validate conversation data')
    parser.add_argument('--input-dir', type=str, default='datasets/raw',
                       help='Input directory containing raw conversations')
    parser.add_argument('--output-dir', type=str, default='datasets/processed',
                       help='Output directory for cleaned conversations')
    parser.add_argument('--file-pattern', type=str, default='*.json',
                       help='File pattern to match')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all matching files
    input_files = list(input_dir.glob(args.file_pattern))
    
    if not input_files:
        logger.error(f"No files found matching {args.file_pattern} in {input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Aggregate statistics across all files
    all_stats = {}
    
    for input_file in input_files:
        if input_file.name == 'conversations_metadata.json':
            # Just copy metadata file
            output_file = output_dir / input_file.name
            with open(input_file, 'r') as f:
                metadata = json.load(f)
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Copied metadata file to {output_file}")
            continue
        
        output_file = output_dir / input_file.name
        logger.info(f"\nProcessing {input_file.name}...")
        
        stats = clean_conversations(input_file, output_file)
        
        # Merge stats
        for key, value in stats.items():
            all_stats[key] = all_stats.get(key, 0) + value
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("CLEANING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total conversations processed: {all_stats.get('total_conversations', 0):,}")
    logger.info(f"Total messages processed: {all_stats.get('total_messages', 0):,}")
    logger.info(f"Messages with dependencies: {all_stats.get('messages_with_deps', 0):,}")
    logger.info(f"Conversations modified: {all_stats.get('conversations_modified', 0):,}")
    logger.info(f"Messages fixed: {all_stats.get('messages_fixed', 0):,}")
    
    if all_stats.get('total_deps_removed', 0) > 0:
        logger.info(f"\nDependencies removed:")
        logger.info(f"  Self-references: {all_stats.get('self_references_removed', 0):,}")
        logger.info(f"  Future references: {all_stats.get('future_references_removed', 0):,}")
        logger.info(f"  Negative indices: {all_stats.get('negative_references_removed', 0):,}")
        logger.info(f"  Total removed: {all_stats.get('total_deps_removed', 0):,}")
        
        # Calculate percentages
        total_msgs = all_stats.get('total_messages', 1)
        pct_fixed = 100 * all_stats.get('messages_fixed', 0) / total_msgs
        logger.info(f"\nPercentage of messages fixed: {pct_fixed:.2f}%")
    else:
        logger.info("\nNo invalid dependencies found - data is already clean!")
    
    # Save cleaning report
    from datetime import datetime
    report_path = output_dir / 'cleaning_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'statistics': all_stats,
            'processed_files': [str(f.name) for f in input_files],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\nCleaning report saved to {report_path}")


if __name__ == "__main__":
    main()