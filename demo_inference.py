#!/usr/bin/env python3
"""
Demo script showing how to use the inference module
"""

import json
from pathlib import Path
from inference import create_context_selector


def main():
    # Create a context selector
    selector = create_context_selector(
        model_path='checkpoints/best_model.pt',
        streaming=False,  # Use False for batch processing
        device='cuda'
    )
    
    # Example conversation
    conversation = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
        {"role": "user", "content": "What are the main types?"},
        {"role": "assistant", "content": "The main types of machine learning are: supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction and rewards)."},
        {"role": "user", "content": "Can you give me an example of supervised learning?"},
        {"role": "assistant", "content": "A classic example of supervised learning is email spam classification. The model is trained on a dataset of emails labeled as 'spam' or 'not spam', and learns to classify new emails based on patterns it discovered."},
        {"role": "user", "content": "What about unsupervised?"},
        {"role": "assistant", "content": "Customer segmentation is a good example of unsupervised learning. A retail company might use clustering algorithms to group customers based on purchasing behavior, without predefined categories."},
    ]
    
    # New query
    query = "How does reinforcement learning differ from the types we discussed?"
    
    print("=== Conversation Context Selection Demo ===\n")
    print(f"Conversation has {len(conversation)} messages")
    print(f"Query: {query}\n")
    
    # Select context
    result = selector.select_context(
        messages=conversation,
        query=query,
        max_context=3,
        temperature=1.5,
        min_score_threshold=None
    )
    
    # Display results
    print("Selected context messages:")
    for i, idx in enumerate(result['selected_indices']):
        msg = conversation[idx]
        score = result['scores'][idx]
        prob = result['probabilities'][idx]
        print(f"\n{i+1}. Message {idx} (Score: {score:.3f}, Prob: {prob:.3f}):")
        print(f"   Role: {msg['role']}")
        print(f"   Content: {msg['content'][:100]}...")
    
    print(f"\nTotal messages selected: {len(result['selected_indices'])}")
    print(f"Cache size: {result['debug_info']['cache_size']}")
    
    # Example with streaming selector
    print("\n\n=== Streaming Context Selection Demo ===\n")
    
    streaming_selector = create_context_selector(
        model_path='checkpoints/best_model.pt',
        streaming=True,
        device='cuda'
    )
    
    # Add messages one by one
    for msg in conversation:
        streaming_selector.add_message(msg)
    
    # Select context for the same query
    streaming_result = streaming_selector.select_context_streaming(
        query=query,
        max_context=3,
        temperature=1.5
    )
    
    print("Streaming selection results:")
    print(f"Selected indices: {streaming_result['selected_indices']}")
    
    # Demonstrate batch processing
    print("\n\n=== Batch Processing Demo ===\n")
    
    # Multiple conversations and queries
    conversations = [conversation, conversation[:4]]  # Two different conversations
    queries = [
        "How does reinforcement learning differ from the types we discussed?",
        "Tell me more about the main types"
    ]
    
    batch_results = selector.batch_select_context(
        conversations=conversations,
        queries=queries,
        max_context=2,
        temperature=1.0
    )
    
    for i, (conv, query, result) in enumerate(zip(conversations, queries, batch_results)):
        print(f"\nConversation {i+1} ({len(conv)} messages):")
        print(f"Query: {query}")
        print(f"Selected indices: {result['selected_indices']}")
    
    # Clear cache
    selector.clear_cache()
    print("\n\nCache cleared!")


if __name__ == "__main__":
    main()