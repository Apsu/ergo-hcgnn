#!/usr/bin/env python3
"""
Interactive LLM chat with Hierarchical Conversation GNN V2 context selection
Enhanced with prompt_toolkit for better UX and visualization commands
"""

import os
import sys
import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# Add the hierarchical_gnn_v2 directory to path if needed
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import print_formatted_text

from openai import OpenAI
from inference import StreamingContextSelector


class ChatInterface:
    """Enhanced chat interface with visualization and debugging features"""
    
    def __init__(self, model_path: str, temperature: float = 0.7, device: str = 'cuda'):
        """Initialize chat interface with model and settings"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature
        self.verbose = False
        self.show_scores = False
        self.max_context_messages = 10
        
        # Initialize the streaming context selector
        print(f"Loading model from {model_path}...")
        self.context_selector = StreamingContextSelector(
            model_path=model_path,
            device=self.device,
            tokenizer_name='bert-base-uncased'
        )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Conversation history - using selector's buffer
        self.selected_contexts = []  # Track which messages were selected
        self.selection_results = []  # Track full selection results
        
        # Statistics
        self.stats = {
            'total_messages': 0,
            'total_tokens_sent': 0,
            'avg_context_selected': 0,
            'selection_distribution': defaultdict(int)
        }
        
        # Setup prompt toolkit
        self.setup_prompt_toolkit()
        
        print(f"Model loaded successfully! Device: {self.device}")
        print("Type /help for available commands")
    
    def setup_prompt_toolkit(self):
        """Setup prompt toolkit with completions and styling"""
        commands = ['/help', '/verbose', '/scores', '/stats', '/graph', '/clear', 
                   '/save', '/load', '/context', '/temperature', '/quit']
        self.completer = WordCompleter(commands, ignore_case=True)
        
        self.style = Style.from_dict({
            'prompt': 'fg:green bold',
            'ai': 'fg:blue',
            'info': 'fg:gray italic',
            'error': 'fg:red bold',
        })
        
        self.history = FileHistory('.chat_history.txt')
    
    def get_prompt_text(self):
        """Get formatted prompt text"""
        mode_indicators = []
        if self.verbose:
            mode_indicators.append('verbose')
        if self.show_scores:
            mode_indicators.append('scores')
        
        mode_str = f" [{', '.join(mode_indicators)}]" if mode_indicators else ""
        return HTML(f'<prompt>You{mode_str}> </prompt>')
    
    @property
    def messages(self):
        """Get conversation messages from selector buffer"""
        return self.context_selector.conversation_buffer
    
    def add_message(self, message: Dict[str, str]):
        """Add a message to the conversation"""
        self.context_selector.add_message(message)
    
    def format_context_for_llm(self, selected_indices: List[int], query: str) -> List[Dict]:
        """Format selected context messages for the LLM"""
        context_messages = []
        
        # Add system message
        context_messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant. Respond based on the conversation context."
        })
        
        # Add selected context messages
        for idx in sorted(selected_indices):
            context_messages.append(self.messages[idx])
        
        # Add current query
        context_messages.append({"role": "user", "content": query})
        
        return context_messages
    
    def print_verbose_info(self, selected_indices: List[int], scores: List[float], probs: List[float]):
        """Print verbose selection information"""
        print_formatted_text(HTML('<info>\n--- Context Selection ---</info>'), style=self.style)
        
        for i, (score, prob) in enumerate(zip(scores, probs)):
            marker = "→" if i in selected_indices else " "
            msg_preview = self.messages[i]['content'][:50] + "..." if len(self.messages[i]['content']) > 50 else self.messages[i]['content']
            msg_preview = msg_preview.replace('\n', ' ')
            
            print(f"{marker} [{i:2d}] Score: {score:6.3f} | Prob: {prob:5.3f} | {self.messages[i]['role']:9s} | {msg_preview}")
        
        print_formatted_text(HTML('<info>--- End Selection ---\n</info>'), style=self.style)
    
    def show_statistics(self):
        """Display conversation statistics"""
        print_formatted_text(HTML('<info>\n--- Conversation Statistics ---</info>'), style=self.style)
        print(f"Total messages: {self.stats['total_messages']}")
        print(f"Total tokens sent to LLM: {self.stats['total_tokens_sent']:,}")
        print(f"Average context messages selected: {self.stats['avg_context_selected']:.2f}")
        
        print("\nSelection frequency by position:")
        if self.stats['selection_distribution']:
            max_pos = max(self.stats['selection_distribution'].keys())
            for i in range(max_pos + 1):
                count = self.stats['selection_distribution'].get(i, 0)
                bar = "█" * int(count * 20 / max(self.stats['selection_distribution'].values()))
                print(f"  Position {i:2d}: {bar} ({count})")
        
        print_formatted_text(HTML('<info>--- End Statistics ---</info>'), style=self.style)
    
    def visualize_attention_graph(self, selected_indices: List[int], scores: List[float]):
        """Visualize the attention graph"""
        plt.figure(figsize=(10, 8))
        
        # Create graph
        G = nx.DiGraph()
        num_messages = len(self.messages) + 1  # Including current query
        
        # Add nodes
        for i in range(num_messages):
            if i < len(self.messages):
                label = f"{i}: {self.messages[i]['role'][:3]}"
                color = 'lightblue' if i in selected_indices else 'lightgray'
            else:
                label = f"{i}: query"
                color = 'lightgreen'
            G.add_node(i, label=label, color=color)
        
        # Add edges based on attention scores
        query_idx = num_messages - 1
        for i, score in enumerate(scores):
            if score > -2:  # Only show stronger connections
                G.add_edge(query_idx, i, weight=float(score + 3))  # Shift scores to positive
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.9)
        
        # Draw edges with varying width based on score
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                              width=weights, alpha=0.6, arrowsize=20)
        
        # Draw labels
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title(f"Context Selection Graph (Temperature: {self.temperature})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if should continue, False to quit"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            print("""
Available commands:
  /help              - Show this help message
  /verbose           - Toggle verbose mode (show context selection)
  /scores            - Toggle showing attention scores
  /stats             - Show conversation statistics
  /graph             - Visualize the last context selection
  /clear             - Clear conversation history
  /save <filename>   - Save conversation to file
  /load <filename>   - Load conversation from file
  /context <n>       - Set max context messages (current: {})
  /temperature <t>   - Set selection temperature (current: {:.2f})
  /quit              - Exit the chat
            """.format(self.max_context_messages, self.temperature))
        
        elif cmd == '/verbose':
            self.verbose = not self.verbose
            print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
        
        elif cmd == '/scores':
            self.show_scores = not self.show_scores
            print(f"Show scores: {'ON' if self.show_scores else 'OFF'}")
        
        elif cmd == '/stats':
            self.show_statistics()
        
        elif cmd == '/graph':
            if self.selected_contexts and self.selection_results:
                last_result = self.selection_results[-1]
                self.visualize_attention_graph(
                    self.selected_contexts[-1], 
                    last_result['scores']
                )
            else:
                print("No context selection to visualize yet.")
        
        elif cmd == '/clear':
            self.context_selector.clear_conversation()
            self.selected_contexts = []
            self.selection_results = []
            print("Conversation cleared.")
        
        elif cmd == '/save':
            if len(parts) > 1:
                filename = parts[1]
                with open(filename, 'w') as f:
                    json.dump({
                        'messages': list(self.messages),
                        'stats': dict(self.stats)
                    }, f, indent=2)
                print(f"Conversation saved to {filename}")
            else:
                print("Usage: /save <filename>")
        
        elif cmd == '/load':
            if len(parts) > 1:
                filename = parts[1]
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        # Clear and reload messages
                        self.context_selector.clear_conversation()
                        for msg in data['messages']:
                            self.context_selector.add_message(msg)
                        self.stats = defaultdict(int, data.get('stats', {}))
                    print(f"Conversation loaded from {filename}")
                except Exception as e:
                    print(f"Error loading file: {e}")
            else:
                print("Usage: /load <filename>")
        
        elif cmd == '/context':
            if len(parts) > 1:
                try:
                    self.max_context_messages = int(parts[1])
                    print(f"Max context messages set to {self.max_context_messages}")
                except ValueError:
                    print("Usage: /context <number>")
            else:
                print(f"Current max context messages: {self.max_context_messages}")
        
        elif cmd == '/temperature':
            if len(parts) > 1:
                try:
                    self.temperature = float(parts[1])
                    print(f"Temperature set to {self.temperature}")
                except ValueError:
                    print("Usage: /temperature <float>")
            else:
                print(f"Current temperature: {self.temperature}")
        
        elif cmd == '/quit':
            return False
        
        else:
            print(f"Unknown command: {command}")
        
        return True
    
    def chat_loop(self):
        """Main chat loop"""
        print("\nWelcome to the Hierarchical GNN-powered chat!")
        print("The model will intelligently select relevant context from the conversation.")
        print()
        
        while True:
            try:
                # Get user input
                user_input = prompt(
                    self.get_prompt_text(),
                    history=self.history,
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=self.completer,
                    style=self.style
                ).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Select context using the inference API
                start_time = time.time()
                result = self.context_selector.select_context_streaming(
                    query=user_input,
                    max_context=self.max_context_messages,
                    temperature=self.temperature,
                    min_score_threshold=None
                )
                selection_time = time.time() - start_time
                
                # Extract data from result
                selected_indices = result['selected_indices']
                
                # Store for visualization
                self.selected_contexts.append(selected_indices)
                self.selection_results.append(result)
                
                # Update statistics
                self.stats['total_messages'] += 1
                for idx in selected_indices:
                    self.stats['selection_distribution'][idx] += 1
                
                # Show verbose info if enabled
                if self.verbose:
                    self.print_verbose_info(
                        selected_indices, 
                        result['scores'],
                        result['probabilities']
                    )
                
                # Format context for LLM
                llm_messages = self.format_context_for_llm(selected_indices, user_input)
                
                # Count tokens (approximate)
                total_tokens = sum(len(msg['content'].split()) * 1.3 for msg in llm_messages)
                self.stats['total_tokens_sent'] += int(total_tokens)
                
                # Get LLM response
                response = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=llm_messages,
                    # temperature=0.7,
                    # max_tokens=500
                )
                
                assistant_message = response.choices[0].message.content
                
                # Print response
                print_formatted_text(HTML(f'<ai>Assistant: {assistant_message}</ai>'), style=self.style)
                
                if self.show_scores:
                    print_formatted_text(HTML(f'<info>  [Selected {len(selected_indices)} messages in {selection_time:.3f}s]</info>'), style=self.style)
                
                # Update conversation history
                self.add_message({"role": "user", "content": user_input})
                self.add_message({"role": "assistant", "content": assistant_message})
                
                # Update average context selected
                total_selections = sum(len(s) for s in self.selected_contexts)
                self.stats['avg_context_selected'] = total_selections / len(self.selected_contexts)
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\nUse /quit to exit")
                continue
            except Exception as e:
                print_formatted_text(HTML(f'<error>Error: {e}</error>'), style=self.style)
                import traceback
                traceback.print_exc()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive chat with Hierarchical GNN context selection')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--temperature', type=float, default=1.5,
                       help='Temperature for context selection')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Create and run chat interface
    chat = ChatInterface(
        model_path=args.model,
        temperature=args.temperature,
        device=args.device
    )
    
    try:
        chat.chat_loop()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
