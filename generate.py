#!/usr/bin/env python3
"""
Enhanced conversation generator for GNN training
Focuses on creating conversations that mirror real-world long-running dialogues
with complex reference patterns, topic shifts, and context dependencies
"""

import asyncio
import json
import argparse
from typing import List, Dict, Optional, Tuple, Literal
import aiohttp
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import random
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import signal
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Import configuration
try:
    from config import (
        RAW_DATA_DIR, DEFAULT_RAW_CONVERSATION_PATH,
        DEFAULT_GENERATION_COUNT, DEFAULT_GENERATION_BATCH_SIZE,
        DEFAULT_CONCURRENT_REQUESTS
    )
except ImportError:
    # Fallback if config.py is not available
    RAW_DATA_DIR = Path("datasets/raw")
    DEFAULT_RAW_CONVERSATION_PATH = RAW_DATA_DIR / "conversations.json"
    DEFAULT_GENERATION_COUNT = 100
    DEFAULT_GENERATION_BATCH_SIZE = 50
    DEFAULT_CONCURRENT_REQUESTS = 10

# Pydantic models for structured output
class MessageWithMetadata(BaseModel):
    role: str = Field(description="Either 'user' or 'assistant'")
    text: str = Field(description="The message content")
    is_context_dependent: bool = Field(
        default=False,
        description="True if this message requires previous context to be understood"
    )
    depends_on_indices: List[int] = Field(
        default_factory=list,
        description="List of message indices this message depends on (0-based)"
    )
    dependency_type: Optional[Literal[
        "pronoun_reference",
        "continuation", 
        "clarification",
        "topic_reference",
        "agreement",
        "disagreement",
        "follow_up",
        "correction",
        "example_request"
    ]] = Field(
        default=None,
        description="Type of dependency if message is context-dependent"
    )

class ConversationWithMetadata(BaseModel):
    messages: List[MessageWithMetadata] = Field(
        description="List of messages with context dependency metadata"
    )
    conversation_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns used in this conversation: 'topic_shift', 'long_range_reference', 'clarification_loop', 'pronoun_chaining', 'agreement_sequence', etc."
    )

@dataclass
class ConversationConfig:
    topics: List[str]  # Multiple topics for topic drift
    complexity: str
    turns: int
    reference_style: str
    should_include_vague: bool
    conversation_type: str  # New: type of conversation pattern
    has_topic_shift: bool  # New: whether topics shift during conversation
    has_long_range_ref: bool  # New: references across many messages
    has_clarification_loops: bool  # New: user asks for clarification multiple times

class ConversationGenerator:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        
        # Expanded topics for richer conversations
        self.topic_groups = {
            "system_design": [
                "microservices architecture", "event-driven systems", "caching strategies",
                "database sharding", "load balancing", "message queues", "API gateways"
            ],
            "programming": [
                "async programming", "memory management", "design patterns",
                "functional programming", "concurrency", "error handling", "testing"
            ],
            "ml_ai": [
                "neural networks", "transformer models", "training optimization",
                "model deployment", "feature engineering", "evaluation metrics", "bias detection"
            ],
            "devops": [
                "CI/CD pipelines", "container orchestration", "monitoring systems",
                "infrastructure as code", "security scanning", "deployment strategies"
            ],
            "frontend": [
                "state management", "component architecture", "performance optimization",
                "responsive design", "accessibility", "build tools", "SSR vs CSR"
            ]
        }
        
        # Conversation types that create different reference patterns
        self.conversation_types = [
            "deep_dive",  # Start broad, go deep into specifics
            "exploration",  # Jump between related topics
            "problem_solving",  # Iterative debugging/solving
            "implementation",  # Plan then implement step by step
            "comparison",  # Compare multiple approaches
            "learning_journey",  # Build knowledge progressively
            "troubleshooting"  # Debug with many back-references
        ]
        
        # Vague responses that require context to understand
        self.contextual_responses = [
            "Ok let's do that",
            "Yes, show me how",
            "Tell me more about that approach",
            "How does that compare to what we discussed earlier?",
            "That makes sense, but what about the other thing?",
            "I see, let's go with the first option",
            "Can we apply this to our original problem?",
            "Wait, how does this relate to what you mentioned before?",
            "Let's try it but with the modification you suggested",
            "Actually, let's go back to your previous suggestion"
        ]
    
    def get_system_prompt(self, config: ConversationConfig) -> str:
        """Create system prompt based on configuration"""
        topics_str = " and ".join(config.topics[:2]) if len(config.topics) > 1 else config.topics[0]
        
        base_prompt = f"""You are simulating a realistic technical conversation between a user and an AI assistant.

The conversation should feel like a real extended dialogue about {topics_str}.

CORE REQUIREMENTS:
1. Generate exactly {config.turns} exchanges (user and assistant messages)
2. Make it feel like a continuous, coherent discussion
3. The conversation type is: {config.conversation_type}
4. FOR EACH MESSAGE, accurately label:
   - is_context_dependent: whether the message needs previous context to be understood
   - depends_on_indices: which previous message(s) it refers to (use 0-based indices)
   - dependency_type: what kind of dependency it is

DEPENDENCY TYPES (use EXACTLY these values):
- pronoun_reference: Uses pronouns ("it", "that", "this") referring to previous concepts
- continuation: Continues or expands on previous topic ("Tell me more", "Go on")
- clarification: Asks for or provides clarification of previous message
- topic_reference: Explicitly refers back to earlier topic ("the approach you mentioned")
- agreement: Agrees with or accepts previous suggestion ("Ok let's do that", "Yes")
- disagreement: Disagrees or suggests alternative to previous message
- follow_up: Follows up on previous action or suggestion
- correction: Corrects or modifies previous statement
- example_request: Asks for or provides example of previous concept

Be very precise in your labeling - identify ALL context dependencies accurately."""

        # Add reference style instructions
        if config.reference_style == "heavy_pronouns":
            base_prompt += """
4. The user frequently uses pronouns like "that", "it", "this" to refer to previous topics
5. Include vague affirmatives like "Ok let's do that", "Tell me more about that"
6. Have the user refer back to earlier messages with phrases like "the approach you mentioned earlier"
7. Create situations where context from 3-5 messages ago is crucial to understand the current message"""
        elif config.reference_style == "mixed":
            base_prompt += """
4. Mix specific and vague references naturally
5. Sometimes use "that approach" or "this method" instead of repeating the full name
6. Include some contextual responses that need previous messages to understand"""
        
        # Add pattern-specific instructions
        pattern_prompts = {
            "deep_dive": """
This is a deep-dive conversation:
- Start with a high-level question
- Each response should go deeper into technical details
- User asks follow-up questions that drill down into specifics
- References should often point to concepts introduced 2-3 messages earlier""",
            
            "exploration": """
This is an exploration conversation:
- User is exploring related concepts
- Topics naturally drift from one to another
- Include moments where user connects current topic to something discussed earlier
- Use phrases like "how does this relate to..." or "is this similar to...""",
            
            "problem_solving": """
This is a problem-solving conversation:
- User has a specific problem to solve
- Solution is built iteratively
- User reports results and asks for next steps
- Many references to "the error", "that solution", "the approach we tried""",
            
            "implementation": """
This is an implementation conversation:
- Start with planning/design discussion
- Move to implementation details
- User asks about specific parts referencing the plan
- Lots of "for the X we discussed" type references""",
            
            "troubleshooting": """
This is a troubleshooting conversation:
- User describes a problem
- Assistant suggests debugging steps
- User tries things and reports back
- Heavy use of "that didn't work", "the error is still there", "it's different now"""
        }
        
        if config.conversation_type in pattern_prompts:
            base_prompt += "\n" + pattern_prompts[config.conversation_type]
        
        # Add special features
        if config.has_topic_shift:
            base_prompt += "\n\nIMPORTANT: The conversation should naturally shift between these topics: " + ", ".join(config.topics)
            base_prompt += "\nMake the transitions feel organic, with the user connecting concepts between topics."
        
        if config.has_long_range_ref:
            base_prompt += "\n\nIMPORTANT: Include at least 2 instances where the user refers back to something from much earlier in the conversation (5+ messages ago)."
        
        if config.has_clarification_loops:
            base_prompt += "\n\nIMPORTANT: Include 1-2 clarification loops where the user doesn't fully understand and asks for clarification using vague references."
        
        return base_prompt
    
    def get_user_prompt(self, config: ConversationConfig) -> str:
        """Create specific instructions for this conversation"""
        examples = {
            "deep_dive": "User starts by asking about system design, then dives into specific implementation details, database choices, and performance considerations.",
            "exploration": "User is learning about related technologies, jumping from one to another as they see connections.",
            "problem_solving": "User has a specific bug or performance issue and works with assistant to solve it step by step.",
            "implementation": "User wants to build something specific, starts with architecture discussion, then implements each part.",
            "troubleshooting": "User has a system that's not working correctly and needs help debugging it."
        }
        
        prompt = f"Generate a {config.conversation_type} conversation. "
        prompt += examples.get(config.conversation_type, "")
        
        if config.should_include_vague:
            prompt += f"\n\nInclude these types of contextual responses from the user:\n"
            sample_responses = random.sample(self.contextual_responses, min(3, len(self.contextual_responses)))
            for resp in sample_responses:
                prompt += f"- '{resp}'\n"
        
        return prompt
    
    async def generate_conversation(self, config: ConversationConfig) -> Dict[str, any]:
        """Generate a single conversation with metadata"""
        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_system_prompt(config)},
                    {"role": "user", "content": self.get_user_prompt(config)}
                ],
                response_format=ConversationWithMetadata,
                temperature=0.85,  # Slightly higher for more natural variation
                max_tokens=8192  # Allow longer conversations
            )
            
            conversation = response.choices[0].message.parsed
            # Return the full conversation with metadata
            return {
                "messages": [msg.model_dump() for msg in conversation.messages],
                "conversation_patterns": conversation.conversation_patterns,
                "config": asdict(config)  # Include config for analysis
            }
                
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None
    
    def create_diverse_configs(self, count: int) -> List[ConversationConfig]:
        """Create diverse conversation configurations for robust training"""
        configs = []
        
        for i in range(count):
            # Select conversation type
            conv_type = random.choice(self.conversation_types)
            
            # Select 1-3 related topics (more topics = more complex reference patterns)
            topic_group = random.choice(list(self.topic_groups.keys()))
            num_topics = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
            topics = random.sample(self.topic_groups[topic_group], num_topics)
            
            # Complexity based on conversation type
            if conv_type in ["deep_dive", "implementation", "troubleshooting"]:
                complexity = random.choice(["medium", "complex"])
                turns = random.randint(10, 20)  # Longer conversations
            else:
                complexity = random.choice(["simple", "medium", "complex"])
                turns = random.randint(8, 15)
            
            # Reference style - heavily biased toward pronoun use for training
            rand = random.random()
            if rand < 0.7:
                reference_style = "heavy_pronouns"
            elif rand < 0.95:
                reference_style = "mixed"
            else:
                reference_style = "minimal"
            
            # Features that create interesting reference patterns
            should_include_vague = random.random() < 0.8  # 80% include vague responses
            has_topic_shift = len(topics) > 1 and random.random() < 0.6  # 60% of multi-topic convs shift
            has_long_range_ref = turns > 12 and random.random() < 0.5  # 50% of long convs have long-range refs
            has_clarification_loops = random.random() < 0.3  # 30% have clarification loops
            
            configs.append(ConversationConfig(
                topics=topics,
                complexity=complexity,
                turns=turns,
                reference_style=reference_style,
                should_include_vague=should_include_vague,
                conversation_type=conv_type,
                has_topic_shift=has_topic_shift,
                has_long_range_ref=has_long_range_ref,
                has_clarification_loops=has_clarification_loops
            ))
        
        return configs
    
    async def generate_batch(self, configs: List[ConversationConfig], 
                           semaphore: asyncio.Semaphore) -> List[Dict[str, any]]:
        """Generate multiple conversations concurrently with rate limiting"""
        async def generate_with_semaphore(config):
            async with semaphore:
                return await self.generate_conversation(config)
        
        tasks = [generate_with_semaphore(config) for config in configs]
        results = []
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating conversations"):
            result = await coro
            if result:
                results.append(result)
        
        return results

class DataGenerationManager:
    """Manages data generation with robust interruption handling"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.temp_path = self.output_path.with_suffix('.tmp')
        self.checkpoint_path = self.output_path.with_suffix('.checkpoint')
        self.conversations = []
        self.completed_count = 0
        self.shutdown_requested = False
        self.current_batch_future = None
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            print(f"\n\n⚠️  Received signal {signum}. Initiating graceful shutdown...")
            self.shutdown_requested = True
            if self.current_batch_future and not self.current_batch_future.done():
                print("   Waiting for current batch to complete...")
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def load_checkpoint(self, allow_resume: bool = True) -> Tuple[List[Dict], int]:
        """Load existing conversations from checkpoint or output file"""
        if not allow_resume:
            return [], 0
            
        # First try checkpoint file
        if self.checkpoint_path.exists():
            print(f"📥 Loading checkpoint from {self.checkpoint_path}")
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
                return data['conversations'], data['completed_count']
        
        # Then try existing output file
        if self.output_path.exists():
            print(f"📥 Loading existing conversations from {self.output_path}")
            with open(self.output_path, 'r') as f:
                conversations = json.load(f)
                return conversations, len(conversations)
        
        # Finally try temp file
        if self.temp_path.exists():
            print(f"📥 Loading partial results from {self.temp_path}")
            with open(self.temp_path, 'r') as f:
                conversations = json.load(f)
                return conversations, len(conversations)
        
        return [], 0
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            'conversations': self.conversations,
            'completed_count': self.completed_count,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Checkpoint saved: {self.completed_count} conversations")
    
    def save_final_output(self):
        """Save final output and clean up temporary files"""
        # Save conversations with full metadata
        with open(self.output_path, 'w') as f:
            json.dump(self.conversations, f, indent=2)
        
        # Clean up temporary files
        if self.temp_path.exists():
            self.temp_path.unlink()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        
        print(f"\n✅ Successfully saved {len(self.conversations)} conversations to {self.output_path}")
    
    async def generate_with_recovery(self, generator, configs, semaphore, batch_size, allow_resume=True):
        """Generate conversations with interruption recovery"""
        self.setup_signal_handlers()
        
        # Load any existing progress
        self.conversations, self.completed_count = self.load_checkpoint(allow_resume)
        
        if self.completed_count > 0:
            print(f"ℹ️  Resuming from {self.completed_count} completed conversations")
            # Skip already completed configs
            configs = configs[self.completed_count:]
        
        # Generate in batches
        for i in range(0, len(configs), batch_size):
            if self.shutdown_requested:
                print("\n🛑 Shutdown requested, stopping generation...")
                break
            
            batch_configs = configs[i:i + batch_size]
            batch_num = (self.completed_count + i) // batch_size + 1
            total_batches = (self.completed_count + len(configs) + batch_size - 1) // batch_size
            
            print(f"\n🔄 Generating batch {batch_num}/{total_batches}")
            
            try:
                # Store the task so we can wait for it on shutdown
                self.current_batch_future = asyncio.create_task(
                    generator.generate_batch(batch_configs, semaphore)
                )
                batch_conversations = await self.current_batch_future
                
                # Add successful conversations
                self.conversations.extend(batch_conversations)
                self.completed_count += len(batch_conversations)
                
                # Save checkpoint after each batch
                self.save_checkpoint()
                
                print(f"   Batch complete: {len(batch_conversations)} conversations")
                print(f"   Total progress: {self.completed_count} conversations")
                
            except Exception as e:
                print(f"\n❌ Error in batch {batch_num}: {e}")
                print("   Saving progress and continuing...")
                self.save_checkpoint()
                
            finally:
                self.current_batch_future = None
        
        # Save final output
        if self.conversations:
            self.save_final_output()
            return True
        else:
            print("\n⚠️  No conversations were generated")
            return False

async def main():
    parser = argparse.ArgumentParser(description='Generate enhanced conversations for GNN training')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--base-url', type=str, required=True, help='API base URL')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--count', type=int, default=DEFAULT_GENERATION_COUNT, help='Number of conversations to generate')
    parser.add_argument('--output-path', type=str, default=str(DEFAULT_RAW_CONVERSATION_PATH), help='Output file path')
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT_REQUESTS, help='Number of concurrent requests')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_GENERATION_BATCH_SIZE, help='Batch size for saving')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    
    args = parser.parse_args()
    
    generator = ConversationGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model
    )
    
    print(f"Generating {args.count} enhanced conversations using {args.model}")
    print(f"API endpoint: {args.base_url}")
    
    # Create configurations
    configs = generator.create_diverse_configs(args.count)
    
    # Use DataGenerationManager for robust generation
    manager = DataGenerationManager(args.output_path)
    semaphore = asyncio.Semaphore(args.concurrent)
    
    # Generate with interruption recovery
    success = await manager.generate_with_recovery(
        generator, configs, semaphore, args.batch_size, args.resume
    )
    
    if not success:
        print("\n❌ Generation failed or was interrupted")
        return
    
    # Use the manager's conversations for statistics
    all_conversations = manager.conversations
    
    # Enhanced metadata with dependency statistics
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "count": len(all_conversations),
        "api_endpoint": args.base_url,
        "statistics": {
            "avg_turns": sum(len(c["messages"]) for c in all_conversations) / len(all_conversations) if all_conversations else 0,
            "conversation_types": {},
            "topic_groups": {},
            "features": {
                "with_vague_responses": sum(1 for c in configs if c.should_include_vague),
                "with_topic_shift": sum(1 for c in configs if c.has_topic_shift),
                "with_long_range_ref": sum(1 for c in configs if c.has_long_range_ref),
                "with_clarification_loops": sum(1 for c in configs if c.has_clarification_loops)
            },
            "dependency_stats": {
                "total_messages": sum(len(c["messages"]) for c in all_conversations),
                "context_dependent_messages": 0,
                "dependency_types": {},
                "avg_dependencies_per_message": 0
            }
        }
    }
    
    # Calculate dependency statistics
    total_dependencies = 0
    for conv in all_conversations:
        for msg in conv["messages"]:
            if msg.get("is_context_dependent", False):
                metadata["statistics"]["dependency_stats"]["context_dependent_messages"] += 1
                dep_type = msg.get("dependency_type")
                if dep_type:
                    metadata["statistics"]["dependency_stats"]["dependency_types"][dep_type] = \
                        metadata["statistics"]["dependency_stats"]["dependency_types"].get(dep_type, 0) + 1
                total_dependencies += len(msg.get("depends_on_indices", []))
    
    if metadata["statistics"]["dependency_stats"]["total_messages"] > 0:
        metadata["statistics"]["dependency_stats"]["avg_dependencies_per_message"] = \
            total_dependencies / metadata["statistics"]["dependency_stats"]["total_messages"]
        metadata["statistics"]["dependency_stats"]["context_dependency_rate"] = \
            metadata["statistics"]["dependency_stats"]["context_dependent_messages"] / \
            metadata["statistics"]["dependency_stats"]["total_messages"]
    
    # Count conversation types
    for config in configs:
        conv_type = config.conversation_type
        metadata["statistics"]["conversation_types"][conv_type] = \
            metadata["statistics"]["conversation_types"].get(conv_type, 0) + 1
    
    metadata_path = args.output_path.replace('.json', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📊 Metadata saved to: {metadata_path}")
    
    # Print statistics
    print("\n📈 Generation Statistics:")
    print(f"   Average conversation length: {metadata['statistics']['avg_turns']:.1f} messages")
    print(f"   Conversations with topic shifts: {metadata['statistics']['features']['with_topic_shift']}")
    print(f"   Conversations with long-range references: {metadata['statistics']['features']['with_long_range_ref']}")
    
    # Print dependency statistics
    dep_stats = metadata['statistics']['dependency_stats']
    if dep_stats['total_messages'] > 0:
        print(f"\n📊 Dependency Statistics:")
        print(f"   Context-dependent messages: {dep_stats['context_dependent_messages']}/{dep_stats['total_messages']} ({dep_stats.get('context_dependency_rate', 0)*100:.1f}%)")
        print(f"   Average dependencies per message: {dep_stats['avg_dependencies_per_message']:.2f}")
        if dep_stats['dependency_types']:
            print(f"   Most common dependency types:")
            for dep_type, count in sorted(dep_stats['dependency_types'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      - {dep_type}: {count}")
    
    # Print sample with dependency info
    if all_conversations:
        print("\n📝 Sample conversation with dependencies:")
        sample = random.choice(all_conversations)
        for i, msg in enumerate(sample["messages"][:8]):
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            print(f"{role_emoji} [{i}] {msg['role']}: {msg['text'][:80]}...")
            if msg.get('is_context_dependent'):
                print(f"      ↳ Depends on: {msg.get('depends_on_indices', [])} ({msg.get('dependency_type', 'unknown')})")

if __name__ == "__main__":
    asyncio.run(main())
