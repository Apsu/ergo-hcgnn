# Hierarchical Conversation GNN Architecture v2

## Overview

The Hierarchical Conversation GNN v2 is an advanced two-level graph neural network that learns conversation context through sophisticated **token-level** and **message-level** processing. This enhanced architecture introduces learnable attention mechanisms that bridge the token and message hierarchies, enabling fine-grained understanding of conversational dependencies.

## Key Architectural Enhancements

### 1. **Learnable Token-to-Message Attention**
Instead of simple pooling, the model learns which tokens are most important for message representation. **This preserves critical signals** - if "it" is important, the attention mechanism will weight it highly.

### 2. **Cross-Message Token Attention**
Tokens can attend to tokens in other messages, capturing fine-grained cross-message dependencies. **This is the key innovation** - the pronoun "it" in message 10 can directly attend to "Redis instance" in message 1, preserving the coreference signal BEFORE any pooling occurs.

### 3. **Multi-Objective Training**
Combines relevance, contrastive, ranking, and margin losses for robust learning.

### 4. **Dynamic Role and Position Encoding**
Flexible handling of conversation metadata without hard-coded assumptions.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        T["Raw Text Messages"] --> TOK["BERT Tokenizer"]
        TOK --> TID["Token IDs"]
        ROLE["Message Roles"] --> META["Metadata"]
        POS["Message Positions"] --> META
    end

    subgraph "Token Level Processing"
        TID --> TE["Token Embeddings<br/>30K × 256"]
        TID --> PE["Position Embeddings<br/>512 × 256"]
        TE --> ADD["Add"]
        PE --> ADD
        
        ADD --> SEG["Segment Embeddings<br/>(by role)"]
        SEG --> TG["Token Graphs<br/>per message"]
        
        TG --> TGAT1["Token GAT Layer 1<br/>256 → 256<br/>8 heads"]
        TGAT1 --> RES1["Residual + LayerNorm"]
        RES1 --> TGAT2["Token GAT Layer 2<br/>256 → 256<br/>8 heads"]
        TGAT2 --> RES2["Residual + LayerNorm"]
    end

    subgraph "Token-to-Message Attention"
        RES2 --> TMA["Learnable Token→Message<br/>Attention Module"]
        TMA --> WPOOL["Weighted Pooling"]
        WPOOL --> MPROJ["Projection<br/>256 → 128"]
    end

    subgraph "Cross-Message Attention"
        RES2 --> CMA["Cross-Message<br/>Token Attention"]
        CMA --> ENHANCE["Enhanced Tokens"]
    end

    subgraph "Message Level Processing"
        MPROJ --> ME["Message Embeddings<br/>128-dim"]
        META --> MENC["Message Encoder<br/>with role/position"]
        ME --> MENC
        
        MENC --> MGAT1["Message GAT Layer 1<br/>128 → 256<br/>4 heads"]
        MGAT1 --> MRES1["Residual + LayerNorm"]
        MRES1 --> MGAT2["Message GAT Layer 2<br/>256 → 128<br/>4 heads"]
        MGAT2 --> MRES2["Residual + LayerNorm"]
    end

    subgraph "Multi-Task Outputs"
        MRES2 --> REL["Relevance Scorer"]
        MRES2 --> CONTRAST["Contrastive Head"]
        MRES2 --> RANK["Ranking Head"]
        ENHANCE --> REF["Reference Resolution"]
    end
```

## Learnable Token-to-Message Attention

This mechanism learns which tokens contribute most to the message representation:

```mermaid
graph LR
    subgraph "Token-to-Message Attention Mechanism"
        TF["Token Features<br/>N × 256"] --> Q["Query Transform<br/>256 → 128"]
        TF --> K["Key Transform<br/>256 → 128"]
        TF --> V["Value Transform<br/>256 → 256"]
        
        Q --> ATT["Attention Scores<br/>softmax(QK^T/√d)"]
        K --> ATT
        
        ATT --> WEIGHT["Attention Weights<br/>N × 1"]
        V --> APPLY["Weighted Sum"]
        WEIGHT --> APPLY
        
        APPLY --> MSG["Message Representation<br/>1 × 256"]
        
        subgraph "Learnable Parameters"
            WQ["W_query"]
            WK["W_key"]
            WV["W_value"]
        end
    end
```

### Mathematical Formulation

```
For tokens t₁, t₂, ..., tₙ in a message:

q = LayerNorm(mean(t₁, ..., tₙ)) @ W_query  # Query is aggregate
K = [t₁, ..., tₙ] @ W_key                   # But keys are per-token
V = [t₁, ..., tₙ] @ W_value                 # Values are per-token

α = softmax(q @ K^T / √d_k)  # Learn which tokens matter
message_embedding = α @ V     # Weighted sum, NOT uniform pooling

Example weights for "How do I configure it?":
α = [0.05, 0.10, 0.05, 0.15, 0.60, 0.05]
                               ↑
                      "it" gets 60% weight!
```

## Addressing the "Information Bottleneck" Problem

Traditional architectures suffer from information loss when compressing token representations into message embeddings. Our architecture solves this in two ways:

1. **Learnable Attention (Not Pooling)**: The token-to-message attention learns to preserve important tokens. Unlike mean/max pooling which treats all tokens equally, our attention mechanism can give high weight to critical tokens like pronouns, entities, or technical terms.

2. **Cross-Message Token Attention**: Before any compression happens, tokens can directly attend to tokens in other messages. This means:
   - The pronoun "it" can find "Redis instance" across messages
   - Technical terms can connect to their definitions
   - Coreference chains are preserved at the token level

### Example: Pronoun Resolution
```
Message 1: "You should use Redis for caching"
Message 10: "How do I configure it?"

Traditional pooling: "it" gets averaged away
Our approach: 
- Token attention identifies "it" as important (high attention weight)
- Cross-message attention connects "it" → "Redis"
- Signal preserved through to message embeddings
```

## Cross-Message Token Attention

Allows tokens to attend to relevant tokens in other messages:

```mermaid
graph TB
    subgraph "Cross-Message Token Attention"
        subgraph "Message i"
            Ti["Tokens"] --> Qi["Queries"]
        end
        
        subgraph "All Messages"
            T1["Message 1 Tokens"] --> K1["Keys"]
            T2["Message 2 Tokens"] --> K2["Keys"]
            T3["Message 3 Tokens"] --> K3["Keys"]
            Tn["Message n Tokens"] --> Kn["Keys"]
            
            K1 --> KALL["All Keys"]
            K2 --> KALL
            K3 --> KALL
            Kn --> KALL
        end
        
        Qi --> MATT["Multi-Head<br/>Attention"]
        KALL --> MATT
        KALL --> V["Values"]
        V --> MATT
        
        MATT --> MASK["Dependency<br/>Masking"]
        DEPS["Known Dependencies"] --> MASK
        
        MASK --> ENH["Enhanced<br/>Token Features"]
    end
```

### Dependency-Aware Masking

```mermaid
graph LR
    subgraph "Attention Masking Strategy"
        DEP["Dependency Labels"] --> SOFT["Soft Masking"]
        SOFT --> SCORE["α = 1.0 for dependencies<br/>α = 0.1 for others"]
        
        SCORE --> NORM["Normalize"]
        NORM --> FINAL["Final Attention"]
    end
```

## Enhanced Token Graph Structure

```mermaid
graph TD
    subgraph "Rich Token Graph"
        T0["[CLS]"] -.->|sequential| T1["How"]
        T1 -.->|sequential| T2["to"]
        T2 -.->|sequential| T3["implement"]
        T3 -.->|sequential| T4["caching"]
        T4 -.->|sequential| T5["?"]
        T5 -.->|sequential| T6["[SEP]"]
        
        T0 ==>|skip-2| T2
        T1 ==>|skip-2| T3
        T2 ==>|skip-2| T4
        T3 ==>|skip-2| T5
        T4 ==>|skip-2| T6
        
        T0 -->|global| T3
        T0 -->|global| T4
        T6 -->|global| T3
        T6 -->|global| T4
        
        style T3 fill:#f9f,stroke:#333,stroke-width:4px
        style T4 fill:#f9f,stroke:#333,stroke-width:4px
    end
```

**Edge Types:**
- **Sequential**: Adjacent tokens (bidirectional)
- **Skip-2**: Bigram patterns
- **Global**: [CLS] and [SEP] to content words
- **Self-loops**: Information retention (all nodes)

## Message Graph with Semantic Edges

```mermaid
graph TB
    subgraph "Enhanced Message Graph"
        M0["User: How to cache?"] -.->|temporal| M1["Assistant: Use Redis..."]
        M1 -.->|temporal| M2["User: What about TTL?"]
        M2 -.->|temporal| M3["Assistant: Set expiry..."]
        M3 -.->|temporal| M4["User: Thanks!"]
        
        M0 ==>|topic_ref| M2
        M1 ==>|continuation| M3
        M2 -.->|clarification| M1
        
        style M0 fill:#bbf,stroke:#333,stroke-width:2px
        style M2 fill:#bbf,stroke:#333,stroke-width:2px
    end
```

**Edge Types:**
- **Temporal**: Sequential conversation flow
- **Topic Reference**: Topical connections
- **Continuation**: Direct follow-ups
- **Clarification**: Questions about previous content

## Multi-Objective Training

```mermaid
graph TD
    subgraph "Loss Components"
        EMB["Message Embeddings"] --> L1["Relevance Loss<br/>KL Divergence"]
        EMB --> L2["Contrastive Loss<br/>InfoNCE"]
        EMB --> L3["Ranking Loss<br/>Pairwise Ranking"]
        EMB --> L4["Margin Loss<br/>Triplet Margin"]
        
        L1 --> W1["Weight: adaptive"]
        L2 --> W2["Weight: 0.1"]
        L3 --> W3["Weight: 0.1"] 
        L4 --> W4["Weight: 0.05"]
        
        W1 --> TOTAL["Total Loss"]
        W2 --> TOTAL
        W3 --> TOTAL
        W4 --> TOTAL
    end
```

### Adaptive Loss Weighting

```python
# Dynamically weight relevance loss based on training progress
relevance_weight = 1.0 + 0.5 * (1 - epoch / max_epochs)
```

## Complete Information Flow

```mermaid
sequenceDiagram
    participant User as User Input
    participant Tok as Tokenizer
    participant TEmb as Token Embeddings
    participant TGAT as Token GAT
    participant CMA as Cross-Message Attention
    participant TMA as Token→Message Attention
    participant MGAT as Message GAT
    participant Score as Scoring Head
    
    User->>Tok: "How do I configure it?"
    Tok->>TEmb: [101, 2129, 2079, 1045, 8736, 2009, 102]
    TEmb->>TGAT: Token features + position
    TGAT->>CMA: Enhanced tokens (including "it")
    
    Note over CMA: "it" attends to "Redis" in Message 1
    CMA->>TMA: Tokens with cross-message context
    
    Note over TMA: Attention weights "it" highly
    TMA->>MGAT: Weighted message embedding
    
    Note over MGAT: Message-level reasoning
    MGAT->>Score: Final message representation
    Score-->>User: High relevance for Message 1
```

## Key Architectural Components

### 1. Token Embedding Layer
```mermaid
graph LR
    subgraph "Token Embedding Details"
        TID["Token ID"] --> EMBED["Embedding<br/>Matrix<br/>30522 × 256"]
        PID["Position ID"] --> PEMBED["Position<br/>Matrix<br/>512 × 256"]
        RID["Role ID"] --> SEGMENT["Segment<br/>Matrix<br/>3 × 256"]
        
        EMBED --> ADD["Element-wise<br/>Addition"]
        PEMBED --> ADD
        SEGMENT --> ADD
        
        ADD --> NORM["LayerNorm"]
        NORM --> DROP["Dropout<br/>p=0.1"]
    end
```

### 2. GAT Layer Architecture
```mermaid
graph TD
    subgraph "Multi-Head GAT Block"
        IN["Input<br/>N × F_in"] --> HEADS["8 Attention Heads"]
        
        HEADS --> H1["Head 1<br/>α¹ᵢⱼ"]
        HEADS --> H2["Head 2<br/>α²ᵢⱼ"]
        HEADS --> H8["Head 8<br/>α⁸ᵢⱼ"]
        
        H1 --> CONCAT["Concatenate"]
        H2 --> CONCAT
        H8 --> CONCAT
        
        CONCAT --> PROJ["Project<br/>8*F_head → F_out"]
        IN --> RES["Residual"]
        PROJ --> ADD["Add"]
        RES --> ADD
        ADD --> LN["LayerNorm"]
        LN --> OUT["Output<br/>N × F_out"]
    end
```

### 3. Relevance Scoring Network
```mermaid
graph LR
    subgraph "Scoring Architecture"
        Q["Query Message<br/>128-dim"] --> DOT["Dot Product"]
        C["Context Message<br/>128-dim"] --> DOT
        
        Q --> DIFF["Difference<br/>|Q - C|"]
        C --> DIFF
        
        Q --> HAD["Hadamard<br/>Q ⊙ C"]
        C --> HAD
        
        DOT --> FEAT["Concatenate<br/>Features"]
        DIFF --> FEAT
        HAD --> FEAT
        
        FEAT --> MLP["MLP<br/>387 → 128 → 64 → 1"]
        MLP --> SIGMOID["Sigmoid"]
        SIGMOID --> SCORE["Relevance"]
    end
```

## Model Statistics

```mermaid
graph LR
    subgraph "Parameter Distribution"
        TE["Token Embeddings<br/>30,522 × 256 = 7.8M"]
        PE["Position Embeddings<br/>512 × 256 = 131K"]
        SE["Segment Embeddings<br/>3 × 256 = 768"]
        
        TGAT["Token GAT (2 layers)<br/>8 heads each<br/>~2.1M"]
        TMA["Token→Message Attention<br/>3 × (256 × 128)<br/>~98K"]
        CMA["Cross-Message Attention<br/>8 heads × 3 matrices<br/>~196K"]
        
        MGAT["Message GAT (2 layers)<br/>4 heads each<br/>~525K"]
        SCORE["Scoring Networks<br/>4 task heads<br/>~165K"]
        
        TOTAL["Total Parameters<br/>~11M"]
    end
```

## Training Innovations

### 1. Curriculum Learning
```mermaid
graph TD
    subgraph "Progressive Difficulty"
        E1["Epochs 1-10<br/>Simple Dependencies<br/>Recent context only"]
        E2["Epochs 11-25<br/>Medium Complexity<br/>Topic references"]
        E3["Epochs 26-50<br/>Full Complexity<br/>Long-range deps"]
        
        E1 --> E2
        E2 --> E3
    end
```

### 2. Temperature Scheduling
```mermaid
graph LR
    subgraph "Temperature Control"
        TRAIN["Training<br/>T = 1.0"] --> FINE["Fine-tuning<br/>T = 0.5-2.0"]
        FINE --> INFER["Inference<br/>T = user-defined"]
    end
```

### 3. Negative Sampling Strategy
```mermaid
graph TD
    subgraph "Smart Negative Sampling"
        POS["Positive Pairs<br/>(true dependencies)"]
        
        HARD["Hard Negatives<br/>Same topic, wrong message<br/>40%"]
        MED["Medium Negatives<br/>Related conversation<br/>40%"]
        EASY["Easy Negatives<br/>Random messages<br/>20%"]
        
        POS --> LOSS["Contrastive Loss"]
        HARD --> LOSS
        MED --> LOSS
        EASY --> LOSS
    end
```

## Performance Optimizations

### 1. Efficient Attention Computation
- Vectorized cross-message attention
- Sparse attention patterns for long conversations
- Gradient checkpointing for memory efficiency

### 2. Dynamic Batching
- Groups conversations by length
- Minimizes padding overhead
- Enables larger effective batch sizes

### 3. Mixed Precision Training
- FP16 for forward pass
- FP32 for loss computation
- 2x speedup with minimal accuracy loss

## Advantages Over Previous Version

1. **Better Token Understanding**: Learnable attention identifies important tokens instead of uniform pooling

2. **Fine-grained Dependencies**: Cross-message token attention captures pronoun references and entity mentions

3. **Robust Training**: Multi-objective learning prevents overfitting to single metric

4. **Flexible Architecture**: Easily extended with new edge types or attention patterns

5. **Production Ready**: Optimized for both training efficiency and inference speed

## Future Extensions

1. **Syntax-Aware Edges**: Incorporate dependency parsing for richer token graphs
2. **Entity Tracking**: Special handling for named entities across messages  
3. **Multilingual Support**: Language-agnostic token processing
4. **Adaptive Architecture**: Dynamic depth based on conversation complexity
5. **Explainable Attention**: Visualize why certain contexts were selected