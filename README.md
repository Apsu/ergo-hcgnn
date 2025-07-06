# Hierarchical Conversation GNN V2

A clean, modular implementation of the Hierarchical Conversation GNN with architectural improvements for better context retrieval in conversations.

## Key Improvements

1. **Learnable Token-to-Message Attention**: Replaces fixed pooling with multi-head attention mechanism that learns which tokens are important for message-level understanding.

2. **Cross-Message Token Attention**: Enables direct token-level connections across messages for better pronoun resolution and reference tracking.

3. **Curriculum Learning**: Progressive training from simple to complex conversations with hard example mining.

4. **Multi-Objective Training**: Combines relevance, contrastive, ranking, and margin losses with adaptive weighting.

5. **Enhanced Architecture**:
   - Relative position encoding for better distance awareness
   - Learnable temperature parameters per dependency type
   - Improved GAT layers with residual connections and normalization
   - Semantic edges in message graph (optional)

## Directory Structure

```
hierarchical_gnn_v2/
├── models/
│   └── hierarchical_gnn.py      # Main model implementation
├── layers/
│   ├── token_attention.py       # Token-to-message attention layers
│   └── graph_layers.py          # Improved GAT and graph builders
├── training/
│   ├── trainer.py               # Complete training pipeline
│   ├── losses.py                # Loss functions
│   └── curriculum.py            # Curriculum learning
├── data/
│   └── dataset.py               # Dataset and data loading
├── utils/
│   ├── evaluation.py            # Evaluation metrics
│   └── visualization.py         # Plotting utilities
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
├── generate.py                  # Generate synthetic conversations
└── clean_data.py                # Clean and validate conversation data
```

## Data Preparation

### Generate Data
```bash
python hierarchical_gnn_v2/generate.py --num-conversations 5000
```

### Clean Data
The generated data may contain invalid dependencies (self-references, future references). Clean the data before training:

```bash
python hierarchical_gnn_v2/clean_data.py \
    --input-dir datasets/raw \
    --output-dir datasets/processed
```

This removes invalid dependencies and saves cleaned conversations to `datasets/processed/`.

## Training

### Performance Optimization

For faster training, pre-tokenize your data:

```bash
python hierarchical_gnn_v2/utils/pretokenize.py \
    --input datasets/processed/conversations.json \
    --output datasets/processed/conversations_tokenized.pt
```

### Basic Training

```bash
# With pre-tokenized data (recommended for speed)
python hierarchical_gnn_v2/train.py \
    --data-paths datasets/processed/conversations_tokenized.pt \
    --pre-tokenized \
    --output-dir checkpoints/v2 \
    --batch-size 8 \
    --num-epochs 30 \
    --learning-rate 1e-4

# With raw data (slower, tokenizes on the fly)
python hierarchical_gnn_v2/train.py \
    --data-paths datasets/processed/conversations.json \
    --output-dir checkpoints/v2 \
    --batch-size 8 \
    --num-epochs 30 \
    --learning-rate 1e-4
```

### Advanced Training with All Features

```bash
# With pre-tokenized data (recommended)
python hierarchical_gnn_v2/train.py \
    --data-paths datasets/processed/conversations_tokenized.pt \
    --pre-tokenized \
    --output-dir checkpoints/v2_advanced \
    --batch-size 8 \
    --accumulation-steps 2 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --use-curriculum \
    --use-adaptive-loss \
    --use-cross-message-attention \
    --use-semantic-edges \
    --contrastive-weight 0.1 \
    --ranking-weight 0.3 \
    --margin-weight 0.1 \
    --window-sizes 1 2 3 \
    --num-token-gat-layers 2 \
    --num-message-gat-layers 2
```

### Resume Training

```bash
python hierarchical_gnn_v2/train.py \
    --data-paths datasets/processed/conversations_tokenized.pt \
    --pre-tokenized \
    --output-dir checkpoints/v2 \
    --resume-from checkpoints/v2/checkpoint_epoch_10.pt \
    --num-epochs 30
```

## Evaluation

```bash
python hierarchical_gnn_v2/evaluate.py \
    --checkpoint checkpoints/v2/best_model.pt \
    --test-data checkpoints/v2/test_conversations.json \
    --output-dir evaluation_results/v2
```

## Model Architecture

### Token Level
- Learnable token embeddings (128-dim) + position embeddings
- Token graphs with sequential and skip connections
- 2-layer GAT processing with residual connections
- **NEW**: Cross-message token attention for reference resolution
- **NEW**: Multi-head attention aggregation to message embeddings

### Message Level
- Message embeddings from token attention (128-dim)
- Temporal + optional semantic edges
- 2-layer GAT with improved architecture
- Relative position-aware relevance scoring

### Training Improvements
- Curriculum learning with progressive difficulty
- Hard negative mining after warm-up
- Adaptive loss weighting that learns importance
- Gradient accumulation for larger effective batch sizes

## Performance Optimizations

1. **Batched Operations**: All token graphs processed in parallel
2. **Efficient Attention**: Sparse cross-message connections
3. **Cached Embeddings**: Reuse computed embeddings when possible
4. **Mixed Precision**: Support for automatic mixed precision training
5. **Multi-worker Loading**: Parallel data loading with persistent workers

## Key Hyperparameters

- `token_embedding_dim`: 128 (learned from scratch)
- `hidden_dim`: 256 (GAT hidden dimension)
- `message_dim`: 128 (final message embeddings)
- `num_heads`: 4 (attention heads)
- `window_sizes`: [1, 2, 3] (token graph connections)
- `learning_rate`: 1e-4 with AdamW
- `dropout`: 0.1 throughout

## Metrics

The model is evaluated on:
- **Precision/Recall@k**: How well it retrieves relevant messages
- **MRR**: Mean Reciprocal Rank of first relevant message
- **MAP**: Mean Average Precision across queries
- **NDCG**: Normalized Discounted Cumulative Gain
- **F1@k**: Harmonic mean of precision and recall

Results are broken down by dependency type for detailed analysis.

## Visualizations

The evaluation script generates:
- Attention heatmaps showing message dependencies
- Token importance scores within messages
- Retrieval metric plots
- Training history curves
- Error analysis reports