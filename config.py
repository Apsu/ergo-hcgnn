"""
Central configuration for the GNN conversation system
"""
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories  
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_CHECKPOINT_DIR = CHECKPOINT_DIR / "models"
TRAINING_CHECKPOINT_DIR = CHECKPOINT_DIR / "training"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# Default file names
DEFAULT_CONVERSATION_FILE = "conversations.json"
DEFAULT_METADATA_FILE = "conversations_metadata.json"
DEFAULT_MODEL_NAME = "conversation_gnn.pt"

# Default paths
DEFAULT_RAW_CONVERSATION_PATH = RAW_DATA_DIR / DEFAULT_CONVERSATION_FILE
DEFAULT_PROCESSED_CONVERSATION_PATH = PROCESSED_DATA_DIR / DEFAULT_CONVERSATION_FILE
DEFAULT_MODEL_PATH = MODEL_CHECKPOINT_DIR / DEFAULT_MODEL_NAME

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 50
DEFAULT_HIDDEN_DIM = 256
DEFAULT_OUTPUT_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.1

# Generation defaults
DEFAULT_GENERATION_COUNT = 100
DEFAULT_GENERATION_BATCH_SIZE = 50
DEFAULT_CONCURRENT_REQUESTS = 10

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR, 
                 TRAINING_CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
