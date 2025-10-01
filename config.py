import os

# Base directory for saving models and related files
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Data Directory 
data_dir = os.path.join(BASE_DIR,"data")

# Directory to save trained models
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models/model.h5')

# Directory to save logs
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Directory to save checkpoints
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')

# Directory to save results
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure all directories exist
for directory in [MODEL_DIR, LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)