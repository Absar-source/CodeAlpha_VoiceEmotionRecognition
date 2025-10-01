from data_preprocessing import Data_Prepare
from model import Model
import config
import time

def log_time(message):
    """Helper function to log messages with timestamps."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Data Preprocessing
log_time("Starting data preprocessing...")
data_prep = Data_Prepare()
data, label = data_prep.load(config.data_dir)
log_time("Data loaded successfully.")

log_time("Standardizing data...")
data = data_prep.standarize(data, max_len=300)
log_time("Data standardized.")

log_time("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = data_prep.split(data, label)
log_time("Data split completed.")

# Train CNN Model
log_time("Initializing the model...")
model = Model()
model.create_model()
log_time("Model initialized.")

log_time("Starting model training...")
start_time = time.time()
history = model.train(X_train, y_train, X_test, y_test)
end_time = time.time()
log_time(f"Model training completed in {end_time - start_time:.2f} seconds.")

log_time("Saving the trained model...")
model.save(config.MODEL_DIR)
log_time("Model saved successfully.")