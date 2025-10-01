import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import config

def plot_accuracy_loss(history):
    """
    Plot the training and validation accuracy and loss graphs.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, label_classes):
    """
    Plot the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

def visualize_model_performance(history_path, y_true, y_pred):
    """
    Visualize the model's performance including accuracy/loss graphs and confusion matrix.

    Args:
        history_path (str): Path to the saved training history file (e.g., history.npy).
        y_true (array): True labels for the test set.
        y_pred (array): Predicted labels for the test set.
    """
    # Load training history
    history = np.load(history_path, allow_pickle=True).item()

    # Load label classes
    label_classes = np.load(os.path.join(config.RESULTS_DIR, "label_classes.npy"))

    # Plot accuracy and loss
    plot_accuracy_loss(history)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, label_classes)



from keras.models import load_model
import numpy as np
from data_preprocessing import Data_Prepare
import config

# Initialize the Data_Prepare class
data_prep = Data_Prepare()

# Load the data and labels
data, label = data_prep.load(config.data_dir)

# Standardize the data
data = data_prep.standarize(data, max_len=300)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = data_prep.split(data, label)
# Load true and predicted labels
y_true = np.argmax(y_test, axis=1)  # True labels
model = load_model(os.path.join(config.RESULTS_DIR, "model.h5"))
try:
    if model:
        y_pred = np.argmax(model.predict(X_test), axis=1)  # Predicted labels

        # Visualize performance
        visualize_model_performance(
            history_path=os.path.join(config.RESULTS_DIR, "history.npy"),
            y_true=y_true,
            y_pred=y_pred
    )
except  Exception  as e: print("check this error: ",e)