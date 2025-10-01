# Emotion Detection from Voice

This project uses **Convolutional Neural Networks (CNNs)** to detect emotions from voice recordings. The dataset is preprocessed to extract **MFCC (Mel Frequency Cepstral Coefficients)** features, which are then used to train a deep learning model to classify emotions.

## Features
- **Emotion Detection**: Classifies emotions such as happy, sad, angry, calm, etc.
- **Visualization**: Displays MFCC curves and confusion matrices for better understanding.
- **Preprocessing**: Standardizes audio features for consistent input to the model.

---

## Results

### Model Performance
- **Test Accuracy**: ~92%
- **Confusion Matrix**:
  ![Confusion Matrix](results/confusion_graph.jpg)

- **Accuracy and Loss Graphs**:
  - **Accuracy**:
    ![Accuracy Graph](results/emotion_accuracy.jpg)
  - **Loss**:
    ![Loss Graph](results/emotion_loss.jpg)


---

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Clone the Repository
```bash
git clone https://github.com/your-username/emotion-detection-from-voice.git
cd emotion-detection-from-voice
```

###Set Up Virtual Environment
```bash
python -m venv .venv
[activate](http://_vscodecontentref_/1)  # On Windows
# source .venv/bin/activate  # On Linux/Mac
```
Install Dependencies
```bash
pip install -r requirements.txt
```

Usage
1. Preprocess the Dataset
Place your .wav files in the data/ directory. The dataset should follow the RAVDESS naming convention.

2. Train the Model
Run the main.py script to preprocess the data, train the model, and save the results:

```bash
python [main.py](http://_vscodecontentref_/3)
```


3. Visualize Results
Run the Visualize.py script to generate the confusion matrix and accuracy/loss graphs:

```bash
python [Visualize.py](http://_vscodecontentref_/4)
```

4. Test with New Audio
Use the evaluate.py script to test the model with a new .wav file:

```bash
python [evaluate.py](http://_vscodecontentref_/5)
```



##Project Structure
emotionDetectionFromVoice/
│
├── data/                     # Dataset folder
├── [main.py](http://_vscodecontentref_/6)                   # Main script for training
├── [evaluate.py](http://_vscodecontentref_/7)               # Script for testing the model with new audio
├── [Visualize.py](http://_vscodecontentref_/8)              # Script for visualizing confusion matrix and training graphs
├── [data_preprocessing.py](http://_vscodecontentref_/9)     # Data preprocessing module
├── [model.py](http://_vscodecontentref_/10)                  # Model creation and training module
├── result/                   # Directory for saved models and results
│   ├── model.h5              # Trained model
│   ├── history.npy           # Training history
│   ├── label_classes.npy     # Label encoder classes
├── requirements.txt          # Python dependencies
├── [README.md](http://_vscodecontentref_/11)                 # Project documentation
└── .gitignore                # Git ignore file



##Dependencies
- librosa
- numpy
- matplotlib
- keras
- scikit-learn
- tensorflow
- tqdm
Install all dependencies using:
```bash 
pip install -r requirements.txt
```

##References
-  RAVDESS Dataset
-  Librosa Documentation


License
This project is licensed under the MIT License. See the LICENSE file for details.