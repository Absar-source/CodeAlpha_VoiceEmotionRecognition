import config
import librosa, librosa.display

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

class Data_Prepare():
    def __init__(self) -> None:
        self.emotion_dict = {
                '01': 'neutral',
                '02': 'calm',
                '03': 'happy',
                '04': 'sad',
                '05': 'angry',
                '06': 'fearful',
                '07': 'disgust',
                '08': 'surprised'
            }

    def load(self,data_dir):
        data = []
        label = []
        


        for root, dirs, files in (os.walk(data_dir)):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    
                    # Load audio
                    y, sr = librosa.load(file_path, sr=None)
                    # Extract MFCC features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    data.append(mfccs)
                    label.append(self.get_emotion(file))  
        return data,label
    
    def standarize(self,mfccs, max_len=340):
        """
        MFCC convert into standard shape (n_mfcc, max_len) .
        
        Args:
            mfcc: numpy array of shape (n_mfcc, time_frames)
            max_len: target time frame length (default 300)

        Returns:
            standardized mfcc of shape (n_mfcc, max_len)
        """
        standardized_mfccs = []
        for mfcc in mfccs:    
            n_mfcc, t = mfcc.shape
            
            if t < max_len:  # padding
                pad_width = max_len - t
                mfcc_padded = np.pad(mfcc, ((0,0), (0, pad_width)), mode='constant')
                standardized_mfccs.append(mfcc_padded)
            
            elif t > max_len:  # truncation
                standardized_mfccs.append(mfcc[:, :max_len])
            
            else:  # already equal
                standardized_mfccs.append(mfcc)
        return np.array(standardized_mfccs)
    
    
    def get_emotion(self,path):
        filename = os.path.basename(path)
        emotion_code = filename.split("-")[2]
        emotion = self.emotion_dict.get(emotion_code, "unknown")
        return emotion
    
    def split(self,data,label):
                
        scaler = StandardScaler()
        X = data.reshape(data.shape[0], -1)  # flatten
        X = scaler.fit_transform(X)
        X = X.reshape(-1, 40, 300, 1)

        # data = [...]   # list of MFCC arrays
        # labels = [...] # list of emotion strings


        le = LabelEncoder()
        y = le.fit_transform(label)   # e.g. ['happy','sad'] → [2,5]
        y = to_categorical(y)

        # Save the LabelEncoder classes
        np.save(os.path.join(config.RESULTS_DIR, "label_classes.npy"), le.classes_)
        print("Label classes saved to 'label_classes.npy'.")


        print("Final dataset shape:", X.shape, y.shape)  
        # Example: (1440, 300, 40), (1440, 8)

        # Step 4: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Train shape:", X_train.shape, y_train.shape)
        print("Test shape:", X_test.shape, y_test.shape)

        return X_train, X_test, y_train, y_test
