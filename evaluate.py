import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_preprocessing import Data_Prepare
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import config
import os

# Load the trained model

model = load_model(config.MODEL_DIR)

# Load the label encoder
le = LabelEncoder()
le.classes_ = np.load( os.path.join( config.RESULTS_DIR, "label_classes.npy"))  # Assuming label classes are saved during training

# 1. Load the audio file
y_new, sr_new = librosa.load(r"data\Actor_01\03-01-01-01-01-01-01.wav", sr=None)

# 2. Extract MFCC features (same as training)
mfcc_new = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=40)

# 3. Standardize shape (same as training)
DPrep = Data_Prepare()
mfcc_new_std = DPrep.standarize([mfcc_new], max_len=300)  # returns shape (1, 40, 300)

# 4. Reshape for model input (add channel dimension)
mfcc_new_std = mfcc_new_std.reshape(1, 40, 300, 1)

# 5. Predict
if model:
    pred = model.predict(mfcc_new_std)


    # 6. Decode the predicted class
    pred_class = np.argmax(pred, axis=1)[0]
    pred_label = le.inverse_transform([pred_class])[0]

    print("Predicted emotion:", pred_label)

else:
    print("model have not find in :" ,config.MODEL_DIR)