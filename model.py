import config
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping


class Model():
    def __init__(self):
        pass

    def create_model(self):
        
        self.model = Sequential([
            Conv2D(32, (2,2), activation='relu', input_shape=(40, 300, 1)),
            BatchNormalization(),
            MaxPooling2D((2,2)),   # height=5 is small, so pool only in width
            Conv2D(64, (2,2), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2,2)),

            Conv2D(128, (2,2), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((1,2)),

            Conv2D(256, (2,2), activation='relu'),
            # BatchNormalization(),
            MaxPooling2D((1,2)),

            Conv2D(128, (2,2), activation='relu'),
            # BatchNormalization(),
            MaxPooling2D((1,2)),

            Conv2D(64, (2,2), activation='relu'),
            # BatchNormalization(),
            MaxPooling2D((1,2)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(8, activation='softmax')   # 8 emotions
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

        return self.model
    def train(self,X_train, y_train,X_test, y_test):
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        np.save(os.path.join(config.RESULTS_DIR,"history.npy"), history.history)
        print(f"Training history saved to {os.path.join(config.RESULTS_DIR,"history.npy")}")
        return history
    def save(self,dir):

        self.model.save(dir)