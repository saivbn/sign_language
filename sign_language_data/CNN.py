import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


labels = ["A", "B", "C", "D"]  
data = []
targets = []

for i, label in enumerate(labels):
    df = pd.read_csv(f'sign_language_data/{label}.csv', header=None)
    data.append(df.values)
    targets.extend([i] * len(df))

data = np.vstack(data)
targets = np.array(targets)

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
model.save("sign_language_model.keras")

