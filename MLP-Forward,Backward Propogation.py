import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the MLP model using Keras (TensorFlow backend)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(5, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 1000
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Evaluate the model on training data
_, train_accuracy = model.evaluate(X_train, y_train)
print("Train Accuracy:", train_accuracy)

# Evaluate the model on test data
_, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
