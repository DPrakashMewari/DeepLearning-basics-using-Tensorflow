from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

df= load_breast_cancer()
X,y = df.data,df.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a list of activation function to iterate over 
activationa_fn = [
    'sigmoid',
    'relu',
    'elu',
    'softmax',
    'swish',
    'tanh'
    ]
for activationfn in activationa_fn:
    model = Sequential()
    model.add(Dense(10, activation=activationfn, input_shape=(X_train.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile and train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
     # Evaluate the model on the test set
    accuracys = []
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Activation Function: {activationfn}')
    accuracys.append(accuracy)
    print(f'Test Loss: {loss:.4f} - Test Accuracy: {accuracy:.4f}\n')

print(f"Best Activation Model:{max(accuracys)}")

 # Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
print(y_pred)