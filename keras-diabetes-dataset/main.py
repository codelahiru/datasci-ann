import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
dataset = np.loadtxt('data.csv', delimiter=',')
# Split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

# Evaluate the Keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy * 100))

#predictions = myANN.predict_classes(data_input)
predictions = (model.predict(X) > 0.5).astype(int)

for i in range(20):
  print(Y[i], " ", predictions[i])


