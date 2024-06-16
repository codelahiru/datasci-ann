import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
dataset = np.loadtxt('data.csv', delimiter=',')
# Split data into train input (X) and train output (Y) / test input and output variables

train_X = dataset[0:615, 0:8]
train_Y = dataset[0:615, 8]
test_X = dataset[615:768, 0:8]
test_Y = dataset[615:768, 8]

# Define the Keras model
model = Sequential()
model.add(Dense(1, input_dim=8, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(train_X, train_Y, epochs=150, batch_size=10, verbose=0)

# Evaluate the Keras model
_, accuracy = model.evaluate(train_X, train_Y)
print('Accuracy: %.2f' % (accuracy * 100))

#predictions = myANN.predict_classes(data_input)
predictions = (model.predict(test_X) > 0.9).astype(int)

for i in range(20):
  print(test_Y[i], " ", predictions[i])


