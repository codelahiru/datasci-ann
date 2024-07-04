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
model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(train_X, train_Y, epochs=200, batch_size=10, verbose=0)

# Evaluate the Keras model
_, accuracy = model.evaluate(train_X, train_Y)
print('Training Accuracy: %.2f' % (accuracy * 100))

#predictions = myANN.predict_classes(data_input)
predictions = (model.predict(test_X) > 0.4).astype(int)

# for i in range(20):
#   print(test_Y[i], " ", predictions[i])

# Evaluate the Keras model
_, accuracy = model.evaluate(test_X, test_Y)
print('Testing Accuracy: %.2f' % (accuracy * 100))







# it is possible to use a hard limit function for your output, where the output is strictly 0 or 1. However,
# hard limit functions (also known as step functions) are not commonly used in the training of neural networks
# because they are not differentiable, which is a requirement for backpropagation.

# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
# # Load data
# dataset = np.loadtxt('data.csv', delimiter=',')
#
# # Split data into input (x) and output (y)
# train_input = dataset[0:615, 0:8]
# train_output = dataset[0:615, 8]
# test_input = dataset[615:768, 0:8]
# test_output = dataset[615:768, 8]
#
# # Define the keras model
# myANN = Sequential()
# myANN.add(Dense(120, input_dim=8, activation='relu'))
# myANN.add(Dense(340, activation='relu'))
# myANN.add(Dense(1, activation='sigmoid'))  # Use sigmoid here
#
# myANN.summary()
#
# # Compile the keras model
# myANN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Fit the keras model on the dataset
# myANN.fit(train_input, train_output, epochs=100, batch_size=10, verbose=0, validation_split=0.2)
#
# # Evaluate the Keras model
# _, accuracy = myANN.evaluate(train_input, train_output)
# print('Accuracy: %.2f' % (accuracy * 100))
#
# # Make predictions
# raw_predictions = myANN.predict(test_input)
#
# # Apply hard limit function (threshold)
# threshold = 0.5
# predictions = (raw_predictions > threshold).astype(int)
#
# # Summarize the first 20 cases
# # for i in range(20):
# #     print(f"Expected: {test_output[i]}, Predicted: {predictions[i][0]}")
