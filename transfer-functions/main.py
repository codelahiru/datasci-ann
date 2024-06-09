import matplotlib.pyplot as plt
import numpy as np

# sigmoid / log sigmoid / soft step
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# identity / linear
def identity(x):
  return x

#ReLU function
def ReLU(x):
  return np.maximum(0, x)


# creating vectors x and y
x = np.linspace(-20, 20, 100)

# fig = plt.figure(figsize=(10, 6))
# plt.plot(x, sigmoid(x), label='sigmoid')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# fig = plt.figure(figsize=(10, 6))
# plt.plot(x, identity(x), label='identity')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# fig = plt.figure(figsize=(10, 6))
# plt.plot(x, ReLU(x), label='ReLU')
# plt.grid(True)
# plt.legend()
# plt.show()

# Plot all functions in one figure
fig = plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid(x), label='sigmoid')
plt.plot(x, identity(x), label='identity')
plt.plot(x, ReLU(x), label='ReLU')
plt.grid(True)
plt.legend()
plt.show()

# # creating vectors x and y near zero
# x = np.linspace(-1, 1, 100)
#
# # Plot ReLU function near zero
# fig = plt.figure(figsize=(10, 6))
# plt.plot(x, ReLU(x), label='ReLU')
# plt.axhline(0, color='gray', lw=0.5)
# plt.axvline(0, color='gray', lw=0.5)
# plt.grid(True)
# plt.legend()
# plt.title('ReLU Function Near Zero')
# plt.show()