import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradientDescent as grad

# reading in data
data = pd.read_csv('Data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

#creating matrices
B = np.zeros([len(X), 1])
A = np.zeros([len(X), 2])

for i in range(len(X)):
    B[i, 0] = Y[i]
    A[i, 0] = X[i]
    A[i, 1] = 1

# gradient descent optimization
iterations = 100
learning_rate = 0.0001

theta = np.array([[2], [9]])  # initial condition

theta, cost_history = grad.gradient_descent(A, B, theta, learning_rate, iterations)

print(theta)

# plot cost function
iters = []
for i in range(iterations):
    iters.append(i)

YP = np.dot(A, theta)

plt.figure()
plt.subplot(121)
plt.plot(iters, cost_history, 'r-')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.grid()
plt.subplot(122)
plt.plot(X, Y, 'bo')
plt.plot(X, YP[:, 0], 'r-')
plt.grid()
plt.show()



