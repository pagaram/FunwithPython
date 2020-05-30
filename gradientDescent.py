import numpy as np


def cost_function(theta, A, B):
    m = len(A[:, 0])
    predictions = np.dot(A, theta)
    cost = (1/(1*float(m)))*np.sum(np.square(predictions - B))

    return cost


def gradient_descent(A, B, theta, learning_rate, iterations):
    m = len(A[:, 0])
    cost_history = np.zeros([iterations, 1])

    for it in range(iterations):
        prediction = np.dot(A, theta)
        theta = theta - (2/float(m))*learning_rate*(np.dot(np.transpose(A), (prediction-B)))
        cost_history[it, 0] = cost_function(theta, A, B)

    return theta, cost_history



