import numpy as np
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_relu(Z):
    return Z > 0


def huber_loss(y_true, y_pred, delta=0.1):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)


def derivative_huber(error, delta=1.0):
    # error is A3 - target
    return np.where(np.abs(error) <= delta, error, delta * np.sign(error))
