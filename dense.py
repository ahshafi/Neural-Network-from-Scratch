from sqlite3 import Row
import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.uniform(-.1, .1, (output_size, input_size))
        self.bias = np.random.uniform(-.1, .1, (output_size, 1))

    def forward(self, _input):
        self.input = _input
        # print("input", self.input)
        # assert not np.any(np.isnan(self.weights))
        # assert not np.any(np.isnan(self.input))
        # assert not np.any(np.isnan(np.dot(self.weights, self.input))), self.weights
        # print("forward", self.weights.shape, self.input.shape, self.bias.shape)
        tmp = np.dot(self.weights, self.input) + self.bias
        # print("output", tmp)
        # # input()
        # assert not np.any(np.isnan(tmp))
        return tmp

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        # self.weights = (self.weights - self.weights.min(axis=1, keepdims=True)) / (self.weights.max(axis=1, keepdims=True) - self.weights.min(axis=1, keepdims=True))
        # self.weights /= np.sum(self.weights, axis= 1).reshape(self.weights.shape[0], 1)
        # print("output_gradient", output_gradient.shape)
        tmp = np.array([sum(row) for row in output_gradient])
        tmp = tmp.reshape(tmp.shape[0], 1)
        self.bias -= learning_rate * tmp
        # self.bias = (self.bias - self.bias.min(axis=0, keepdims=True)) / (self.bias.max(axis=0, keepdims=True) - self.bias.min(axis=0, keepdims=True))
        # self.bias /= np.sum(self.bias)
        return input_gradient
