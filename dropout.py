from layer import Layer
import numpy as np
class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
         
    def forward(self, _input):
        self.input = _input
        self.dropout_vec = np.random.binomial(1, self.dropout_rate, size=_input.shape[0]).reshape(_input.shape[0], 1).astype("float64") 
        self.dropout_vec /= (1 - self.dropout_rate)
        self.output = self.dropout_vec * _input
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.dropout_vec