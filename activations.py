import numpy as np
from layer import Layer
from activation import Activation


def reLU(x):
        # print("reLU", x)
        # input()
        return np.maximum(x, 0).astype("float64")

def reLU_prime(x):
    return (x > 0).astype("float64")
class ReLU(Activation):
    def __init__(self):
        super().__init__(reLU, reLU_prime)


class Softmax(Layer):
    def forward(self, _input):
        self.input = _input
        # print("softmax input", self.input)
        tmp = np.copy(_input)
        tmp -= np.max(tmp, axis= 0)
        # print(tmp)
        tmp = np.exp(tmp)
        # print("tmp", tmp.shape)
        # print(tmp)
        self.output = tmp / np.sum(tmp, axis= 0)
        # print("softmax output", self.output)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n, m = np.shape(self.output)
        
        input_gradient =np.array([np.dot((np.identity(n) - self.output[:,i].T) * self.output[:,i], output_gradient[:,i])
                                   for i in range(m)]).T
        
        return input_gradient
       

    
