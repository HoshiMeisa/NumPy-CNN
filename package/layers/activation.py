from abc import ABC
from .layer import Layer
import numpy as np


class Relu(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta


class LeakyRelu(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        x[x < 0] = x[x < 0] / 5.5
        return x

    def backward(self, eta):
        eta[self.x < 0] = self.x[self.x < 0] / 5.5
        return eta


class Softmax(Layer):
    def forward(self, x):
        """
        x.shape = (N, C)
        接收批量的输入，每个输入是一维向量
        """
        v = np.exp(x)
        return v / v.sum(axis=-1, keepdims=True)
    
    def backward(self, y):
        # Softmax的反向传播在交叉熵损失函数中
        pass


class Sigmoid(Layer):
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, eta):
        return np.einsum('...,...,...->...', self.y, 1 - self.y, eta, optimize=True)


class Tanh(Layer):
    def __init__(self):
        self.y = None

    def forward(self, x):
        ex = np.exp(x)
        esx = np.exp(-x)
        self.y = (ex - esx) / (ex + esx)
        return self.y

    def backward(self, eta):
        return np.einsum('...,...,...->...', 1 - self.y, 1 + self.y, eta, optimize=True)


class InferMean(Layer):
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = np.mean(x, axis=0)
        return self.y

    def backward(self, eta):
        raise TypeError('InferMean is just used for inference.')
