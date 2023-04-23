from .layers import *


class Net(Layer):
    def __init__(self, layer_configures):
        self.layers = []
        self.parameters = []
        for config in layer_configures:
            self.layers.append(self.createLayer(config))

    def createLayer(self, config):
        return self.getDefaultLayer(config)

    def getDefaultLayer(self, config):
        t = config['type']
        if t == 'linear':
            new_layer = Linear(**config)
            self.parameters.append(new_layer.W)
            if new_layer.b is not None:
                self.parameters.append(new_layer.b)
        elif t == 'leakyrelu':
            new_layer = LeakyRelu()
        elif t == 'relu':
            new_layer = Relu()
        elif t == 'softmax':
            new_layer = Softmax()
        elif t == 'sigmoid':
            new_layer = Sigmoid()
        elif t == 'tanh':
            new_layer = Tanh()
        elif t == 'dropout':
            new_layer = Dropout(**config)
        elif t == 'transform':
            new_layer = Transform(**config)
        elif t == 'conv':
            new_layer = Conv(**config)
            self.parameters.append(new_layer.W)
            if new_layer.b is not None:
                self.parameters.append(new_layer.b)
        elif t == 'maxpool':
            new_layer = MaxPooling(**config)
        elif t == 'upsample':
            new_layer = Unsample(**config)
        elif t == 'reflectionpad':
            new_layer = ReflectionPad(**config)
        elif t == 'batchnorm':
            new_layer = BatchNorm(**config)
            if new_layer.affine:
                self.parameters.append(new_layer.gamma)
                self.parameters.append(new_layer.beta)
            self.parameters.append(new_layer.overall_ave)
            self.parameters.append(new_layer.overall_var)
        elif t == 'infermean':
            new_layer = InferMean()
        else:
            raise TypeError
        return new_layer

    def forward(self, x):
        for layer_forward in self.layers:
            x = layer_forward.forward(x)
        return x
    
    def backward(self, eta):
        for layer_backward in self.layers[::-1]:
            eta = layer_backward.backward(eta)
        return eta
