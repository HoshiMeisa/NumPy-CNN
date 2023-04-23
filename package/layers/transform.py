from .layer import Layer

# In regular convolutional networks, data needs to be reshaped due to the presence of linear layers
# 通常の畳み込みニューラルネットワークでは、線形層が存在するため、データを再形成する必要があります


class Transform(Layer):
    def __init__(self, input_shape, output_shape, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        return x.reshape(self.output_shape)

    def backward(self, eta):
        return eta.reshape(self.input_shape)
