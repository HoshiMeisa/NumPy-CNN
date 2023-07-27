from .layer import Layer


class MaxPooling(Layer):
    def __init__(self, size, **kwargs):
        """
        size: The window size of pooling is simplified to one parameter because in use, the window size is basically
        consistent with the stride
        """
        self.mask = None
        self.size = size

    def forward(self, x):
        # First, divide the input into several subsets according to the window size
        out = x.reshape(x.shape[0], x.shape[1] // self.size, self.size, x.shape[2] // self.size, self.size, x.shape[3])
        # Take the maximum value of each subset
        out = out.max(axis=(2, 4))
        # Record the positions in each window that are not the maximum value for backpropagation
        self.mask = out.repeat(self.size, axis=1).repeat(self.size, axis=2) != x
        return out

    def backward(self, eta):
        # Copy the gradient passed from the previous layer and expand its shape to match the size of the input in forward
        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
        # Set the gradient of positions other than the maximum value to 0
        eta[self.mask] = 0
        return eta
