from .layer import Layer
import numpy as np


# Padding is applied to the data using the values of boundary elements, and the padding only occurs in the
# width (W) and height (H) dimensions

class EdgePad(Layer):
    def __init__(self, pad_width, **kwargs):
        # pad_width: Padding width
        self.top = self.bottom = self.left = self.right = pad_width

    def forward(self, x):
        return np.pad(x, ((0, 0), (self.top, self.bottom), (self.left, self.right), (0, 0)), 'edge')

    def backward(self, eta):
        eta[:, self.top, :, :] += eta[:, :self.top, :, :].sum(axis=1)
        eta[:, -self.bottom - 1, :, :] += eta[:, -self.bottom:, :, :].sum(axis=1)
        eta[:, :, self.left, :] += eta[:, :, :self.left, :].sum(axis=2)
        eta[:, :, -self.right - 1, :] += eta[:, :, -self.right:, :].sum(axis=2)
        return eta[:, self.top:-self.bottom, self.left:-self.right, :]
