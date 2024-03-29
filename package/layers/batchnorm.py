from .layer import Layer
from ..parameter import Parameter
import numpy as np


class BatchNorm(Layer):
    def __init__(self, shape, requires_grad=True, affine=True, is_test=False, **kwargs):
        if affine:
            # Parameters for input normalization without the need for affine transformation
            self.gamma = Parameter(np.random.uniform(0.9, 1.1, shape), requires_grad, True)
            self.beta = Parameter(np.random.uniform(-0.1, 0.1, shape), requires_grad, True)
            self.requires_grad = requires_grad
        self.eps = 1e-8
        self.affine = affine
        self.is_test = is_test
        self.coe = 0.02
        self.overall_var = Parameter(np.zeros(shape), requires_grad=False)
        self.overall_ave = Parameter(np.zeros(shape), requires_grad=False)
        self.gamma_s = None
        self.normalized = None

    def forward(self, x):
        if self.is_test:
            # Normalize using the estimated overall variance and mean of the training set during testing
            sample_ave = self.overall_ave.data
            sample_std = np.sqrt(self.overall_var.data)
        else:
            # During training, the mean and variance of the samples are used to estimate the mean and
            # variance of the entire training set using a weighted average.
            # sample_ave = x.mean(axis=0)
            sample_var = x.var(axis=0)
            sample_std = np.sqrt(sample_var + self.eps)
            self.overall_ave.data = (1 - self.coe) * self.overall_ave.data + self.coe * sample_ave
            self.overall_var.data = (1 - self.coe) * self.overall_var.data + self.coe * sample_var
        return (x - sample_ave) / sample_std if not self.affine else self.forward_internal(x - sample_ave, sample_std)

    def backward(self, eta):
        # If normalization is performed on the input layer, there will be no gradient propagation
        if not self.affine:
            return
        self.beta.grad = eta.mean(axis=0)
        self.gamma.grad = (eta * self.normalized).mean(axis=0)
        return self.gamma_s * (eta - self.normalized * self.gamma.grad - self.beta.grad)

    def forward_internal(self, sample_diff, sample_std):
        # If BatchNorm is used within the network, further affine transformation is necessary
        # If normalization is applied to the input, it is not needed
        self.normalized = sample_diff / sample_std
        self.gamma_s = self.gamma.data / sample_std
        return self.gamma.data * self.normalized + self.beta.data
