import numpy as np


class MSELoss(object):
    def gradient(self):
        # Return the gradient of the loss with respect to the output
        # \frac{\partial L}{\partial a_{ij}}=\frac{a_{ij}-y_{ij}}{C}
        return self.u / self.u.shape[1]
    
    def __call__(self, a, y):
        """
        a: batch of sample outputs
        y: batch of sample ground truth values
        return: the average loss of the batch of samples[, accuracy]

        The shapes of the output and ground truth values are the same, both in batch, and a single output or ground truth value is a one-dimensional vector.
        a.shape = y.shape = (N, C), where N is the number of samples in the batch and C is the length of the final output vector for a single sample.
        """

        # u_{ij} = a_{ij} - y_{ij}
        self.u = a - y
        # Total loss of samples
        # L_{i}=\frac{1}{C} \sum_{j}^{C}\left(a_{ij}-y_{ij}\right)^{2}=\frac{1}{C} \sum_{j}^{C} u_{ij} u_{ij}
        # Average loss of samples
        # L_{mean}=\frac{1}{N} \sum_{i}^{N} L_{i}=\frac{1}{NC} \sum_{i}^{N} \sum_{j}^{C} u_{ij} u_{ij}
        return np.einsum('ij,ij->', self.u, self.u, optimize=True) / self.u.size
