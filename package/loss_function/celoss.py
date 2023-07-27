from package.layers.activation import Softmax
import numpy as np


# Cross-entropy loss function
class CrossEntropyLoss(object):
    def __init__(self):
        # Include a softmax as the classifier
        self.classifier = Softmax()

    def gradient(self):
        return self.grad

    def __call__(self, a, y, requires_acc=True):
        """
        a: batch of sample outputs
        y: batch of sample ground truth values
        requires_acc: whether to output accuracy
        return: the average loss of the batch of samples[, accuracy]

        The shapes of the output and ground truth values are the same, both in batch, and a single output or ground truth value is a one-dimensional vector.
        a.shape = y.shape = (N, C), where N is the number of samples in the batch and C is the length of the final output vector for a single sample.
        """
        # The network's output is not passed through a softmax classification, but rather through the cross-entropy loss function to obtain the softmax classification result.
        a = self.classifier.forward(a)
        # Precompute gradients
        self.grad = a - y
        # Total loss of samples
        # L_{i}=-\sum_{j}^{C} y_{ij} \ln a_{ij}
        # Average loss of samples
        # L_{mean}=\frac{1}{N} \sum_{i}^{N} L_{i}=-\frac{1}{N} \sum_{i}^{N} \sum_{j}^{C} y_{ij} \ln a_{ij}
        loss = -1 * np.einsum('ij,ij->', y, np.log(a), optimize=True) / y.shape[0]
        if requires_acc:
            acc = np.argmax(a, axis=-1) == np.argmax(y, axis=-1)
            return acc.mean(), loss
        return loss
