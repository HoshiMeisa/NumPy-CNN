from package.layers.activation import Softmax
import numpy as np


# Cross-entropy loss function
# クロスエントロピー損失関数
class CrossEntropyLoss(object):
    def __init__(self):
        # Include a softmax as the classifier
        # 分類器としてsoftmaxを組み込む
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


        a: バッチのサンプル出力
        y: バッチのサンプル正解値
        requires_acc: 正解率を出力するかどうか
        return: そのバッチの平均損失[, 正解率]

        出力と正解値のshapeは同じで、どちらもバッチであり、単一の出力または正解値は一次元ベクトルです。
        a.shape = y.shape = (N, C)、ここでNはバッチ中のサンプル数、Cは単一のサンプルの最終出力ベクトルの長さです。
        """
        # The network's output is not passed through a softmax classification, but rather through the cross-entropy loss function to obtain the softmax classification result.
        # ネットワークの出力はsoftmax分類を通過せず、代わりに交差エントロピー損失関数を通じてsoftmax分類結果を取得します
        a = self.classifier.forward(a)
        # Precompute gradients
        # 勾配の事前計算
        self.grad = a - y
        # Total loss of samples
        # サンプル全体の損失
        # L_{i}=-\sum_{j}^{C} y_{ij} \ln a_{ij}
        # Average loss of samples
        # サンプルの平均損失
        # L_{mean}=\frac{1}{N} \sum_{i}^{N} L_{i}=-\frac{1}{N} \sum_{i}^{N} \sum_{j}^{C} y_{ij} \ln a_{ij}
        loss = -1 * np.einsum('ij,ij->', y, np.log(a), optimize=True) / y.shape[0]
        if requires_acc:
            acc = np.argmax(a, axis=-1) == np.argmax(y, axis=-1)
            return acc.mean(), loss
        return loss
