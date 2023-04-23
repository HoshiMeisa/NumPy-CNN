from .layer import Layer
import numpy as np


# Random dropout
# ランダムドロップアウト
class Dropout(Layer):
    def __init__(self, drop_rate, is_test=False, **kwargs):
        """
        drop_rate: dropout probability
        is_test: indicates whether the system is in test mode

        rop_rate: ドロップアウト率
        is_test: システムがテストモードであるかどうかを示します
        """
        self.backward_path = None
        self.forward_path = None
        self.mask = None
        self.drop_rate = drop_rate
        # Correction value, to keep the overall expectation of forward output unchanged
        # 修正値を使用して、forwardの出力の全体的な期待値を不変に保ちます
        self.fix_value = 1 / (1 - drop_rate)
        self.is_test = is_test
        self.first_forward = True
        self.first_backward = True

    def forward(self, x):
        if self.is_test:
            # If it is a test, return directly
            # もしテストであれば、直接戻ります
            return x
        else:
            # If it's training, set the output to 0 with probability
            # 学習時は確率に応じて出力を0に設定
            self.mask = np.random.uniform(0, 1, x.shape) > self.drop_rate
            if self.first_forward:
                # Calculate optimization path during first training
                # 最初のトレーニング時に最適化パスを計算します
                self.first_forward = False
                self.forward_path = np.einsum_path('...,...,->...', x, self.mask, self.fix_value, optimize='greedy')[0]
            return np.einsum('...,...,->...', x, self.mask, self.fix_value, optimize=self.first_forward)

    def backward(self, eta):
        if self.is_test:
            return eta
        else:
            if self.first_backward:
                # Calculate optimization path during first training
                # 最初のトレーニング時に最適化パスを計算します
                self.first_backward = False
                self.backward_path = np.einsum_path('...,...->...', eta, self.mask, optimize='greedy')[0]
            return np.einsum('...,...->...', eta, self.mask, optimize=self.backward_path)
