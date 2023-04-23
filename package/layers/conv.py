from .layer import Layer
from ..parameter import Parameter
from functools import reduce
import numpy as np


class Conv(Layer):
    def __init__(self, shape, method='VALID', stride=1, requires_grad=True, bias=True, **kwargs):
        """
        shape: (out_channel, kernel_size, kernel_size, in_channel)
        method: The available options for padding are {'VALID', 'SAME'}.
        stride: Convolution stride.
        requires_grad: Whether to compute weight gradients during backpropagation.
        bias: Whether to include bias.

        shape: (out_channel, kernel_size, kernel_size, in_channel)
        method: パディングに使用できるオプションには{'VALID'、 'SAME'}があります。
        stride: 畳み込みストライド。
        requires_grad: 逆伝播中に重みの勾配を計算するかどうか。
        bias: バイアスを含めるかどうか。
        """

        # Randomly initialize W, * indicates taking values from the dictionary. If there are operations to read
        # the saved model later, the values of W and b will be overwritten by the saved model
        # をランダムに初期化し、*は辞書から値を取得することを示します。後で保存されたモデルを読み込む操作がある場合、
        # Wとbの値は保存されたモデルによって上書きされます
        W = np.random.randn(*shape) * (2 / reduce(lambda x, y: x * y, shape[1:]) ** 0.5)

        self.W = Parameter(W, requires_grad)    # 将self.W作为实例保存，其中储存了是否需要计算梯度
        self.b = Parameter(np.zeros(shape[0]), requires_grad) if bias else None

        # Access other data
        # 他のデータにアクセスする
        self.method = method
        self.s = stride
        self.kn = shape[0]  # 畳み込みカーネルの数
        self.ksize = shape[1]
        self.require_grad = requires_grad
        self.first_forward = True
        self.first_backward = True
        self.backward_path = None
        self.b_grad_path = None
        self.W_grad_path = None
        self.ow = None
        self.oh = None
        self.forward_path = None
        self.x_split = None

    def padding(self, x, forward=True):
        # Automatically pad data based on the padding method and whether it is in the forward or backward process
        # パディングの方法と前方向プロセスか後方向プロセスかに応じて、データを自動的にパディングします
        p = self.ksize // 2 if self.method == 'SAME' else self.ksize - 1
        if forward:
            return x if self.method == 'VALID' else np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
        else:
            return np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')

    def split_by_strides(self, x):
        """
        Divide the data into subsets of the same size as the convolutional kernel according to the convolutional stride.
        When it is not divisible by the stride, there will be no out-of-bounds error, but some information data
        will not be used

        データを畳み込みストライドに従ってカーネルと同じサイズのサブセットに分割します。ストライドで割り切れない場合は、
        オーバーフローエラーは発生しませんが、一部の情報データは使用されません
        """
        # Compute the output matrix size
        # 出力行列のサイズを計算する
        N, H, W, C = x.shape
        oh = (H - self.ksize) // self.s + 1
        ow = (W - self.ksize) // self.s + 1
        shape = (N, oh, ow, self.ksize, self.ksize, C)
        strides = (x.strides[0], x.strides[1] * self.s, x.strides[2] * self.s, *x.strides[1:])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def forward(self, x):
        x = self.padding(x)
        if self.s > 1:
            # If the stride of convolution is greater than 1, it is necessary to calculate the width and height of the
            # output data with stride 1 in order to restore the incoming gradient during the backpropagation process
            # 畳み込みのストライドが1より大きい場合、逆伝播の過程で入力された勾配を復元するために、
            # ストライド1の出力データの幅と高さを計算する必要があります
            self.oh = x.shape[1] - self.ksize + 1
            self.ow = x.shape[2] - self.ksize + 1
        self.x_split = self.split_by_strides(x)
        if self.first_forward:
            # Calculate optimization path during first training
            # 最初のトレーニング時に最適化パスを計算します
            self.first_forward = False
            self.forward_path = np.einsum_path('ijk...,o...->ijko', self.x_split, self.W.data, optimize='greedy')[0]
        a = np.einsum('ijk...,o...->ijko', self.x_split, self.W.data, optimize=self.forward_path)
        return a if self.b is None else a + self.b.data

    def backward(self, eta):
        if self.require_grad:
            if self.first_backward:
                # Calculate optimization path during first training
                # 最初のトレーニング時に最適化パスを計算します
                self.W_grad_path = np.einsum_path('...i,...jkl->ijkl', eta, self.x_split, optimize='greedy')[0]
                self.b_grad_path = np.einsum_path('...i->i', eta, optimize='greedy')[0]
            self.W.grad = np.einsum('...i,...jkl->ijkl', eta, self.x_split, optimize=self.W_grad_path) / eta.shape[0]
            if self.b is not None:
                self.b.grad = np.einsum('...i->i', eta, optimize=self.b_grad_path) / eta.shape[0]

        if self.s > 1:
            temp = np.zeros((eta.shape[0], self.oh, self.ow, eta.shape[3]))
            temp[:, ::self.s, ::self.s, :] = eta
            eta = temp

        if self.first_backward:
            # Calculate optimization path during first training
            # 最初のトレーニング時に最適化パスを計算します
            self.first_backward = False
            self.backward_path = np.einsum_path('ijklmn,nlmo->ijko', self.split_by_strides(self.padding(eta, False)),
                                                self.W.data[:, ::-1, ::-1, :], optimize='greedy')[0]

        return np.einsum('ijklmn,nlmo->ijko', self.split_by_strides(self.padding(eta, False)),
                         self.W.data[:, ::-1, ::-1, :], optimize=self.backward_path)
