from .layer import Layer


class MaxPooling(Layer):
    def __init__(self, size, **kwargs):
        """
        size: The window size of pooling is simplified to one parameter because in use, the window size is basically
        consistent with the stride

        プーリングのウィンドウサイズは、使用中にウィンドウサイズがストライドと基本的に一致するため、1つのパラメータに簡略化されています
        """
        self.mask = None
        self.size = size

    def forward(self, x):
        # First, divide the input into several subsets according to the window size
        # 最初に、ウィンドウサイズに応じて入力をいくつかのサブセットに分割します
        out = x.reshape(x.shape[0], x.shape[1] // self.size, self.size, x.shape[2] // self.size, self.size, x.shape[3])
        # Take the maximum value of each subset
        # 各サブセットの最大値を取得します
        out = out.max(axis=(2, 4))
        # Record the positions in each window that are not the maximum value for backpropagation
        # 逆伝播のために、各ウィンドウで最大値でない位置を記録します
        self.mask = out.repeat(self.size, axis=1).repeat(self.size, axis=2) != x
        return out

    def backward(self, eta):
        # Copy the gradient passed from the previous layer and expand its shape to match the size of the input in forward
        # 前の層から渡された勾配をコピーし、その形状を前向きに入力されたサイズに拡張します
        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
        # Set the gradient of positions other than the maximum value to 0
        # 最大値以外の位置の勾配を0に設定します
        eta[self.mask] = 0
        return eta
