from .layers import *


class Net(Layer):
    def __init__(self, layer_configures):
        # 创建空列表，用来储存网络的各个层
        self.layers = []
        # 创建空列表，用来储存网络的权重
        self.parameters = []
        # 读取各个层的信息，创建所需的层
        for config in layer_configures:
            self.layers.append(self.createLayer(config))

    def createLayer(self, config):
        """
        方法：获取新的层
        参数：config -- 层的类型
        返回：self.getDefaultLayer(config) -- 新的一层
        """
        return self.getDefaultLayer(config)

    def getDefaultLayer(self, config):
        """
        方法：创建新的层
        参数：config -- 层的类型
        返回：new_layer -- 新的一层
        """
        t = config['type']      # config是一个字典，此处读取键“type”对应的值，即层的类型
        if t == 'linear':
            new_layer = Linear(**config)           # 创建名为new_layer的实例（新层），两个星号代表只取字典里的值
            self.parameters.append(new_layer.W)    # 初始化权重并添加到储存权重的列表中
            if new_layer.b is not None:            # 如果指出需要设置偏差，则将偏差也添加到储存权重的列表中
                self.parameters.append(new_layer.b)
        elif t == 'leakyrelu':                          # 下面的层同理
            new_layer = LeakyRelu()
        elif t == 'relu':
            new_layer = Relu()
        elif t == 'softmax':
            new_layer = Softmax()
        elif t == 'sigmoid':
            new_layer = Sigmoid()
        elif t == 'tanh':
            new_layer = Tanh()
        elif t == 'dropout':
            new_layer = Dropout(**config)
        elif t == 'transform':
            new_layer = Transform(**config)
        elif t == 'conv':
            new_layer = Conv(**config)
            self.parameters.append(new_layer.W)
            if new_layer.b is not None:
                self.parameters.append(new_layer.b)
        elif t == 'maxpool':
            new_layer = MaxPooling(**config)
        elif t == 'upsample':
            new_layer = Unsample(**config)
        elif t == 'reflectionpad':
            new_layer = ReflectionPad(**config)
        elif t == 'batchnorm':
            new_layer = BatchNorm(**config)
            if new_layer.affine:
                self.parameters.append(new_layer.gamma)
                self.parameters.append(new_layer.beta)
            self.parameters.append(new_layer.overall_ave)
            self.parameters.append(new_layer.overall_var)
        elif t == 'batchnorm_inf':
            new_layer = Batchnorm_Inf()
        else:
            raise TypeError
        return new_layer

    def forward(self, x):
        for layer_forward in self.layers:
            x = layer_forward.forward(x)
        return x
    
    def backward(self, eta):
        for layer_backward in self.layers[::-1]:
            eta = layer_backward.backward(eta)
        return eta
