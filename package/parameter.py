"""
在网络中，参数被当成一个类。该类不仅储存了参数本身，还储存了梯度等其他信息
"""


class Parameter(object):
    def __init__(self, data, requires_grad, skip_decay=False):
        self.data = data
        self.grad = None
        self.skip_decay = skip_decay
        self.requires_grad = requires_grad
    
    @property
    def T(self):
        return self.data.T
