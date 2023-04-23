from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    """
    As a base class for all layers, if you want to define a new layer, you should inherit from this class and override
    the following two methods

    すべてのレイヤーの基本クラスとして、新しいレイヤーを定義する場合は、このクラスから継承し、
    以下の2つのメソッドをオーバーライドする必要があります
    """
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass
