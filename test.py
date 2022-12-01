import numpy as np


class FC:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  # 初始化
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        self.b3 = np.random.rand(1)
        self.b4 = np.random.rand(1)
        self.activation_function = Sigmoid.forward

    def fc_forward(self, input_list, outputs_list, hparameters):  # 训练
        # 前向传播
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(outputs_list).T
        hidden_inputs = np.dot(self.wih, inputs) + self.b3
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs) + self.b4
        final_outputs = softmax(final_inputs)
        return final_outputs, hidden_outputs, inputs

    def fc_backward(self, grad, final_outputs, hidden_outputs, inputs):
        hidden_errors = np.dot(self.who.T, grad)
        self.who += self.lr * np.dot((grad * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs)))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        self.b4 -= self.lr * np.sum(grad, axis=0)
        self.b3 -= self.lr * np.sum(hidden_errors, axis=0)
