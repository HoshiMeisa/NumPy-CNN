class SGD(object):
    def __init__(self, parameters, lr, decay=0):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.decay_rate = 1.0 - decay

    def update(self):
        for p in self.parameters:
            if self.decay_rate < 1 and not p.skip_decay:
                p.data *= self.decay_rate
            p.data -= self.lr * p.grad
