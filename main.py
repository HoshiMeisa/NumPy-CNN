import yaml
import package
import package.optim as optim

from train import Train
from plot import plot_acc_loss

with open('layers.yaml') as f:
    config = yaml.safe_load(f)


if __name__ == "__main__":
    batch_size = 10
    classes = 5

    train_layers = config['train_layers']
    test_layers = config['test_layers']

    loss_fn = package.CrossEntropyLoss()
    train_net = package.Net(train_layers)
    test_net = package.Net(test_layers)
    optimizer = optim.Adam(train_net.parameters, 0.0003)

    T = Train(loss_fn, 'scenes',
              save_model_path='/home/kana/LinuxData/CNN/saved_model/scenes/new')
    for epoch in range(99999):
        T.train(net_train=train_net, Optimizer=optimizer, epoch=epoch)
        T.vali(net_vali=test_net)
        plot_acc_loss(T.history())
