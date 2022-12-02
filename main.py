import package
import package.optim as optim
from train import TRAIN
from plot import plot_acc_loss

if __name__ == "__main__":
    batch_size = 20
    classes = 10

    train_layers = [
        {'type': 'batchnorm', 'shape': (batch_size, 224, 224, 3), 'affine': False},

        {'type': 'conv', 'shape': (8, 9, 9, 3)},
        {'type': 'batchnorm', 'shape': (216, 216, 8)},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 4},

        {'type': 'conv', 'shape': (16, 5, 5, 8)},
        {'type': 'batchnorm', 'shape': (50, 50, 16)},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 2},

        {'type': 'conv', 'shape': (32, 6, 6, 16)},
        {'type': 'batchnorm', 'shape': (20, 20, 32)},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 2},

        {'type': 'transform', 'input_shape': (-1, 10, 10, 32), 'output_shape': (-1, 3200)},

        {'type': 'linear', 'shape': (3200, 128)},
        {'type': 'batchnorm', 'shape': (batch_size, 128)},
        {'type': 'relu'},

        {'type': 'dropout', 'drop_rate': 0.8},

        {'type': 'linear', 'shape': (128, classes)},
        {'type': 'batchnorm', 'shape': (batch_size, classes)},
        {'type': 'relu'},
    ]
    test_layers  = [
        {'type': 'batchnorm', 'shape': (batch_size, 224, 224, 3), 'affine': False},

        {'type': 'conv', 'shape': (8, 9, 9, 3)},
        {'type': 'batchnorm', 'shape': (216, 216, 8)},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 4},

        {'type': 'conv', 'shape': (16, 5, 5, 8)},
        {'type': 'batchnorm', 'shape': (50, 50, 16)},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 2},

        {'type': 'conv', 'shape': (32, 6, 6, 16)},
        {'type': 'batchnorm', 'shape': (20, 20, 32)},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 2},

        {'type': 'transform', 'input_shape': (-1, 10, 10, 32), 'output_shape': (-1, 3200)},

        {'type': 'linear', 'shape': (3200, 128)},
        {'type': 'batchnorm', 'shape': (batch_size, 128)},
        {'type': 'relu'},

        {'type': 'dropout', 'drop_rate': 0.8, 'is_test': True},

        {'type': 'linear', 'shape': (128, classes)},
        {'type': 'batchnorm', 'shape': (batch_size, classes)},
        {'type': 'relu'},
    ]

    loss_fn = package.CrossEntropyLoss()
    train_net = package.Net(train_layers)
    test_net = package.Net(test_layers)
    optimizer = optim.Adam(train_net.parameters, 0.0003)

    T = TRAIN(loss_fn, 'car', save_model_path='/home/kana/LinuxData/CNN/saved_model/car')
    for epoch in range(99999):
        T.train(net_train=train_net, Optimizer=optimizer, epoch=epoch)
        T.vali(net_vali=test_net)
        plot_acc_loss(T.history())
