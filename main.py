import package
import package.optim as optim
from train import TRAIN, inf
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

        {'type': 'linear', 'shape': (3200, 64)},
        {'type': 'batchnorm', 'shape': (batch_size, 64)},
        {'type': 'relu'},

        {'type': 'dropout', 'drop_rate': 0.8},

        {'type': 'linear', 'shape': (64, classes)},
        {'type': 'batchnorm', 'shape': (batch_size, classes)},
        {'type': 'relu'},
    ]
    test_layers  = [
        {'type': 'batchnorm', 'shape': (batch_size, 224, 224, 3), 'affine': False, 'requires_grad': False},

        {'type': 'conv', 'shape': (8, 9, 9, 3), 'requires_grad': False},
        {'type': 'batchnorm', 'shape': (216, 216, 8), 'requires_grad': False},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 4},

        {'type': 'conv', 'shape': (16, 5, 5, 8), 'requires_grad': False},
        {'type': 'batchnorm', 'shape': (50, 50, 16), 'requires_grad': False},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 2},

        {'type': 'conv', 'shape': (32, 6, 6, 16), 'requires_grad': False},
        {'type': 'batchnorm', 'shape': (20, 20, 32), 'requires_grad': False},
        {'type': 'relu'},

        {'type': 'maxpool', 'size': 2},

        {'type': 'transform', 'input_shape': (-1, 10, 10, 32), 'output_shape': (-1, 3200)},

        {'type': 'linear', 'shape': (3200, 64), 'requires_grad': False},
        {'type': 'batchnorm', 'shape': (batch_size, 64), 'requires_grad': False},
        {'type': 'relu'},

        {'type': 'dropout', 'drop_rate': 0.8, 'is_test': True},

        {'type': 'linear', 'shape': (64, classes), 'requires_grad': False},
        {'type': 'batchnorm', 'shape': (batch_size, classes), 'requires_grad': False},
        {'type': 'relu'},
    ]

    loss_fn = package.CrossEntropyLoss()
    train_net = package.Net(train_layers)
    test_net = package.Net(test_layers)
    optimizer = optim.Adam(train_net.parameters, 0.00006)

    T = TRAIN(loss_fn, 'car', load_model_path='/home/kana/LinuxData/CNN/saved_model/car/train1/model_epoch4.npz',
              save_model_path='/home/kana/LinuxData/CNN/saved_model/car')
    for epoch in range(99999):
        T.train(net_train=train_net, Optimizer=optimizer, epoch=epoch)
        T.vali(net_vali=test_net)
        plot_acc_loss(T.history())

    # inf_layers   = [
    #         {'type': 'batchnorm', 'shape': (batch_size, 224, 224, 3), 'affine': False, 'requires_grad': False},
    #
    #         {'type': 'conv', 'shape': (8, 9, 9, 3), 'requires_grad': False},
    #         {'type': 'batchnorm', 'shape': (216, 216, 8), 'requires_grad': False},
    #         {'type': 'relu'},
    #
    #         {'type': 'maxpool', 'size': 4},
    #
    #         {'type': 'conv', 'shape': (16, 5, 5, 8), 'requires_grad': False},
    #         {'type': 'batchnorm', 'shape': (50, 50, 16), 'requires_grad': False},
    #         {'type': 'relu'},
    #
    #         {'type': 'maxpool', 'size': 2},
    #
    #         {'type': 'conv', 'shape': (32, 5, 5, 16), 'requires_grad': False},
    #         {'type': 'batchnorm', 'shape': (21, 21, 32), 'requires_grad': False},
    #         {'type': 'relu'},
    #
    #         {'type': 'transform', 'input_shape': (-1, 21, 21, 32), 'output_shape': (-1, 14112)},
    #
    #         {'type': 'linear', 'shape': (14112, 128), 'requires_grad': False},
    #         {'type': 'batchnorm', 'shape': (batch_size, 128), 'requires_grad': False},
    #         {'type': 'relu'},
    #
    #         {'type': 'dropout', 'drop_rate': 0.5, 'is_test': True},
    #
    #         {'type': 'linear', 'shape': (128, classes), 'requires_grad': False},
    #         {'type': 'batchnorm', 'shape': (batch_size, classes), 'requires_grad': False},
    #         {'type': 'relu'},
    #         {'type': 'softmax'},
    #     ]
    # inf_net = package.Net(inf_layers)
    # inf(inf_net=inf_net, dataset='car', filepath='/home/kana/1.png')
