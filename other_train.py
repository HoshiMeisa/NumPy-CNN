import package
import numpy as np
from package.load_dataset import read
import package.optim as optim
from tqdm import tqdm
from collections import defaultdict
from package.load_model import load_model, save_model
from plot import plot_acc_loss


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
net = package.Net(train_layers)
loss_func = package.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters, 0.0003)

train_history = defaultdict(list)
sum_acc_train = 0
sum_loss_train = 0

j = 0
load_model(net.parameters, '')
for i in tqdm(range(10)):
    j += 2
    x, y = load(j)
    x /= 255
    output = net.forward(x)
    acc, loss = loss_func(output, y)

    sum_acc_train += acc
    sum_loss_train += loss

    eta = loss_func.gradient()
    net.backward(eta)
    optimizer.update()

sum_acc_train /= 10
sum_loss_train /= 10

train_history['train_acc'].append(sum_acc_train)
train_history['train_loss'].append(sum_loss_train)

save_model(net.parameters, './saved_model/temp/temp_model.npz')

sum_acc = float(0)
sum_loss = float(0)
j = 0

for i in tqdm(range(10)):
    j += 2

    X1 /= 255
    output = net_vali.forward(X1)
    batch_acc, batch_loss = self.loss_func(output, Y1)

    sum_acc += batch_acc
    sum_loss += batch_loss

sum_acc /= batch_number
sum_loss /= batch_number

self.train_history['val_acc'].append(sum_acc)
self.train_history['val_loss'].append(sum_loss)

print()
print("=======================================")
print("Epoch %d" % (self.epoch + 1))
print("---------------------------------------")
print("Vali_Acc : %.5f, Vali_Loss : %.5f" % (sum_acc, sum_loss))
print("---------------------------------------")
print("Train_Acc: %.5f, Train_Loss: %.5f" % (self.sum_acc_train, self.sum_loss_train))
print("=======================================")
print()


plot_acc_loss(train_history)

