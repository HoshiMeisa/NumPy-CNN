from tqdm import tqdm
from collections import defaultdict
from package.load_model import load_model, save_model
from package.load_dataset import load_train_car, load_train_scenes, load_vali_car, load_vali_scenes
from PIL import Image
import numpy as np
import time


class TRAIN:
    def __init__(self, loss_func, dataset, load_model_path=None, save_model_path=None):
        self.sum_loss_train = float(0)
        self.sum_acc_train = float(0)
        self.epoch = None
        self.loss_func = loss_func
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.dataset = dataset
        self.train_history = defaultdict(list)
        self.randomlist = np.random.permutation(1680)

    def train(self, net_train, Optimizer, epoch):
        self.epoch = epoch
        print(f"Epoch {self.epoch + 1}:")

        if self.load_model_path is not None:
            load_model(net_train.parameters, self.load_model_path)

        if self.dataset == 'car':
            batch_number = 840
        elif self.dataset == 'scenes':
            batch_number = 350
        else:
            raise TypeError

        j = 0
        for i in tqdm(range(batch_number)):
            if self.dataset == 'car':
                x, y = load_train_car(j, self.randomlist)
            elif self.dataset == 'scenes':
                x, y = load_train_scenes(j)
            j += 2

            x /= 255
            output = net_train.forward(x)
            acc, loss = self.loss_func(output, y)
            self.sum_acc_train += acc
            self.sum_loss_train += loss
            eta = self.loss_func.gradient()
            net_train.backward(eta)
            Optimizer.update()

        self.sum_acc_train /= batch_number
        self.sum_loss_train /= batch_number

        self.train_history['train_acc'].append(self.sum_acc_train)
        self.train_history['train_loss'].append(self.sum_loss_train)

        save_model(net_train.parameters, './saved_model/temp/temp_model.npz')
        if self.save_model_path is not None:
            save_model(net_train.parameters, f'{str(self.save_model_path)}/model_epoch{str(self.epoch + 1)}.npz')

    def vali(self, net_vali):
        load_model(net_vali.parameters, f'{str(self.save_model_path)}/model_epoch{str(self.epoch + 1)}.npz')
        print("Validating:")

        if self.dataset == 'car':
            batch_number = 10
        elif self.dataset == 'scenes':
            batch_number = 25
        else:
            raise TypeError

        sum_acc = float(0)
        sum_loss = float(0)
        j = 0

        for i in tqdm(range(batch_number)):
            if self.dataset == 'car':
                X1, Y1 = load_vali_car(j)
            elif self.dataset == 'scenes':
                X1, Y1 = load_vali_scenes(j)
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

    def history(self):
        return self.train_history


def inf(inf_net, dataset, filepath):
    start_time = time.time()
    x = Image.open(filepath, mode='r').resize((224, 224))
    x /= 255
    output = inf_net.forward(x)
    y = np.argmax(output)

    if dataset == 'car':
        if y == 0:
            print("bus")
        elif y == 1:
            print("family sedan")
        elif y == 2:
            print("fire engine")
        elif y == 3:
            print("heavy truck")
        elif y == 4:
            print("jeep")
        elif y == 5:
            print("minibus")
        elif y == 6:
            print("racing car")
        elif y == 7:
            print("SUV")
        elif y == 8:
            print("taxi")
        elif y == 9:
            print("truck")
    elif dataset == 'scenes':
        if y == 0:
            print('church')
        elif y == 1:
            print('desert')
        elif y == 2:
            print('ice')
        elif y == 3:
            print('lawn')
        elif y == 4:
            print('river')
    else:
        raise TypeError

    print(y)
    end_time = time.time()
    print(f"Cost {end_time-start_time}s")
