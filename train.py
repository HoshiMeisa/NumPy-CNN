from tqdm import tqdm
from collections import defaultdict
from package.load_model import load_model, save_model
from package.load_dataset import load_train_car, load_train_scenes, load_vali_car, load_vali_scenes
import numpy as np


class TRAIN:
    def __init__(self, loss_func, dataset, load_model_path=None, save_model_path=None):
        self.sum_loss_train = float(0)
        self.sum_acc_train = float(0)
        self.epoch = 0
        self.loss_func = loss_func
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.dataset = dataset
        self.train_history = defaultdict(list)
        self.car_randomlist = np.random.permutation(1680)
        self.scenes_randomlist = np.random.permutation(2800)

    def train(self, net_train, Optimizer, epoch):
        self.epoch = epoch
        print(f"Epoch {self.epoch + 1}:")

        if self.load_model_path is not None:
            load_model(net_train.parameters, self.load_model_path)
            self.load_model_path = None

        if self.dataset == 'car':
            batch_number = 840
        elif self.dataset == 'scenes':
            batch_number = 25
        else:
            raise TypeError

        j = 0
        for i in tqdm(range(batch_number)):
            if self.dataset == 'car':
                x, y = load_train_car(j, self.car_randomlist)
            elif self.dataset == 'scenes':
                x, y = load_train_scenes(j, self.scenes_randomlist)
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

    def vali(self, net_vali, load_model_path=None):
        if load_model_path is not None:
            load_model(net_vali.parameters, load_model_path)
        else:
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
