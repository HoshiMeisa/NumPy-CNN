from collections import defaultdict
import numpy as np
from tqdm import tqdm
from package.load_dataset import load_train_car, load_train_scenes, load_vali_car, load_vali_scenes
from package.load_model import load_model, save_model


class Train:
    def __init__(self, loss_func, dataset, load_model_path=None, save_model_path=None):
        self.sum_loss_train = 0.0
        self.sum_acc_train = 0.0
        self.epoch = 0
        self.loss_func = loss_func
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.dataset = dataset
        self.train_history = defaultdict(list)
        self.car_randomlist = np.random.permutation(1680)
        self.scenes_randomlist = np.random.permutation(2800)

    def train(self, net_train, optimizer, epoch):
        self.epoch = epoch
        print(f"Epoch {self.epoch + 1}:")

        # 加载模型
        if self.load_model_path is not None:
            load_model(net_train.parameters, self.load_model_path)
            self.load_model_path = None

        # 设置批次数
        if self.dataset == 'car':
            batch_number = 840
        elif self.dataset == 'scenes':
            batch_number = 25
        else:
            raise TypeError

        j = 0
        for i in tqdm(range(batch_number)):
            # 加载训练数据
            if self.dataset == 'car':
                x, y = load_train_car(j, self.car_randomlist)
            elif self.dataset == 'scenes':
                x, y = load_train_scenes(j, self.scenes_randomlist)
            j += 2

            # 数据归一化
            x /= 255

            # 前向传播
            output = net_train.forward(x)

            # 计算损失和准确率
            acc, loss = self.loss_func(output, y)
            self.sum_acc_train += acc
            self.sum_loss_train += loss

            # 反向传播
            eta = self.loss_func.gradient()
            net_train.backward(eta)
            optimizer.update()

        # 计算平均训练准确率和损失
        self.sum_acc_train /= batch_number
        self.sum_loss_train /= batch_number

        # 保存模型和训练历史
        save_model(net_train.parameters, './saved_model/temp/temp_model.npz')
        self.train_history['train_acc'].append(self.sum_acc_train)
        self.train_history['train_loss'].append(self.sum_loss_train)

        if self.save_model_path is not None:
            save_model(net_train.parameters, f'{str(self.save_model_path)}/model_epoch{str(self.epoch + 1)}.npz')

    def vali(self, net_vali, load_model_path=None):
        # 加载模型
        if load_model_path is not None:
            load_model(net_vali.parameters, load_model_path)
        else:
            load_model(net_vali.parameters, f'{str(self.save_model_path)}/model_epoch{str(self.epoch + 1)}.npz')

        print("Validating:")

        # 设置批次数
        if self.dataset == 'car':
            batch_number = 10
        elif self.dataset == 'scenes':
            batch_number = 25
        else:
            raise TypeError

        sum_acc = 0.0
        sum_loss = 0.0
        j = 0

        for i in tqdm(range(batch_number)):
            # 加载验证数据
            if self.dataset == 'car':
                x, y = load_vali_car(j)
            elif self.dataset == 'scenes':
                x, y = load_vali_scenes(j)
            j += 2

            # 数据归一化
            x /= 255

            # 前向传播
            output = net_vali.forward(x)

            # 计算损失和准确率
            batch_acc, batch_loss = self.loss_func(output, y)
            sum_acc+= batch_acc
            sum_loss += batch_loss

        # 计算平均验证准确率和损失
        sum_acc /= batch_number
        sum_loss /= batch_number

        # 保存验证历史
        self.train_history['val_acc'].append(sum_acc)
        self.train_history['val_loss'].append(sum_loss)

        # 输出验证结果和训练结果
        print()
        print("=======================================")
        print(f"Epoch {self.epoch + 1}")
        print("---------------------------------------")
        print(f"Vali_Acc : {sum_acc:.5f}, Vali_Loss : {sum_loss:.5f}")
        print("---------------------------------------")
        print(f"Train_Acc: {self.sum_acc_train:.5f}, Train_Loss: {self.sum_loss_train:.5f}")
        print("=======================================")
        print()

    def history(self):
        return self.train_history