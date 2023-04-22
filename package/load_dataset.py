import numpy as np
from PIL import Image


def read(filepath, k):
    """
    参数：图片文件路径, k
    """
    img2 = np.zeros((2, 224, 224, 3))
    j = 0
    for i in range(k, k + 2):
        img = Image.open(filepath + "/" + str(i) + ".jpg", mode='r').resize((224, 224))
        img2[j] = img
        j += 1
    return img2


def readlist(filepath, k, randomlist):
    """
    参数：图片文件路径, k
    """
    img2 = np.zeros((2, 224, 224, 3))
    j = 0
    for i in range(k, k + 2):
        img = Image.open(filepath + "/" + str(randomlist[i]) + ".jpg", mode='r').resize((224, 224))
        img2[j] = img
        j += 1
    return img2


def load_train_scenes(k, randomlist):
    minibatch_class1 = readlist('./dataset/scenes/mix_train/0', k, randomlist)
    minibatch_class2 = readlist('./dataset/scenes/mix_train/1', k, randomlist)
    minibatch_class3 = readlist('./dataset/scenes/mix_train/2', k, randomlist)
    minibatch_class4 = readlist('./dataset/scenes/mix_train/3', k, randomlist)
    minibatch_class5 = readlist('./dataset/scenes/mix_train/4', k, randomlist)

    minibatch = np.concatenate((minibatch_class1, minibatch_class2,
                                minibatch_class3, minibatch_class4,
                                minibatch_class5), axis=0)

    minibatch_label = np.zeros((10, 5))

    for i in range(2):
        minibatch_label[i, 0] = 1
    for i in range(2, 4):
        minibatch_label[i, 1] = 1
    for i in range(4, 6):
        minibatch_label[i, 2] = 1
    for i in range(6, 8):
        minibatch_label[i, 3] = 1
    for i in range(8, 10):
        minibatch_label[i, 4] = 1

    random = np.random.permutation(10)
    minibatch = minibatch[random]
    minibatch_label = minibatch_label[random]

    return minibatch, minibatch_label


def load_vali_scenes(k):
    minibatch_class1 = read('./dataset/scenes/veri/church', k)
    minibatch_class2 = read('./dataset/scenes/veri/desert', k)
    minibatch_class3 = read('./dataset/scenes/veri/ice', k)
    minibatch_class4 = read('./dataset/scenes/veri/lawn', k)
    minibatch_class5 = read('./dataset/scenes/veri/river', k)

    minibatch = np.concatenate((minibatch_class1, minibatch_class2,
                                minibatch_class3, minibatch_class4,
                                minibatch_class5), axis=0)
    minibatch_label = np.zeros((10, 5))

    for i in range(2):
        minibatch_label[i, 0] = 1
    for i in range(2, 4):
        minibatch_label[i, 1] = 1
    for i in range(4, 6):
        minibatch_label[i, 2] = 1
    for i in range(6, 8):
        minibatch_label[i, 3] = 1
    for i in range(8, 10):
        minibatch_label[i, 4] = 1

    randoms = np.random.permutation(10)
    minibatch = minibatch[randoms]
    minibatch_label = minibatch_label[randoms]

    return minibatch, minibatch_label


def load_train_car(k, randomlist):
    minibatch_class1 = readlist('./dataset/Car/mix_train/0', k, randomlist)
    minibatch_class2 = readlist('./dataset/Car/mix_train/1', k, randomlist)
    minibatch_class3 = readlist('./dataset/Car/mix_train/2', k, randomlist)
    minibatch_class4 = readlist('./dataset/Car/mix_train/3', k, randomlist)
    minibatch_class5 = readlist('./dataset/Car/mix_train/4', k, randomlist)
    minibatch_class6 = readlist('./dataset/Car/mix_train/5', k, randomlist)
    minibatch_class7 = readlist('./dataset/Car/mix_train/6', k, randomlist)
    minibatch_class8 = readlist('./dataset/Car/mix_train/7', k, randomlist)
    minibatch_class9 = readlist('./dataset/Car/mix_train/8', k, randomlist)
    minibatch_class10 = readlist('./dataset/Car/mix_train/9', k, randomlist)

    minibatch = np.concatenate((minibatch_class1, minibatch_class2,
                                minibatch_class3, minibatch_class4,
                                minibatch_class5, minibatch_class6,
                                minibatch_class7, minibatch_class8,
                                minibatch_class9, minibatch_class10), axis=0)

    minibatch_label = np.zeros((20, 10))

    for i in range(2):
        minibatch_label[i, 0] = 1
    for i in range(2, 4):
        minibatch_label[i, 1] = 1
    for i in range(4, 6):
        minibatch_label[i, 2] = 1
    for i in range(6, 8):
        minibatch_label[i, 3] = 1
    for i in range(8, 10):
        minibatch_label[i, 4] = 1
    for i in range(10, 12):
        minibatch_label[i, 5] = 1
    for i in range(12, 14):
        minibatch_label[i, 6] = 1
    for i in range(14, 16):
        minibatch_label[i, 7] = 1
    for i in range(16, 18):
        minibatch_label[i, 8] = 1
    for i in range(18, 20):
        minibatch_label[i, 9] = 1

    random = np.random.permutation(20)
    minibatch = minibatch[random]
    minibatch_label = minibatch_label[random]

    return minibatch, minibatch_label


def load_vali_car(k):
    minibatch_class1 = read('./dataset/Car/val/0bus', k)
    minibatch_class2 = read('./dataset/Car/val/1family sedan', k)
    minibatch_class3 = read('./dataset/Car/val/2fire engine', k)
    minibatch_class4 = read('./dataset/Car/val/3heavy truck', k)
    minibatch_class5 = read('./dataset/Car/val/4jeep', k)
    minibatch_class6 = read('./dataset/Car/val/5minibus', k)
    minibatch_class7 = read('./dataset/Car/val/6racing car', k)
    minibatch_class8 = read('./dataset/Car/val/7SUV', k)
    minibatch_class9 = read('./dataset/Car/val/8taxi', k)
    minibatch_class10 = read('./dataset/Car/val/9truck', k)

    minibatch = np.concatenate((minibatch_class1, minibatch_class2,
                                minibatch_class3, minibatch_class4,
                                minibatch_class5, minibatch_class6,
                                minibatch_class7, minibatch_class8,
                                minibatch_class9, minibatch_class10), axis=0)

    minibatch_label = np.zeros((20, 10))

    for i in range(2):
        minibatch_label[i, 0] = 1
    for i in range(2, 4):
        minibatch_label[i, 1] = 1
    for i in range(4, 6):
        minibatch_label[i, 2] = 1
    for i in range(6, 8):
        minibatch_label[i, 3] = 1
    for i in range(8, 10):
        minibatch_label[i, 4] = 1
    for i in range(10, 12):
        minibatch_label[i, 5] = 1
    for i in range(12, 14):
        minibatch_label[i, 6] = 1
    for i in range(14, 16):
        minibatch_label[i, 7] = 1
    for i in range(16, 18):
        minibatch_label[i, 8] = 1
    for i in range(18, 20):
        minibatch_label[i, 9] = 1

    random = np.random.permutation(20)
    minibatch = minibatch[random]
    minibatch_label = minibatch_label[random]

    return minibatch, minibatch_label


def load_new(k):
    minibatch_class1 = read('./dataset/Car/class/0', k)
    minibatch_class2 = read('./dataset/Car/class/1', k)
    minibatch_class3 = read('./dataset/Car/class/2', k)
    minibatch_class4 = read('./dataset/Car/class/3', k)
    minibatch_class5 = read('./dataset/Car/class/4', k)
    minibatch_class6 = read('./dataset/Car/class/5', k)
    minibatch_class7 = read('./dataset/Car/class/6', k)
    minibatch_class8 = read('./dataset/Car/class/7', k)
    minibatch_class9 = read('./dataset/Car/class/8', k)
    minibatch_class0 = read('./dataset/Car/class/9', k)

    minibatch = np.concatenate((minibatch_class1, minibatch_class2,
                                minibatch_class3, minibatch_class4,
                                minibatch_class5, minibatch_class6,
                                minibatch_class7, minibatch_class8,
                                minibatch_class9, minibatch_class0,), axis=0)

    minibatch_label = np.zeros((20, 10))

    for i in range(2):
        minibatch_label[i, 0] = 1
    for i in range(2, 4):
        minibatch_label[i, 1] = 1
    for i in range(4, 6):
        minibatch_label[i, 2] = 1
    for i in range(6, 8):
        minibatch_label[i, 3] = 1
    for i in range(8, 10):
        minibatch_label[i, 4] = 1
    for i in range(10, 12):
        minibatch_label[i, 5] = 1
    for i in range(12, 14):
        minibatch_label[i, 6] = 1
    for i in range(14, 16):
        minibatch_label[i, 7] = 1
    for i in range(16, 18):
        minibatch_label[i, 8] = 1
    for i in range(18, 20):
        minibatch_label[i, 9] = 1

    random = np.random.permutation(20)
    minibatch = minibatch[random]
    minibatch_label = minibatch_label[random]

    return minibatch, minibatch_label
