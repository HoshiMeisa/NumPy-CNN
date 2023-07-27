import numpy as np
from PIL import Image


def read(filepath, k):
    img2 = np.zeros((2, 224, 224, 3))
    for j, i in enumerate(range(k, k + 2)):
        img = Image.open(filepath + "/" + str(i) + ".jpg", mode='r').resize((224, 224))
        img2[j] = img
    return img2


def readlist(filepath, k, randomlist):
    img2 = np.zeros((2, 224, 224, 3))
    for j, i in enumerate(range(k, k + 2)):
        img = Image.open(filepath + "/" + str(randomlist[i]) + ".jpg", mode='r').resize((224, 224))
        img2[j] = img
    return img2


def load_scenes(filepath, k, randomlist):
    minibatch = np.zeros((10, 2, 224, 224, 3))
    minibatch_label = np.zeros((10, 5))
    for i in range(10):
        minibatch[i] = readlist(filepath + str(i), k, randomlist)
        for j in range(2):
            minibatch_label[i, j] = 1
    random = np.random.permutation(10)
    return minibatch[random], minibatch_label[random]


def load_car(filepath, k, randomlist):
    minibatch = np.zeros((20, 2, 224, 224, 3))
    minibatch_label = np.zeros((20, 10))
    for i in range(10):
        minibatch[i] = readlist(filepath + str(i), k, randomlist)
        minibatch[i + 10] = readlist(filepath + "/val/" + str(i), k)
        for j in range(2):
            minibatch_label[i, j] = 1
            minibatch_label[i + 10, j] = 1
    random = np.random.permutation(20)
    return minibatch[random], minibatch_label[random]


def load_train_scenes(k, randomlist):
    filepath = './dataset/scenes/mix_train/'
    return load_scenes(filepath, k, randomlist)


def load_vali_scenes(k, randomlist):
    filepath = './dataset/scenes/veri/'
    return load_scenes(filepath, k, randomlist)


def load_train_car(k, randomlist):
    filepath = './dataset/Car/mix_train/'
    return load_car(filepath, k, randomlist)


def load_vali_car(k, randomlist):
    filepath = './dataset/Car/val/'
    return load_car(filepath, k, randomlist)


def load_new(k, randomlist):
    filepath = './dataset/Car/class/'
    return load_car(filepath, k, randomlist)
