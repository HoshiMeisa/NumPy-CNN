import numpy as np
import os


def save_model(parameters, save_as):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i].data
    np.savez(save_as, **dic)


def load_model(parameters, file):
    params = np.load(file)
    for i in range(32):
        parameters[i].data = params[str(i)]
