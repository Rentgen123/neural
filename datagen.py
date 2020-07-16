import numpy as np
import random as rnd

def get_test(func = 'sqx', train_set = 100, chek_set = 20):

    if func == 'sqx':
        x_tr = np.array(np.random.sample(train_set), ndmin=2).T
        y_tr = np.power(x_tr, 2)
        x_ch = np.array(np.random.sample(chek_set), ndmin=2).T
        y_ch = np.power(x_ch, 2)

    elif func == 'lin':
        x_tr = np.array(np.random.sample(train_set), ndmin=2).T
        y_tr = 2*x_tr
        x_ch = np.array(np.random.sample(chek_set), ndmin=2).T
        y_ch = 2*x_ch

    return [x_tr, y_tr, x_ch, y_ch]
