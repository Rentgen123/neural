from tools import *
from datagen import *


nw = NeuralNetwork(alfa=1, etha=0.2)
nw.addlayer(1)
nw.addlayer(5, bias=True)
nw.addlayer(4)
nw.addlayer(1)
'''
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([[0],[1],[1],[0]])
'''

[train_x, train_y, chek_x, chek_y] = get_test(func='sqx', train_set = 500, chek_set = 10)

nw.train(train_x, train_y, epoch_number=1000, show=True, banch=True, banch_size=25)
nw.predict(chek_x, chek_y)


'''print(nw)
nw.show_weidths()
nw.show_values()'''
