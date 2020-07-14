from tools import *


nw = NeuralNetwork()
nw.addlayer(2)
nw.addlayer(5)
nw.addlayer(4)
nw.addlayer(1)
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([[0],[1],[1],[0]])
nw.train(inputs, outputs, 10000)
for row in inputs:
    nw.forward(row)
    print(str(row) + '\t' + str(nw.answer))
'''print(nw)
nw.show_weidths()
nw.show_values()'''
