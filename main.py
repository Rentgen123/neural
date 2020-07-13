from tools import *


nw = NeuralNetwork()
nw.addlayer(2)
nw.addlayer(3)
nw.addlayer(1)
nw.show_weidths()
nw.compile()
print(nw)
nw.show_weidths()
