import numpy as np
import numpy.random as rand

class NeuralNetwork():

    def __init__(self):
        self.layerslist = []
        self.layersnumber = 0

    def __str__(self):
        rep = ''
        rep += 'Networks layers number:\t' + str(self.layersnumber) + '\n'
        rep += '\n'
        for l in self.layerslist:
            rep += l.repr()
            rep += '\n'
        return rep

    def addlayer(self, neurons_amount=1):
        layer = Layer(self.layersnumber, neurons_amount)
        self.layerslist.append(layer)
        self.layersnumber += 1

class Layer():

    def __init__(self, number=0, neurons_amount=1):
        self.number = number
        self.neurons_amount = neurons_amount
        #self.w = np.random.sample(neurons_amount)

    def __str__(self):
        rep = ''
        rep += 'Layer number:\t' + str(self.number) + '\n'
        rep += 'Neurons number:\t' + str(self.neurons_amount) + '\n'
        return rep

    def repr(self):
        rep = ''
        rep += 'Layer number:\t' + str(self.number) + '\n'
        rep += 'Neurons number:\t' + str(self.neurons_amount) + '\n'
        return rep
