import numpy as np
import numpy.random as rand

class NeuralNetwork():

    def __init__(self):
        self.layerslist = []
        self.layersnumber = 0
        self.compiled = False

    def __str__(self):
        rep = ''
        rep += 'Networks layers number:\t' + str(self.layersnumber) + '\n'
        rep += '\n'
        for l in self.layerslist:
            rep += l.repr()
            rep += '\n'
        return rep

    def addlayer(self, neurons_amount=1):
        if self.layersnumber == 0:
            layer = Layer(self.layersnumber, neurons_amount, input=True)
        else:
            layer = Layer(self.layersnumber, neurons_amount)
        self.layerslist.append(layer)
        self.layersnumber += 1

    def compile(self):
        self.layerslist[-1].output = True
        for i in range(self.layersnumber-1):
            self.layerslist[i].w = np.random.sample((self.layerslist[i+1].neurons_amount,
             self.layerslist[i].neurons_amount))
        self.compiled = True

    def show_weidths(self):
        if self.compiled:
            for i in self.layerslist:
                print(i.w)
                print('\n')
        else:
            print('Not compiled yet!')

class Layer():

    def __init__(self, number=0, neurons_amount=1,
                input=False, output=False):
        self.input = input
        self.output = output
        self.number = number
        self.neurons_amount = neurons_amount
        self.w = None

    def __str__(self):
        rep = ''
        rep += 'Layer number:\t' + str(self.number) + '\n'
        rep += 'Neurons number:\t' + str(self.neurons_amount) + '\n'
        return rep

    def repr(self):
        rep = ''
        rep += 'Layer number:\t' + str(self.number) + '\n'
        rep += 'Neurons number:\t' + str(self.neurons_amount) + '\n'
        if self.input:
            rep += 'Input layer' + '\n'
        elif self.output:
            rep += 'Output layer' + '\n'
        return rep
