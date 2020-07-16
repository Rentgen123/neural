import numpy as np
import numpy.random as rand

class NeuralNetwork():

    def __init__(self):
        self.layerslist = []
        self.layersnumber = 0
        self.compiled = False
        self.alfa = 1
        self.answer = None
        self.error = None
        self.etha = 0.2

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
            self.layerslist[i].dw = np.zeros_like(self.layerslist[i].w)

        self.compiled = True

    def show_weidths(self):
        if self.compiled:
            for i in self.layerslist:
                print(i.w)
                print('\n')
        else:
            print('Not compiled yet!')

    def show_values(self):
        for i in self.layerslist:
            print('Layer\t' + str(i.number) + '\tvalues')
            print(i.values)
            print('\n')

    def forward(self, inputs):

        if inputs.shape == (self.layerslist[0].neurons_amount,):
            self.layerslist[0].values = inputs
        else:
            print('Length of the input values and amount of a first layers neurons must be the same!')
            return None

        for i in range(1, self.layersnumber):
            self.layerslist[i].values = np.dot(self.layerslist[i-1].w,
                                                self.layerslist[i-1].values)
            for j in range(len(self.layerslist[i].values)):
                self.layerslist[i].values[j] = self.fact(self.layerslist[i].values[j])
        self.answer = self.layerslist[-1].values

    def fact(self, x):
        f = 1/(1 + np.exp(-2*self.alfa*x))
        return f

    def dfact(self, x):
        df = 2*self.alfa*(self.fact(x) - np.power(self.fact(x),2))
        return df

    def get_error(self, outputs):
        if self.answer != None:
            diff = np.sum(np.power(self.answer - outputs, 2))/len(self.answer)
        return diff

    def backward(self, outputs):
        for i in range(self.layersnumber-1,0,-1):
            self.layerslist[i].delta = np.zeros_like(self.layerslist[i].values)
            if i == self.layersnumber-1:
                for j in range(0, len(self.layerslist[i].values)):
                    self.layerslist[i].delta[j] = -2*self.alfa*self.layerslist[i].values[j]*(1 - self.layerslist[i].values[j])*(outputs[j] - self.answer[j])
            else:
                summ = 0
                summ = np.dot(self.layerslist[i].w.T,
                                self.layerslist[i+1].delta)
                for j in range(0,len(self.layerslist[i].values)):
                    self.layerslist[i].delta[j] = 2*self.alfa*self.layerslist[i].values[j]*(1 - self.layerslist[i].values[j])*summ[j]

        for l in range(0,self.layersnumber-1):
            ci = 0
            for i in np.array(self.layerslist[l+1].delta, ndmin=1):
                cj = 0
                for j in self.layerslist[l].values:
                    self.layerslist[l].dw[ci,cj] = -self.etha*i*j
                    cj += 1
                ci += 1
            self.layerslist[l].w += self.layerslist[l].dw

    def train(self, inputs, outputs, epoch_number):

        if inputs.shape[0] != outputs.shape[0]:
            print('Количество строк в inputs и outputs должно совпадать')
            return 0
        if inputs.shape[1] != self.layerslist[0].neurons_amount:
            print('Количество столбцов в inputs должно совпадать с количеством нейронов в первом слое')
            return 0
        if outputs.shape[1] != self.layerslist[-1].neurons_amount:
            print('Количество столбцов в outputs должно совпадать с количеством нейронов в выходном слое')
            return 0

        self.compile()
        print('Compiled!')
        for epoch in range(epoch_number):
            print('Epoch number: ' + str(epoch) + '\n')
            diff = 0
            for rowin, rowout in zip(inputs, outputs):
                self.forward(rowin)
                diff = self.get_error(rowout)
                print('Error: ' + str(diff) + '\n')
                self.backward(rowout)
            print('Error: ' + str(diff) + '\n')


class Layer():

    def __init__(self, number=0, neurons_amount=1,
                input=False, output=False):
        self.input = input
        self.output = output
        self.number = number
        self.neurons_amount = neurons_amount
        self.w = None
        self.values = None
        self.delta = None
        self.dw = None

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
