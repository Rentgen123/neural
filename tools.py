import numpy as np
import numpy.random as rand

class NeuralNetwork():

    def __init__(self, alfa = 1, etha = 0.3):
        self.layerslist = []
        self.layersnumber = 0
        self.compiled = False
        self.alfa = alfa
        self.answer = None
        self.error = None
        self.etha = etha

    def __str__(self):
        rep = ''
        rep += 'Networks layers number:\t' + str(self.layersnumber) + '\n'
        rep += '\n'
        for l in self.layerslist:
            rep += l.repr()
            rep += '\n'
        return rep

    def addlayer(self, neurons_amount=1, bias=False):
        if self.layersnumber == 0:
            layer = Layer(self.layersnumber,
             neurons_amount=neurons_amount, input=True, bias=bias)
        else:
            layer = Layer(self.layersnumber,
             neurons_amount=neurons_amount, bias=bias)
        self.layerslist.append(layer)
        self.layersnumber += 1

    def compile(self):
        self.layerslist[-1].output = True
        for i in range(self.layersnumber-1):
            self.layerslist[i].values = np.ones(self.layerslist[i].neurons_amount)
            if self.layerslist[i+1].bias:
                self.layerslist[i].w = np.random.sample((self.layerslist[i+1].neurons_amount - 1,
                                                        self.layerslist[i].neurons_amount))
                self.layerslist[i].dw = np.zeros_like(self.layerslist[i].w)
            else:
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

            if self.layerslist[i].bias:
                self.layerslist[i].values[0:-1] = np.dot(self.layerslist[i-1].w,
                                                    self.layerslist[i-1].values)

                for j in range(len(self.layerslist[i].values) - 1):
                    self.layerslist[i].values[j] = self.fact(self.layerslist[i].values[j])
            else:
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
                for j in range(len(self.layerslist[i].values)):
                    self.layerslist[i].delta[j] = -2*self.alfa*self.layerslist[i].values[j]*(1 - self.layerslist[i].values[j])*(outputs[j] - self.answer[j])
            else:
                summ = 0
                if not self.layerslist[i+1].bias:
                    summ = np.dot(self.layerslist[i].w.T,
                                    self.layerslist[i+1].delta)
                else:
                    summ = np.dot(self.layerslist[i].w.T,
                                    self.layerslist[i+1].delta[0:-1])
                for j in range(len(self.layerslist[i].values)):
                    self.layerslist[i].delta[j] = 2*self.alfa*self.layerslist[i].values[j]*(1 - self.layerslist[i].values[j])*summ[j]

        for l in range(0,self.layersnumber-1):
            ci = 0
            if not self.layerslist[l+1].bias:
                for i in np.array(self.layerslist[l+1].delta, ndmin=1):
                    cj = 0
                    for j in self.layerslist[l].values:
                        self.layerslist[l].dw[ci,cj] = -self.etha*i*j
                        cj += 1
                    ci += 1
            else:
                for i in np.array(self.layerslist[l+1].delta[0:-1], ndmin=1):
                    cj = 0
                    for j in self.layerslist[l].values:
                        self.layerslist[l].dw[ci,cj] = -self.etha*i*j
                        cj += 1
                    ci += 1
            self.layerslist[l].w += self.layerslist[l].dw


    def train(self, inputs, outputs, epoch_number, show=False, banch = False, banch_size = 10):

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
        if not banch:
            for epoch in range(1, epoch_number + 1):
                if show:    print('Epoch number: ' + str(epoch) + '\n')
                diff = 0
                for rowin, rowout in zip(inputs, outputs):
                    self.forward(rowin)
                    diff = self.get_error(rowout)
                    if show:    print('Error: ' + str(diff) + '\n')
                    self.backward(rowout)
                if show:    print('Error: ' + str(diff) + '\n')
        else:
            for epoch in range(1, epoch_number + 1):
                counter = 0
                if show:    print('Epoch number: ' + str(epoch) + '\n')
                diff = 0
                for rowin, rowout in zip(inputs[0+counter*banch_size:banch_size+counter*banch_size,:],
                 outputs[0+counter*banch_size:banch_size+counter*banch_size,:]):
                    counter += 1
                    self.forward(rowin)
                    diff = self.get_error(rowout)
                    if show:    print('Error: ' + str(diff) + '\n')
                    self.backward(rowout)
                if show:    print('Error: ' + str(diff) + '\n')


    def predict(self, inputs, outputs):
        diff = 0
        for rowin, rowout in zip(inputs, outputs):
            self.forward(rowin)
            diff += self.get_error(rowout)
            print(str(rowin) + '\t' + str(self.answer) + '\t' + str(rowout))
        print('Average error: ' + str(diff/inputs.shape[0]))


class Layer():

    def __init__(self, number=0, neurons_amount=1,
                input=False, output=False, bias=False):
        self.input = input
        self.output = output
        self.number = number
        self.w = None
        self.values = None
        self.delta = None
        self.dw = None
        self.bias = bias
        if self.output:
            self.bias = False
        if self.bias:
            self.neurons_amount = neurons_amount + 1
        else:
            self.neurons_amount = neurons_amount

    def __str__(self):
        rep = ''
        rep += 'Layer number:\t' + str(self.number) + '\n'
        rep += 'Neurons number:\t' + str(self.neurons_amount) + '\n'
        if self.bias:
            rep += 'Bias on'
        return rep

    def repr(self):
        rep = ''
        rep += 'Layer number:\t' + str(self.number) + '\n'
        rep += 'Neurons number:\t' + str(self.neurons_amount) + '\n'
        if self.bias:
            rep += 'Bias on' + '\n'
        if self.input:
            rep += 'Input layer' + '\n'
        elif self.output:
            rep += 'Output layer' + '\n'
        return rep
