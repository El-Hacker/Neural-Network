"""
Title : Neural Network From scratch, 
Author : EL-HACKER 
Youtube Channel : https://www.youtube.com/channel/UCh_6UHjyPWiyC5IRyOadojA?view_as=subscriber

"""


import numpy
import math
import enum
import copy


class TrainingAlgorithms(enum.Enum):
    cBackPropagationError = 0


class ActivationFunction:
    cBinaryStep = "BinaryStep"  # it's not a derivable function, this is why we don't use it
    cLinear = "Linear"
    cSigmoid = "Sigmoid"

    @staticmethod
    def Sigmoid(X, derivative=None, param=None):
        s = []
        for x in X:
            s.append(1/(1+math.exp(-x)))
        s = numpy.array(s)
        if derivative is None or False:
            return s
        elif derivative is True:
            return (1-s)*s

    @staticmethod
    def BinaryStep(X, derivative=None, threshold=0, magnitude=1):
        s = []
        for x in X:
            if x < threshold:
                s.append(0)
            else:
                s.append(magnitude)

        return numpy.array(s)

    @staticmethod
    def Linear(X, derivative=None, slot=1):
        s = []
        for x in X:
            s.append(x*slot)
        s = numpy.array(s)
        if derivative is None or False:
            return s
        elif derivative is True:
            ones = numpy.ones(len(X))
            return ones


class Layer:

    def __init__(self, pos, cType):
        self.pos = pos
        self.function = cType
        self.parameters = []

    def setParam(self, *args):
        self.parameters = args

    def calculate(self, x, derivative=None):
        if self.function == ActivationFunction.cBinaryStep:
            return ActivationFunction.BinaryStep(x, derivative, *self.parameters)
        elif self.function == ActivationFunction.cLinear:
            return ActivationFunction.Linear(x, derivative, *self.parameters)
        elif self.function == ActivationFunction.cSigmoid:
            return ActivationFunction.Sigmoid(x, derivative, *self.parameters)


class NeuralNetwork:

    def __init__(self):
        self.HiddenLayers = []  # array of arrays
        self.Layers = []
        self._architectureEstablished = False
        self.architecture = None
        self.weights = []
        self.thresholds = []
        self.hiddenlayerParam = [ActivationFunction.cLinear, 1]
        self.outputLayerParam = [ActivationFunction.cBinaryStep, 0.5]
        self.nextWeights = []
        self.nextBias = []

    def setParam(self, hiddenlayerParam, outputlayerParam):
        self.hiddenlayerParam = hiddenlayerParam
        self.outputLayerParam = outputlayerParam

    def defineNeuralNet(self, weights, thresholds, architecture):
        self.weights = weights
        self.thresholds = thresholds
        self.build(architecture=architecture, fromOutside=True)

    def _initiate_network(self):
        self.weights = []
        if self._architectureEstablished:
            n = len(self.architecture)
            for i in range(n-1):
                firstLayer = self.architecture[i]
                secondLayer = self.architecture[i+1]
                self.weights.append(numpy.random.uniform(-1,1,(firstLayer, secondLayer)))  # weight Matrices w1, w2, w3..
                self.nextWeights.append(numpy.random.uniform(-1,1,(firstLayer, secondLayer)))  # weight Matrices w1, w2, w3..
                self.thresholds.append(numpy.random.uniform(-1,1,(secondLayer,)))
                self.nextBias.append(numpy.random.uniform(-1,1,(secondLayer,)))

    def build(self, architecture, fromOutside=None):

        if not self._architectureEstablished:
            self._architectureEstablished = True
            self.architecture = architecture
            self.Layers = []
            self.HiddenLayers = []
            if fromOutside is None:
                self._initiate_network()
            for index, layer in enumerate(architecture):
                if index == 0:
                    inputLayer = Layer(index, ActivationFunction.cLinear)
                    inputLayer.setParam(1)
                    self.Layers.append(inputLayer)
                elif index == len(architecture) - 1:
                    outputLayer = Layer(index, self.outputLayerParam[0])
                    outputLayer.setParam(*self.outputLayerParam[1:])
                    self.Layers.append(outputLayer)
                else:
                    layer = Layer(index, self.hiddenlayerParam[0])
                    layer.setParam(*self.hiddenlayerParam[1:])
                    self.HiddenLayers.append(layer)
                    self.Layers.append(layer)

    def predict(self, v_features, layer=None):
        x = numpy.array(v_features)
        param = layer
        if layer is None:
            layer = len(self.architecture) - 1
        elif layer == 0:
            return x, None
        for i in range(layer):
            y = self.weights[i].transpose().dot(x) + self.thresholds[i]
            raw_y = y
            y = self.Layers[i+1].calculate(y)
            x = y
        if param is None:
            return y
        else:
            return y, raw_y

    def train(self, features, labels, algorithm=TrainingAlgorithms.cBackPropagationError, a=0.5
              ,  epochs=200, error=1e-5):

        if algorithm == TrainingAlgorithms.cBackPropagationError:
            error_reached = False
            # doing the process for a specific number of epochs
            for epoch in range(epochs):
                for feature, desired_y in zip(features, labels):  # taking vectors from the Data, each sample is a line!
                    output_layer = len(self.architecture) - 1  # to match with predict function logic!

                    predicted_y, raw_y = self.predict(feature, output_layer)
                    predicted_h1, _ = self.predict(feature, output_layer - 1)
                    e = desired_y - predicted_y
                    e = numpy.array(e)
                    E = sum(e ** 2)
                    derivative = self.Layers[output_layer].calculate(raw_y, True)
                    delta = -(e * derivative)
                    delta_weights = []
                    delta_bias = - a * delta
                    for hi in predicted_h1:
                        delta_wij = - a * hi * delta
                        delta_weights.append(delta_wij)

                    self.nextWeights[output_layer - 1] += delta_weights
                    self.nextBias[output_layer - 1] += delta_bias

                    for i in range(1, output_layer):
                        weights = self.weights[output_layer - i]

                        predicted_y, raw_y = self.predict(feature, output_layer - i)
                        predicted_h1, _ = self.predict(feature, output_layer - i - 1)
                        derivative = self.Layers[output_layer - i].calculate(raw_y, True)
                        n_delta = []
                        for line in weights:
                            s = sum(delta*line)
                            n_delta.append(s)
                        delta = numpy.array(n_delta)*derivative
                        delta_weights = []
                        delta_bias = - a * delta
                        for hi in predicted_h1:
                            delta_wij = - a*hi*delta
                            delta_weights.append(delta_wij)

                        self.nextWeights[output_layer - i - 1] += delta_weights
                        self.nextBias[output_layer - i - 1] += delta_bias

                    self.weights = copy.deepcopy(self.nextWeights)
                    self.thresholds = copy.deepcopy(self.nextBias)
                    #  TODO : it's now working yooo piece of shit
                    if E < error:
                        error_reached = True
                        break
                if error_reached:
                    break

    def validate(self, features, labels):
        # TODO : Accuracy of the MLP
        pass

    def cross_validation(self, features, labels):
        # TODO : to split the Data to two parts validation & training
        pass

    def access_to_weights(self, weights):
        self.weights = weights  # used to train the MLP by Genetic algorithm

    def return_system(self):
        return self.weights, self.thresholds

    def save_model(self):
        # save the weights, thresholds in csv file #TODO later
        pass

    def load_model(self):
        # build the system from csv file #TODO later
        pass


nn1 = NeuralNetwork()
nn1.setParam(hiddenlayerParam=[ActivationFunction.cSigmoid]
             , outputlayerParam=[ActivationFunction.cSigmoid])
nn1.build([2, 6, 7, 9, 4])  # neural network architecture


features = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]

]

# OR, AND, XOR, NXOR
labels = [
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1]
]


nn1.train(features=features, labels=labels, algorithm=TrainingAlgorithms.cBackPropagationError, a=0.15, epochs=10000,
          error=1e-10)

for index, feature in enumerate(features):
    print(nn1.predict(feature) - numpy.array(labels[index]))

