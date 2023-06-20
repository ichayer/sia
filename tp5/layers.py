from abc import ABC, abstractmethod
import numpy as np
import copy

from activations import Sigmoid
from optimizer import Adam


class Layer(ABC):
    """
    An abstract class which represents layers
    which make up a neural network.
    """

    @abstractmethod
    def feedforward(self, input):
        """
        Forward propagate the input through the layer.
        """
        pass

    @abstractmethod
    def backward(self, lastGradient):
        """
        Backward propagate the gradient through the layer.
        """
        pass

    @abstractmethod
    def numParameters(self):
        """
        Get the number of pararameters in this layer.
        """
        pass


class Dense(Layer):
    """
    Dense Layer: input and output layer are fully connected.
    """
    def __init__(self, inputDim = 1, outputDim = 1, activation = Sigmoid(), optimizer = Adam()):
        self.inputDim = inputDim
        self.outputDim = outputDim

        # set the activation function
        self.activation = activation

        # set optimizer
        self.weightOptimizer = copy.copy(optimizer)
        self.biasOptimizer = copy.copy(optimizer)

        # randomly initialize the weight and biases
        limit = np.sqrt(6 / (inputDim + outputDim)) # xavier uniform initializer
        self.weight = np.random.uniform(-limit, limit,(outputDim, inputDim))
        self.bias   = np.zeros(outputDim)

        # trainable decides whether weight and biases are trained in backward pass
        self.trainable = True # Layers can also be frozen !

    def feedforward(self, input):
        if input.ndim == 1:
            input = np.squeeze(input).reshape((input.shape[0], self.batchSize))

        self.input = input

        self.z = np.dot(self.weight, self.input) + np.tile(self.bias, (self.input.shape[1], 1)).T
        self.a = self.activation.apply(self.z)
        return self.a

    def backward(self, lastGradient, outputLayer = False, updateParameters = True):
        oldWeight = np.copy(self.weight)
        if not outputLayer:
            lastGradient *= self.activation.derivative(self.z)

        if self.trainable and updateParameters:
            # layer is NOT frozen, weights and biases should be changed
            gradWeight = np.dot(lastGradient, self.input.T)
            gradBias   = np.sum(lastGradient, axis=1)

            # update weights and biases with optimizer
            self.weightOptimizer.optimize(self.weight, gradWeight)
            self.biasOptimizer.optimize(self.bias, gradBias)

        self.gradient = np.dot(oldWeight.T, lastGradient)
        return self.gradient

    def numParameters(self):
        weightShape = self.weight.shape
        return weightShape[0]*weightShape[1] + self.bias.shape[0]

    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
        self.weightOptimizer.setLearningFactor(self.batchSize)
        self.biasOptimizer.setLearningFactor(self.batchSize)

    def __str__(self):
        out  = f"DENSE {self.inputDim} -> {self.outputDim} [{self.activation}]\n"
        tmp  = len(out) * "-" + "\n"
        out  = tmp + out + tmp
        out += f"Total parameters: {self.numParameters()} \n"
        out += f"---> WEIGHTS: {self.weight.shape}\n"
        out += f"---> BIASES: {self.bias.shape}\n"
        out += tmp
        return out