import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


class mynet(object):

	def linear(x):
		return x
	
	def __init__(self, ninput, noutput, activation=linear):
		self.nlayers = 1
		self.ninput = ninput
		self.layers = [noutput]
		self.weights = [np.random.randn(ninput, noutput)]
		self.biases = [np.random.randn(noutput)]
		self.activation = activation

	def add_layer(self, nnodes, position=None):
		position = self.nlayers-1 if position is None else position
		position = self.nlayers + position if position < 0 else position
		self.nlayers += 1
		self.layers.insert(position, nnodes)
		self.weights.insert(position, np.random.randn(self.layers[position-1] if position > 0 else self.ninput, nnodes))
		self.weights[position+1] = np.random.randn(nnodes, self.layers[position+1])
		self.biases.insert(position, np.random.randn(nnodes))

	def print_layers(self):
		print(self.ninput, end=' ')
		for i in range(self.nlayers):
			print(np.shape(self.weights[i]), np.shape(self.biases[i]), self.layers[i], end=' ')
		print('')
	
	def evaluate(self, ineval):
		if np.shape(ineval) != np.array(self.ninput):
			err_msg = 'Input must be same size as the input layer (=' + str(self.ninput) + ')'
			raise ValueError(err_msg)
		else:
			current = np.copy(ineval)
			for i in range(self.nlayers):
				print(np.shape(current))
				current = self.activation(np.matmul(current, self.weights[i]) + self.biases[i])
			return current


def lincut(x):
	return x.clip(min=0)


def sigmoid(x):
	return 1/(1 + np.exp(-x))


np.random.seed(0)
test = mynet(8, 3, sigmoid)
test.print_layers()
test.add_layer(5, position=0)
test.print_layers()
test.add_layer(3)
test.print_layers()
test.add_layer(16, position=-2)
test.print_layers()
test.add_layer(9, position=2)
test.print_layers()
out = test.evaluate(np.random.randn(8))
print(out)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(len(x_train[:, 0]), -1)
x_test = x_test.reshape(len(x_test[:, 0]), -1)