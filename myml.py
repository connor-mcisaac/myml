import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


class mynet(object):

	def sigmoid(x):
		return 1/(1 + np.exp(-x))

	def softmax(x):
		return np.exp(x)/np.sum(np.exp(x))

	def __init__(self, ninput, noutput, activation=sigmoid, finalf=softmax):
		if type(ninput) != int:
			err_msg = 'ninput must be an int'
			raise TypeError(err_msg)
		if type(noutput) != int:
			err_msg = 'noutput must be an int'
			raise TypeError(err_msg)
		if callable(activation) is False:
			err_msg = 'activation must be a callable function'
			raise TypeError(err_msg)
		if callable(finalf) is False:
			err_msg = 'finalf must be a callable function'
			raise TypeError(err_msg)
		self.nlayers = 1
		self.ninput = ninput
		self.layers = [noutput]
		self.weights = [np.random.randn(noutput, ninput)]
		self.biases = [np.random.randn(noutput)]
		self.activation = activation
		self.finalf = finalf
		self.fulldata = None
		self.traindata = None
		self.testdata = None

	def add_single_layer(self, nnodes, position=None):
		if type(nnodes) != int:
			err_msg = 'nnodes must be an int'
			raise TypeError(err_msg)
		position = self.nlayers-1 if position is None else position
		position = self.nlayers + position if position < 0 else position
		if position >= self.nlayers:
			err_msg = 'position must be less than the number of layers'
			raise ValueError(err_msg)
		self.nlayers += 1
		self.layers.insert(position, nnodes)
		self.weights.insert(position, np.random.randn(nnodes, self.layers[position-1] if position > 0 else self.ninput))
		self.weights[position+1] = np.random.randn(self.layers[position+1], nnodes)
		self.biases.insert(position, np.random.randn(nnodes))

	def add_layers(self, nodelist, position=None):
		if type(nodelist) != list:
			err_msg = 'nodelist must be a list'
			raise TypeError(err_msg)
		position = self.nlayers-1 if position is None else position
		position = self.nlayers + position if position < 0 else position
		for nnodes in nodelist:
			self.add_single_layer(nnodes, position=position)
			position += 1

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
				current = self.activation(np.matmul(self.weights[i], current) + self.biases[i])
			return self.finalf(current)

	def give_data(self, dataset, labels, datatype='all', ftrain=0.5):
		if len(dataset[0]) != self.ninput:
			err_msg = 'Dataset must be of the form [#instances, shape of input]'
			raise ValueError(err_msg)
		if len(labels[0]) != self.layers[-1]:
			err_msg = 'Labels must be of the form [#instances, shape of output]'
			raise ValueError(err_msg)
		if datatype not in ['all', 'train', 'test']:
			err_msg = 'Datatype must be "all", "train" or "test"'
			raise ValueError(err_msg)
		if (ftrain < 0) or (ftrain > 1):
			err_msg = 'ftrain must be between 0 and 1'
			raise ValueError(err_msg)
		if self.fulldata is None:
			self.fulldata = [dataset, labels]
		else:
			self.fulldata[0] = np.concatenate((self.fulldata[0], dataset), axis=0)
			self.fulldata[1] = np.concatenate((self.fulldata[1], labels), axis=0)
		if datatype == 'train' and self.traindata == None:
			self.traindata = [dataset, labels]
		elif datatype == 'train' and self.traindata != None:
			self.traindata[0] = np.concatenate((self.traindata[0], dataset), axis=0)
			self.traindata[1] = np.concatenate((self.traindata[1], labels), axis=0)
		if datatype == 'test' and self.testdata == None:
			self.testdata = [dataset, labels]
		elif datatype == 'test' and self.testdata != None:
			self.testdata[0] = np.concatenate((self.testdata[0], dataset), axis=0)
			self.testdata[1] = np.concatenate((self.testdata[1], labels), axis=0)
		if datatype == 'all':
			cut = int(ftrain*np.size(self.fulldata[0], axis=0))
			self.traindata = [self.fulldata[0][:cut], self.fulldata[1][:cut]]
			self.testdata = [self.fulldata[0][cut:], self.fulldata[1][cut:]]

	#def train_net(repetitions=1, minibatch=None, eta=)


np.random.seed(0)
test = mynet(784, 10)
test.print_layers()
test.add_single_layer(5, position=0)
test.print_layers()
test.add_layers([4, 3, 2], position=-2)
test.print_layers()
out = test.evaluate(np.random.randn(784))
print(out)
print(np.sum(out))

(x_train, y_traini),(x_test, y_testi) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(len(x_train[:, 0]), -1)
x_test = x_test.reshape(len(x_test[:, 0]), -1)
y_train = np.zeros((len(y_traini), 10), dtype='float')
for i in range(len(y_traini)):
	y_train[i, y_traini[i]] = 1
y_test = np.zeros((len(y_testi), 10), dtype='float')
for i in range(len(y_testi)):
	y_test[i, y_testi[i]] = 1

test.give_data(x_train, y_train, datatype='train', ftrain=0.4)
test.give_data(x_test, y_test, datatype='test', ftrain=0.6)
print(np.shape(test.traindata[0]), np.shape(test.traindata[1]))
print(np.shape(test.testdata[0]), np.shape(test.testdata[1]))