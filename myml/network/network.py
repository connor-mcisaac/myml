import numpy as np
import time


class mynet(object):


	def sigmoid(x, dif=False):
		if dif is False:
			return 1/(1 + np.exp(-x))
		else:
			return np.exp(-x)/((1 + np.exp(-x))**2)


	def softmax(x, dif=False):
		if dif is False:
			return np.exp(x)/np.sum(np.exp(x))
		else:
			y = np.exp(x)/np.sum(np.exp(x))
			y = y[:, np.newaxis]
			matrixd = np.zeros((len(y), len(y)))
			for i in range(len(y)):
				matrixd[i, i] = y[i]
			matrix = np.matmul(y, y.T)
			return matrixd - matrix


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
		self.weights = [np.random.randn(noutput, ninput)*0.1]
		self.biases = [np.random.randn(noutput)*0.1]
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
		self.weights.insert(position, np.random.randn(nnodes, self.layers[position-1] if position > 0 else self.ninput)*0.1)
		self.weights[position+1] = np.random.randn(self.layers[position+1], nnodes)*0.1
		self.biases.insert(position, np.random.randn(nnodes)*0.1)


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
		for l, w, b in zip(self.layers, self.weights, self.biases):
			print(np.shape(w), np.shape(b), l, end=' ')
		print('')


	def evaluate(self, ineval):
		if np.shape(ineval) != np.array(self.ninput):
			err_msg = 'Input must be same size as the input layer (=' + str(self.ninput) + ')'
			raise ValueError(err_msg)
		else:
			current = np.matmul(self.weights[0], ineval) + self.biases[0]
			for i in range(1, self.nlayers):
				current = np.matmul(self.weights[i], self.activation(current)) + self.biases[i]
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


	def backprop(self, ineval, expected):
		if np.shape(ineval) != np.array(self.ninput):
			err_msg = 'Input must be same size as the input layer (=' + str(self.ninput) + ')'
			raise ValueError(err_msg)
		else:
			Zs = [np.matmul(self.weights[0], ineval) + self.biases[0]]
			As = [self.activation(Zs[0])]
			for i in range(1, self.nlayers-1):
				Zs.append(np.matmul(self.weights[i], As[-1]) + self.biases[i])
				As.append(self.activation(Zs[-1]))
			Zs.append(np.matmul(self.weights[-1], As[-1]) + self.biases[-1])
			As.append(self.finalf(Zs[-1]))
			deltaC = As[-1] - expected
			deltaBs = [np.matmul(self.finalf(Zs[-1], dif=True), deltaC)]
			deltaWs = [np.matmul(deltaBs[0][:, np.newaxis], np.array(As[-2])[:, np.newaxis].T)]
			for i in range(1, self.nlayers-1):
				deltaBs.insert(0, np.matmul(self.weights[-i].T, deltaBs[0])*self.activation(Zs[-i-1], dif=True))
				deltaWs.insert(0, np.matmul(deltaBs[0][:, np.newaxis], np.array(As[-i-2])[:, np.newaxis].T))
			deltaBs.insert(0, np.matmul(self.weights[1].T, deltaBs[0])*self.activation(Zs[0], dif=True))
			deltaWs.insert(0, np.matmul(deltaBs[0][:, np.newaxis], np.array(ineval)[:, np.newaxis].T))
			return deltaBs, deltaWs


	def mini_batch(self, start, end, eta):
		n = end-start
		comBs = []
		comWs = []
		for i in range(start, end):
			deltaBs, deltaWs = self.backprop(self.traindata[0][i], self.traindata[1][i])
			if i == start:
				for deltaB in deltaBs:
					comBs.append(deltaB/n)
				for deltaW in deltaWs:
					comWs.append(deltaW/n)
			else:
				for j in range(self.nlayers):
					comBs[j] += deltaBs[j]/n
				for j in range(self.nlayers):
					comWs[j] += deltaWs[j]/n
		for i in range(self.nlayers):
			self.biases[i] += -eta*comBs[i]
		for i in range(self.nlayers):
			self.weights[i] += -eta*comWs[i]


	def training_montage(self, batches, rounds, eta, exams=True):
		trainsize = np.size(self.traindata[0], axis=0)
		mostsize = trainsize//batches
		lastsize = mostsize + trainsize%batches
		if exams is True:
			examsize = np.size(self.testdata[0], axis=0)
			passes = 0
			averageC = 0
			for j in range(examsize):
				output = self.evaluate(self.testdata[0][j])
				averageC += 0.5/examsize*np.sum((self.testdata[1][j] - output)**2)
				if np.argmax(output) == np.argmax(self.testdata[1][j]) and np.max(output) > 0.5:
					passes += 1
			grade = np.round(100*passes/examsize, decimals=2)
			print('Before training achieved a grade of ' + str(grade) + ' with an average cost of ' + str(np.round(averageC, decimals=5)))
		for i in range(rounds):
			print('Starting round ' + str(i+1) + '/' + str(rounds))
			time0 = time.time()
			for j in range(batches-1):
				averageC = self.mini_batch(j*mostsize, (j+1)*mostsize, eta)
			averageC = self.mini_batch((batches-1)*mostsize, (batches-1)*mostsize + lastsize, eta)
			if exams is True:
				passes = 0
				averageC = 0
				for j in range(examsize):
					output = self.evaluate(self.testdata[0][j])
					averageC += 0.5/examsize*np.sum((self.testdata[1][j] - output)**2)
					if np.argmax(output) == np.argmax(self.testdata[1][j]) and np.max(output) > 0.5:
						passes += 1
				grade = np.round(100*passes/examsize, decimals=2)
				announce = ' Achieved a grade of ' + str(grade) + ' with an average cost of ' + str(np.round(averageC, decimals=5))
			else:
				announce = ''
			print('Round ' + str(i+1) + '/' + str(rounds) + ' complete in ' + str(np.round(time.time()-time0, decimals=3)) + ' seconds!' + announce)

