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
			matrix = np.matmul(y, y.T)
			matrixd = np.diag(np.diag(np.sqrt(matrix)))
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
			for i, (w, b) in enumerate(zip(self.weights, self.biases)):
				if i == 0:
					current = np.matmul(w, ineval) + b
				else:
					current = np.matmul(w, self.activation(current)) + b
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
		if np.shape(expected) != np.array(self.layers[-1]):
			err_msg = 'Expected output must be same size as the output layer (=' + str(self.layers[-1]) + ')'
			raise ValueError(err_msg)
		else:
			Zs = [np.zeros(np.shape(b)) for b in self.biases]
			As = [np.zeros(np.shape(b)) for b in self.biases]
			for i, (w, b) in enumerate(zip(self.weights, self.biases)):
				if i == 0:
					Zs[i] += np.matmul(w, ineval) + b
				else:
					Zs[i] += np.matmul(w, As[i-1]) + b
				if i != self.nlayers-1:
					As[i] += self.activation(Zs[i])
				else:
					As[i] += self.finalf(Zs[i])
			deltaBs = [np.zeros(np.shape(b)) for b in self.biases]
			deltaWs = [np.zeros(np.shape(w)) for w in self.weights]
			deltaC = As[-1] - expected
			deltaBs[-1] += np.matmul(self.finalf(Zs[-1], dif=True), deltaC)
			deltaWs[-1] += np.matmul(deltaBs[-1][:, np.newaxis], np.array(As[-2])[:, np.newaxis].T)
			for i, (w, z, a) in enumerate(zip(self.weights[2:][::-1], Zs[1:-1][::-1], As[:-2][::-1]), start=2):
				deltaBs[-i] += np.matmul(w.T, deltaBs[-i+1])*self.activation(z, dif=True)
				deltaWs[-i] += np.matmul(deltaBs[-i][:, np.newaxis], a[:, np.newaxis].T)
			deltaBs[0] += np.matmul(self.weights[1].T, deltaBs[1])*self.activation(Zs[0], dif=True)
			deltaWs[0] += np.matmul(deltaBs[0][:, np.newaxis], ineval[:, np.newaxis].T)
			return deltaBs, deltaWs


	def mini_batch(self, start, end, eta):
		n = end-start
		comBs = [np.zeros(np.shape(b)) for b in self.biases]
		comWs = [np.zeros(np.shape(w)) for w in self.weights]
		for i in range(start, end):
			deltaBs, deltaWs = self.backprop(self.traindata[0][i], self.traindata[1][i])
			for j, db in enumerate(deltaBs):
				comBs[j] += db/n
			for j, dw in enumerate(deltaWs):
				comWs[j] += dw/n
		for i, cb in enumerate(comBs):
			self.biases[i] += -eta*cb
		for i, cw in enumerate(comWs):
			self.weights[i] += -eta*cw


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
			for j in range(batches):
				if j != batches-1:
					averageC = self.mini_batch(j*mostsize, (j+1)*mostsize, eta)
				else:
					averageC = self.mini_batch(j*mostsize, j*mostsize + lastsize, eta)
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
			print('Round ' + str(i+1) + '/' + str(rounds) + ' complete in ' + str(int(time.time()-time0)) + ' seconds!' + announce)
			  

