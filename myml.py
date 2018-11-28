import numpy as np


class mynet(object):
	
	def __init__(self, ninput, noutput, activation='linear'):
		self.nlayers = 1
		self.ninput = ninput
		self.layers = [noutput]
		self.weights = [np.random.randn(ninput, noutput)]
		self.biases = [np.random.randn(noutput)]

	def add_layer(self, nnodes, position=None):
		if position is None:
			position = self.nlayers-1
		if position < 0:
			position = self.nlayers + position
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
				current = np.matmul(current, self.weights[i]) + self.biases[i]
			return current


test = mynet(8, 3)
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
