import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

D1 = 0.2
D2 = 0.5
n_hidden = 300

class RNN(chainer.Chain):

	def __init__(self, n_input, n_output):
		self.train = True
		super(RNN, self).__init__(
			lstm1 = L.LSTM(n_input, n_hidden),
			fc1 = L.Linear(n_hidden, n_hidden),
			fc2 = L.Linear(n_hidden, n_hidden),
			fc3 = L.Linear(n_hidden, n_output))
		
	def __call__(self, x):
		h = F.relu(self.lstm1(x))
#		h = F.dropout(h, D1, self.train)
		
		h = F.relu(self.fc1(h))
#		h = F.dropout(h, D2, self.train)
		
		h = F.relu(self.fc2(h))
#		h = F.dropout(h, D2, self.train)
		
		y = self.fc3(h)
		return y
	
	def reset_state(self):
		self.lstm1.reset_state()
	
	def info(self):
		return {
			'dropout1': D1,
			'dropout2': D2,
			'n_hidden': n_hidden,
		}
