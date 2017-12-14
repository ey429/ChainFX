import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

n_layers = 1
n_hidden = 200
D1 = 0.0
D2 = 0.0

class RNN(chainer.Chain):

	def info(self):
		return {
			'lstm_layers': n_layers,
			'fc_layers': 2,
			'n_hidden': n_hidden,
			'dropout1': D1,
			'dropout2': D2,
		}
		
	def __init__(self, n_input, n_output):
		super(RNN, self).__init__()
		with self.init_scope():
			self.lstm1 = L.NStepLSTM(n_layers, n_input, n_hidden, D1)
			self.fc2 = L.Linear(n_hidden, n_hidden * 2)
			self.fc3 = L.Linear(n_hidden * 2, n_hidden * 2)
			self.fc4 = L.Linear(n_hidden * 2, n_output)
	
	def __call__(self, x):
		in_array = [Variable(np.array(x[i], dtype='float32')) for i in range(len(x))]
		states, cells, out_array = self.lstm1(None, None, in_array)
		h = Variable(np.array([v.data[-1] for v in out_array]))
		h = F.dropout(F.relu(h), D1)

		h = F.dropout(F.relu(self.fc2(h)), D2)
		h = F.dropout(F.relu(self.fc3(h)), D2)
		y = self.fc4(h)
		return y
		
