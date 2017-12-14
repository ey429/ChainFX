import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

dropout = [0, 0, 0, 0]
layers = [1, (5, 30), (5, 80), 960, 400, 240]

class CNN(Chain):

	def info(self):
		return {
			'dropout': dropout,
			'layers': layers,
		}
		
	def __init__(self):
		super(CNN, self).__init__(
			conv1 = L.Convolution2D(layers[0], layers[1][1], (1, layers[1][0])),
			conv2 = L.Convolution2D(layers[1][1], layers[2][1], (1, layers[2][0])),
			l1 = L.Linear(layers[3], layers[4]),
			l2 = L.Linear(layers[4], layers[5]),
			l3 = L.Linear(layers[5], 2)
		)
	
	def __call__(self, x):
		h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
		h = F.dropout(h, dropout[0])

		h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
		h = F.dropout(h, dropout[1])

		h = F.relu(self.l1(h))
		h = F.dropout(h, dropout[2])

		h = F.relu(self.l2(h))
		h = F.dropout(h, dropout[3])

		y = self.l3(h)
		return y

