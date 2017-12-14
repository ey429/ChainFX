import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

K = 3

class VGGBlock(chainer.Chain):
	def __init__(self, n_channels, n_convs = 2, n_output = 0):
		w = chainer.initializers.HeNormal()
		super(VGGBlock, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(None, n_channels, K, 1, K / 2, initialW = w)
			self.conv2 = L.Convolution2D(n_channels, n_channels, K, 1, K / 2, initialW = w)
			
			if n_convs == 3:
				self.conv3 = L.Convolution2D(n_channels, n_channels, K, 1, K / 2, initialW = w)
				
			if n_output > 0:
				n_full = 2560
				
				self.fc4 = L.Linear(None, n_full, initialW = w)
				self.fc5 = L.Linear(n_full, n_full, initialW = w)
				self.fc6 = L.Linear(n_full, n_output, initialW = w)
		
		self.n_convs = n_convs
		self.n_output = n_output
	
	def __call__(self, x):
		D1 = 0.2
		D2 = 0.5
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		
		if self.n_convs == 3:
			h = F.relu(self.conv3(h))
		
		h = F.dropout(F.max_pooling_2d(h, 2), D1)
		
		if self.n_output > 0:
			h = F.dropout(F.relu(self.fc4(h)), D2)
			h = F.dropout(F.relu(self.fc5(h)), D2)
			h = self.fc6(h)
		
		return h

class VGG16(chainer.ChainList):
	def __init__(self, n_input, n_output):
		super(VGG16, self).__init__(
			VGGBlock(64), # 128 -> 64
			VGGBlock(128), # 64 -> 32
			VGGBlock(256, 3), # 32 -> 16
			VGGBlock(512, 3), # 16 -> 8
			VGGBlock(512, 3, n_output)) # 8 -> 4 5 * 512 = 2560
	
	def __call__(self, x):
		for f in self.children():
			x = f(x)
		
		if chainer.config.train:
			return x
		else:
			return F.softmax(x)
