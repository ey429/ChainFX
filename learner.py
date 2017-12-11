import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from rnn import RNN
from datamake import CoefDataMake
import sys

W = 0.0001

class Learner:

	def __init__(self, name):
		self.name = name
	
	def run(self, train, test, batch, epoch, rate, resume):
		print self.name

		model = L.Classifier(RNN(1, 2))
		
		if resume:
			serializers.load_npz(self.name, model.predictor)
		
		optimizer = optimizers.Adam(alpha = rate)
		optimizer.setup(model)
#		optimizer.add_hook(chainer.optimizer.WeightDecay(W))
	
		train_iter = chainer.iterators.SerialIterator(train, batch)
		test_iter = chainer.iterators.SerialIterator(test, batch, repeat = False, shuffle = False)
	
		updater = training.StandardUpdater(train_iter, optimizer)

		trainer = training.Trainer(updater, (epoch, 'epoch'), out = 'result/' + self.name)
		trainer.extend(extensions.Evaluator(test_iter, model))
		trainer.extend(extensions.LogReport())
		trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
		trainer.extend(extensions.ProgressBar())
		trainer.run()

		serializers.save_npz(self.name, model.predictor)
		
BATCH = 1000
EPOCH = 100
RATE = 0.001

if __name__ == '__main__':
	data = AverageDataMake()
	train, test = data.make()

	learner = Learner(sys.argv[1])
	learner.run(train, test, BATCH, EPOCH, RATE, True)
