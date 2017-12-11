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
from datamake import AverageDataMake
import sys

W = 0.0000

class Learner:

	def __init__(self, name):
		self.name = name
	
	def run(self, data, batch, epoch, rate, epoch_done):
		print self.name

		train, test = data.make()
		chain = RNN(1, 2)
		model = L.Classifier(chain)
		
		if epoch_done > 0:
			serializers.load_npz(self.name, model.predictor)
		
		optimizer = optimizers.Adam(alpha = rate)
		optimizer.setup(model)
		optimizer.add_hook(chainer.optimizer.WeightDecay(W))
	
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

		run_info = {
			'batch': batch,
			'epoch': epoch,
			'epoch_done': epoch_done + epoch,
			'weight_decay': W,
			'learning_rate': rate,
		}
	
		data_info = data.info()
		chain_info = chain.info()
		
		f = open('result/{0}/info.txt'.format(self.name), 'w')
		
		f.write('run_info:\n')
		for key, value in run_info.items():
			f.write('    {0}: {1}\n'.format(key, value))
		
		f.write('data_info:\n')
		for key, value in data_info.items():
			f.write('    {0}: {1}\n'.format(key, value))
		
		f.write('chain_info:\n')
		for key, value in chain_info.items():
			f.write('    {0}: {1}\n'.format(key, value))
		
		f.close()
				
		
BATCH = 1000
EPOCH = 100
RATE = 0.001

if __name__ == '__main__':
	data = AverageDataMake()
	learner = Learner(sys.argv[1])
	learner.run(data, BATCH, EPOCH, RATE, 0)
