#!/usr/bin/env python
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers
from rnn import RNN
from datamake import *
from iterator import RNNIterator
import sys

GPU = 0
PERIOD = 50
BATCH = 100
EPOCH = 10
RATE = 0.001
CLIP = 3.0
W = 0.0001

class Classifier(chainer.Chain):
	def __init__(self, predictor):
		super(Classifier, self).__init__(
			predictor = predictor
		)

	def __call__(self, x, t):
		y = self.predictor(x)
		loss = F.softmax_cross_entropy(y, t)
		accuracy = F.accuracy(y, t)
		return loss, accuracy


class Learner():

	def __init__(self, name):
		self.name = name

	def evaluate(self, model, test_iter):
		test_iter.reset()
		
		evaluator = model.copy()
		evaluator.predictor.reset_state()
		evaluator.predictor.train = False
		
		sum_loss = 0
		sum_accuracy = 0
		count = 0
		
		while True:
			batch = test_iter.next()
			if batch is None:
				break
			
			x, t = batch
			x = Variable(np.array(x, dtype='float32'))
			t = Variable(np.array(t, dtype='int32'))
			loss, accuracy = evaluator(x, t)
			sum_loss += loss.data
			sum_accuracy += accuracy.data
			count += 1

		return sum_loss / count, sum_accuracy / count

	def run(self, chain, data, epoch_done):
		print self.name
		train, test = data.make()
		model = Classifier(chain)
		
		exe_log = []

#		chainer.cuda.get_device_from_id(GPU).use()
#		model.to_gpu()
		
		optimizer = chainer.optimizers.Adam(alpha = RATE)

		if epoch_done > 0:
			serializers.load_npz(self.name + '.model', model.predictor)
			serializers.load_npz(self.name + '.optimizer', optimizer)
		else:
			optimizer.setup(model)

		optimizer.add_hook(chainer.optimizer.GradientClipping(CLIP))
		optimizer.add_hook(chainer.optimizer.WeightDecay(W))

		train_iter = RNNIterator(train, BATCH, True)
		test_iter = RNNIterator(test, 1, False)
		
		sum_loss = 0
		sum_accuracy = 0
		count = 0
		iteration = 0
		
		while train_iter.epoch < EPOCH:
			epoch = train_iter.epoch
			loss = 0

			for p in range(PERIOD):
				x, t = train_iter.next()
				x = Variable(np.array(x, dtype='float32'))
				t = Variable(np.array(t, dtype='int32'))
				loss_, accuracy = optimizer.target(x, t)
				loss += loss_
				sum_accuracy += accuracy.data
				count += 1

			sum_loss += loss.data
			optimizer.target.cleargrads()
			loss.backward()
			loss.unchain_backward()
			optimizer.update()
			
			iteration += 1

			if iteration % 5 == 0:
				print 'iteration: {}'.format(iteration)
				print '  training loss: {}'.format(sum_loss / count)
				print '  training accuracy: {}'.format(sum_accuracy / count)
				print ''
				sum_loss = 0
				sum_accuracy = 0
				count = 0

			if train_iter.epoch > epoch:
				loss, accuracy = self.evaluate(model, test_iter)
				
				logs = [
					'epoch: {}'.format(train_iter.epoch),
					'  validation loss: {}'.format(loss),
					'  validation accuracy: {}'.format(accuracy),
					''
				]
				
				print '\n'.join(logs)
				exe_log.extend(logs)

		print 'save the model'
		serializers.save_npz(self.name + '.model', model)
		print 'save the optimizer'
		serializers.save_npz(self.name + '.optimizer', optimizer)
		
		run_info = {
			'period': PERIOD,
			'batch': BATCH,
			'epoch': EPOCH,
			'epoch_done': epoch_done + EPOCH,
			'learning_rate': RATE,
			'weight_decay': W,
			'gradient_clip': CLIP,
		}
	
		data_info = data.info()
		chain_info = chain.info()
		
		f = open('result/log_{}.txt'.format(self.name), 'w')
		
		f.write('run_info:\n')
		for key, value in run_info.items():
			f.write('  {}: {}\n'.format(key, value))
		
		f.write('\ndata_info:\n')
		for key, value in data_info.items():
			f.write('  {}: {}\n'.format(key, value))
		
		f.write('\nchain_info:\n')
		for key, value in chain_info.items():
			f.write('  {}: {}\n'.format(key, value))
		
		f.write('\nresult:\n')
		f.write('\n'.join(exe_log))
		
		f.close()


if __name__ == '__main__':
	learner = Learner(sys.argv[1])
	chain = RNN(1, 2)
	chain.reset_state()
	data = TestDataMake()
	learner.run(chain, data, 0)
