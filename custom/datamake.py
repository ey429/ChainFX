import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import random, math

R = 0.5
D = 1

AFTER = 1
INTERVAL = 10

class DataMake:
	
	def info(self):
		return {
			'range': R,
			'after': AFTER,
			'interval': INTERVAL,
			'train': self.train_size,
			'test': self.test_size,
		}

	def normalize(self, x):
		return (x + R) / (R * 2)
	
	def load(self):
		train = []
		test = []	
		
		for i in range(D):
			f = open('history/train{0}.txt'.format(i + 1), 'r')
			for line in f:
				train.append(eval(line))
			f.close()
	
		f = open('history/test.txt', 'r')
		for line in f:
			test.append(eval(line))
		f.close()
		
		return (train, test)

	def arrange(self, array, N, NOISE, shuffle):
		dataset = []
		pos = 0
		L = len(array) - (1 + AFTER + INTERVAL)
		
		if N is None:
			N = L
	
		while len(dataset) < N:
			if shuffle:
				pos = random.randint(0, L)
			else:
				pos = (pos + L - 1) % L
		
			average = [e[0] for e in array[pos : pos + 1 + AFTER + INTERVAL]]
			x = self.normalize(average[pos + 1] - average[pos])
			t = self.evaluate(average[pos + 1], average[-INTERVAL :])

			dataset.append(([x], t))
	
		return dataset

	def make(self):
		train, test = self.load()
		train = self.arrange(train, 1000, 0, False)
		test = self.arrange(test, 100, 0, False)
		
		self.train_size = len(train)
		self.test_size  = len(test)
		
		return (train, test)		
		
class TestDataMake(DataMake):
	def sin(self, i):
		return math.sin(i * 0.01)
		
	def make(self):
		train = []
		for pos in range(100000):
			x = self.sin(pos)
			y = self.sin(pos + 1)
			t = (1 if y > x else 0)
			train.append(([x], t))

		pos = random.randint(0, 90000)
		test = train[pos : pos + 10000]

		self.train_size = len(train)
		self.test_size  = len(test)

		return (train, test)		

class AverageDataMake(DataMake):
	def evaluate(self, now, values):
		t = 0
		if sum(values) / len(values) > now:
			t = 1
		
		return t
			
class CoefDataMake(DataMake):
	
	def coefficient(self, points):
		x_average = sum([float(x) for x, y in points]) / len(points)
		y_average = sum([float(y) for x, y in points]) / len(points)
	
		x_var = 0
		y_var = 0
		r_var = 0
	
		for x, y in points:
			x_var += (x - x_average) * (x - x_average)
#			y_var += (y - y_average) * (y - y_average)
			r_var += (x - x_average) * (y - y_average)
	
		x_sd = math.sqrt(x_var)
#		y_sd = math.sqrt(y_var)
	
		return r_var / x_sd

	def array_coefficient(self, array):
		points = []
	
		for i in range(len(array)):
			points.append((i, array[i]))
	
		return self.coefficient(points)

	def evaluate(self, now, values):
			coef = self.array_coefficient([now].extend(values))

			t = 0
			if coef > 0:
				t = 1
			
			return t

