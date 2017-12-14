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
D = 9

PERIOD = 120
AFTER = 0
INTERVAL = 10

class DataMake:
	
	def info(self):
		return {
			'range': R,
			'period': PERIOD,
			'after': AFTER,
			'interval': INTERVAL,
			'train': D * 100000,
			'test': 38820,
		}

	def dif(self, array):
		dif = []
		for i in range(len(array) - 1):
			dif.append(array[i + 1] - array[i])
	
		return dif

	def normalize(self, array):
		return [(x + R) / (R * 2) for x in array]
	
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
		L = len(array) - (PERIOD + 1 + AFTER + INTERVAL)
		
		if N is None:
			N = L
	
		while len(dataset) < N:
			if shuffle:
				pos = random.randint(0, L)
			else:
				pos = (pos + L - 1) % L
		
			average = [e[0] for e in array[pos : pos + PERIOD + 1 + AFTER + INTERVAL]]
			x = self.normalize(self.dif(average[: PERIOD + 1]))			
			t = self.evaluate(average[-(INTERVAL + 1) :])

			x = np.array([[e] for e in x], dtype='float32')
			t = np.array(t, dtype='int32')
			dataset.append((x, t))
	
		return dataset

	def make(self):
		train, test = self.load()
		train = self.arrange(train, None, 0, False)
		test = self.arrange(test, None, 0, False)
		
		return (train, test)		
		

class AverageDataMake(DataMake):
	def evaluate(self, values):
		t = 0
		if sum(values[1:]) / len(values[1:]) > values[0]:
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
			y_var += (y - y_average) * (y - y_average)
			r_var += (x - x_average) * (y - y_average)
	
		x_sd = math.sqrt(x_var)
		y_sd = math.sqrt(y_var)
	
		if y_sd == 0: return 0
		return r_var / (x_sd * y_sd)

	def array_coefficient(self, array):
		points = []
	
		for i in range(len(array)):
			points.append((i, array[i]))
	
		return self.coefficient(points)

	def evaluate(self, values):
			coef = self.array_coefficient(values)

			t = 0
			if coef > 0:
				t = 1
			
			return t

