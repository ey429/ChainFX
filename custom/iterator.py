class RNNIterator():

	def __init__(self, dataset, batch_size, repeat=True):
		self.dataset = dataset
		self.batch_size = batch_size
		self.epoch = 0
		self.iteration = 0
		self.repeat = repeat

		length = len(dataset)
		self.offsets = [i * length / batch_size for i in range(batch_size)]
	
	def reset(self):
		self.epoch = 0
		self.iteration = 0

	def next(self):
		length = len(self.dataset)

		if not self.repeat and self.iteration * self.batch_size >= length:
			return None
			
		batch = self.get_batch()
		self.iteration += 1

		self.epoch = self.iteration * self.batch_size / length
		
		x = [data[0] for data in batch]
		t = [data[1] for data in batch]

		return (x, t)

	def get_batch(self):
		return [self.dataset[(offset + self.iteration) % len(self.dataset)] for offset in self.offsets]

