import numpy as np
from collections import Sequence

class RingBuffer(Sequence):
	def __init__(self, N, sub_shape, dtype=float, allow_overwrite=True):
		"""
		Create a new ring buffer with the given capacity and element type

		Parameters
		----------
		N: int
			How many entries to buffer in either direction.
		dtype: data-type, optional
			Desired type of buffer elements. Use a type like (float, 2) to
			produce a buffer with shape (N, 2)
		allow_overwrite: bool
			If false, throw an IndexError when trying to append to an already
			full buffer
		"""
		self._capacity = N*2 + 1
		shape = [self._capacity,]
		shape.extend(sub_shape)
		# self._arr = np.empty(shape, dtype)
		self._arr = np.zeros(shape, dtype)
		self._write_index = 0
		self._read_index = 0
		self._allow_overwrite = allow_overwrite

	def _unwrap(self):
		""" Copy the data from this buffer into unwrapped form """
		return np.concatenate((
			self._arr[self._write_index:min(self._read_index, self._capacity)],
			self._arr[:max(self._read_index - self._capacity, 0)]
		))
		
	# numpy compatibility
	def __array__(self):
		return self._arr

	@property
	def dtype(self):
		return self._arr.dtype

	@property
	def shape(self):
		return (len(self),) + self._arr.shape[1:]

	# these mirror methods from deque
	@property
	def maxlen(self):
		return self._capacity

	def append(self, value):
		self._arr[self._write_index] = value
		# print(self._write_index, self._read_index)
		self._write_index = (self._write_index + 1 ) % self._capacity
		self._read_index = (self._write_index - 5 ) % self._capacity
	
	def get(self,):
		return self._arr[self._read_index]
		
	def flush_frames(self,):
		for x in range(self._read_index+1, self._read_index+self._capacity):
			yield self._arr[x % self._capacity]
	
	# implement Sequence methods
	def __len__(self):
		return self._read_index - self._write_index

	def __getitem__(self, item):
		# handle simple (b[1]) and basic (b[np.array([1, 2, 3])]) fancy indexing specially
		if not isinstance(item, tuple):
			item_arr = np.asarray(item)
			if issubclass(item_arr.dtype.type, np.integer):
				item_arr = (item_arr + self._write_index) % self._capacity
				return self._arr[item_arr]

		# for everything else, get it right at the expense of efficiency
		return self._unwrap()[item]

	def __iter__(self):
		# alarmingly, this is comparable in speed to using itertools.chain
		return iter(self._unwrap())

	# Everything else
	def __repr__(self):
		return '<RingBuffer of {!r}>'.format(np.asarray(self))
