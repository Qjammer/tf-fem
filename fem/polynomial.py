import tensorflow as tf

class Polynomial:	
	def __init__(self,dimensions,maxdegree):
		self.sparse = tf.SparseTensor(dense_shape=maxdegree*tf.ones([dimensions,1],tf.int64))

	def __mul__(pol1,pol2):
		pass
		