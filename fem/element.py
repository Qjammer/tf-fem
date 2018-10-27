import tensorflow as tf
import math

class BaseElement:
	def __init__(self,nodes,stdElement):
		self.nodes=nodes
		self.stdElement=stdElement

	def jacobianMatrix(self):
		pass

	def getMatrix(self,N_dims,B_dims,N_vertex,B_vertex):
		return self.jacobianMatrix()*self.stdElement.getMatrix(N_dims,B_dims,N_vertex,B_vertex)

class TetrahedronElement(BaseElement):
	def __init__(self):
		super(TetrahedronElement,self).__init__(nodes,TetrahedronStdElement())
		pass

	def jacobianMatrix(self):
		return tf.transpose(nodes[1:]-nodes[0])



class BaseStdElement:
	def __init__(self):
		pass
	def getMatrix(self,N_dims,B_dims,N_vertex,B_vertex):
		pass

class TetrahedronStdElement(BaseStdElement):

	def __init__(self):
		super(TetrahedronStdElement,self).__init__()
		pass

	def getBasePolynomials(self,order):
		with tf.name_scope("basis_poly"):
			if(order!=1): raise ValueError("Element orders different from 1 are not implemented")
			return tf.constant([[[1.,-1.],[-1.,0.]],\
								[[-1.,0.],[0.,0.]]])
		

	def getMatrix(self,N_dims,B_dims,N_vertex,B_vertex):
		# If dimensions are repeated, they are summed over. Vertex 0 is origin, others are base vectors
		# 1. Generate polynomials Ns and Ni
		# 2. Compute product of polynomials NsNi
		# 3. Calculate integral of each polynomial
		# 4. Return matrix of integral of polynomials
		pass

