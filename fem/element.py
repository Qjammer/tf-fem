import tensorflow as tf
import math

class BaseElement:
	self.nodes
	def __init__(self):
		pass
	def jacobianMatrix(self):
		pass

class TetrahedronElement(BaseElement):
	def __init__(self):
		pass
	def jacobianMatrix(self):
		pass


class BaseStdElement:
	def __init__(self):
		pass
	def integral(self,N_dims,B_dims,N_vertex,B_vertex):
		pass

class TetrahedronStdElement(BaseStdElement):

	def __init__(self):
		pass

	def integral(self,N_dims,B_dims,N_vertex,B_vertex):
		#if dimensions are repeated, they are summed over. Vertex 0 is origin, others are base vectors
		pass

	def monomer_integral_3simplex(self,nx,ny,nz):
		return math.factorial(nx)*math.factorial(ny)*math.factorial(nz)/math.factorial(3+nx+ny+nz)
