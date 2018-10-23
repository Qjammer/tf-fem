import tensorflow as tf

class BaseElement:
	nodes
	def __init__(self):
		pass
	def jacobianMatrix():
		pass

class TetrahedronElement(base_element):
	def __init__():
		pass
	def jacobianMatrix():
		pass


class BaseStdElement:
	def __init__():
		pass
	def integral(N_dims,B_dims,N_vertex,B_vertex):
		pass

class TetrahedronStdElement(base_std_element):

	def __init__():
		pass

	def integral(N_dims,B_dims,N_vertex,B_vertex):
		#if dimensions are repeated, they are summed over. Vertex 0 is origin, others are base vectors
		pass

	def monomer_integral_3simplex(nx,ny,nz):
		return math.factorial(nx)*math.factorial(ny)*math.factorial(nz)/math.factorial(3+nx+ny+nz)
