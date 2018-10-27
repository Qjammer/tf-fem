import numpy as np
import tensorflow as tf
import itertools as itt
import math


class Polynomial:
	def __init__(self,dimensions,maxdegree):
		self.sparse = tf.SparseTensor(indices=np.empty(shape=(0,dimensions)),values=[],dense_shape=maxdegree*tf.ones([dimensions],tf.int64))
	
	@staticmethod
	def getPolyProductMatrix(deg,dim):
		deg+=1
		enabled=[]
		for n in itt.product(range(deg),repeat=3*dim):
			degA=n[0:dim]
			degC=n[dim:2*dim]
			degB=n[2*dim:3*dim]
			succ=True
			for d in range(dim):
				if(degA[d]+degB[d]!=degC[d]):
					succ=False
					break
			if succ:
				enabled.append(n)
		vals=np.ones(len(enabled),dtype=np.float32)
		shape=deg*np.ones(3*dim)
		return tf.SparseTensor(indices=enabled,values=vals,dense_shape=shape)

	@staticmethod
	def dBody(i,D,MAT):
		i+=1
		MAT=tf.expand_dims(MAT,i)
		return [i,D,MAT]

	@staticmethod
	def dCond(i,D,MAT):
		return i<D

	@staticmethod
	def derivativeMatrix(deg,dim):
		#return tf.expand_dims(tf.range(0,deg+1),0)
		rang=tf.range(0,deg+1)
		lbod=Polynomial.dBody 
		lcond=Polynomial.dCond
		lvars=[0,dim,rang]
		tf.while_loop(lcond,lbod,lvars,return_same_structure=False)
		return tf.tile(tf.expand_dims(tf.range(0,deg+1),1),[1,dim])


	@staticmethod
	def derivative(pol,dim):
		return tf.tensordot(pol,Polynomial.derivativeMatrix(tf.maximum(tf.shape(pol)),dim),[[1],[0]])
		pass



	@staticmethod
	def __mul__(pol1,pol2,dim,resdeg):
		with tf.name_scope("polymul"):
			# Aliases for shapes
			with tf.name_scope("sh_A"):
				sh1=tf.shape(pol1)
				sh1r=tf.subtract((resdeg+1)*tf.ones(tf.shape(sh1),dtype=tf.int32),sh1)

			with tf.name_scope("sh_B"):
				sh2=tf.shape(pol2)
				sh2r=tf.subtract((resdeg+1)*tf.ones(tf.shape(sh2),dtype=tf.int32),sh2)

			# Padding construction
			with tf.name_scope("padded_A"):
				pad1=tf.transpose([tf.zeros(dim,dtype=tf.int32),sh2r])
				pol1pad=tf.pad(pol1,pad1)
			with tf.name_scope("padded_B"):
				pad2=tf.transpose([tf.zeros(dim,dtype=tf.int32),sh1r])
				pol2pad=tf.pad(pol2,pad2)

			with tf.name_scope("T"):
				pmult=Polynomial.getPolyProductMatrix(resdeg,dim)
				pmultdense=tf.sparse_tensor_to_dense(pmult)

			with tf.name_scope("AT"):
				prod1=tf.tensordot(pol1pad,pmultdense,axes=[np.r_[0:dim],np.r_[0:dim]])
				prod1s=tf.squeeze(prod1)

			with tf.name_scope("ATB"):
				prod2=tf.tensordot(prod1s,pol2pad,axes=[np.r_[dim:2*dim],np.r_[0:dim]])
				prod2s=tf.squeeze(prod2)

			return prod2s

	@staticmethod
	def tffactorial(n):
		return tf.exp(tf.lgamma(n+1))
	
	@staticmethod
	def monomer_integral_3simplex(nx,ny,nz):
		return Polynomial.tffactorial(nx)*Polynomial.tffactorial(ny)*Polynomial.tffactorial(nz)/Polynomial.tffactorial(3+nx+ny+nz)
	
	@staticmethod
	def fold_integral(prev,vals):
		return prev+vals[0]*Polynomial.monomer_integral_3simplex(vals[1],vals[2],vals[3])

	@staticmethod
	def total_integral(pol):
		with tf.name_scope("polyintegral"):
			# Non-zero indices
			idx=tf.where(tf.not_equal(pol,0))
			# Non-zero values
			vals=tf.gather_nd(pol,idx)
			ccat=tf.concat([tf.expand_dims(vals,1),tf.cast(idx,tf.float32)],1)
			return tf.foldl(Polynomial.fold_integral,ccat,initializer=tf.constant(0.))
