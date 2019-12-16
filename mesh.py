import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from math import log
from material import *
class Mesh:

	def __init__(self, V, F, U):
		self.V = V # world space vertices
		self.F = F # faces
		self.U = U # reference space vertices
		self.dV = np.zeros(U.shape)
		self.N, self.D = V.shape
		self.M, self.C = F.shape
		self.DG = np.zeros([self.D*self.M, self.D])
		self.Dm_inv = np.zeros([self.D*self.M, self.D])
		self.Ds = np.zeros([self.D*self.M, self.D])
		self.P = np.zeros([self.D*self.M, self.D])
		self.force = np.zeros([self.N, self.D])
		# self.Mat = Linear_Elastic()
		# self.Mat = Neohookean(0.47, 0.001)
		self.Mat = SVK(0.47, 0.001)
		self.dDs = np.zeros([self.D*self.M, self.D])
		self.dDG = np.zeros([self.D*self.M, self.D])
		self.dforce = np.zeros([self.N, self.D])
		self.dP = np.zeros([self.D*self.M, self.D])

	def update_DG(self):
		if self.C == 4:
			G = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]])
		elif self.C == 3:
			G = np.array([[1,0],[0,1],[-1,-1]])

		for i in range(self.M):
			v = self.V[self.F[i,:],:].T
			u = self.U[self.F[i,:],:].T
			dv = self.dV[self.F[i,:],:].T
			start_idx = self.D*i
			end_idx = self.D*(i+1)
			self.Dm_inv[start_idx:end_idx,:] = inv(np.dot(u,G))
			self.Ds[start_idx:end_idx,:] = np.dot(v,G)
			self.DG[start_idx:end_idx,:] = np.dot(self.Ds[start_idx:end_idx,:], self.Dm_inv[start_idx:end_idx,:])
			self.dDs[start_idx:end_idx,:] = np.dot(dv,G)
			self.dDG[start_idx:end_idx,:] = np.dot(self.dDs[start_idx:end_idx,:], self.Dm_inv[start_idx:end_idx,:])

	def compute_force_differential(self):
		self.update_DG()
		self.dP = self.Mat.compute_dP(self.DG, self.dDG)
		for i in range(self.M):
			start_idx = self.D*i
			end_idx = self.D*(i+1)
			W = 1.0/(6.0*abs(det(self.Dm_inv[start_idx:end_idx,:])))
			dH = -W*np.dot(self.dP[start_idx:end_idx,:], self.Dm_inv[start_idx:end_idx,:].T)
			if self.C == 4:
				self.dforce[self.F[i,:],:] += [dH[:,0],dH[:,1],dH[:,2], -dH[:,0]-dH[:,1]-dH[:,2]]
			elif self.C == 3:
				self.dforce[self.F[i,:],:] += [dH[:,0],dH[:,1],-dH[:,0]-dH[:,1]]

	def compute_force(self):
		self.update_DG()
		self.P = self.Mat.compute_P(self.DG)
		for i in range(self.M):
			start_idx = self.D*i
			end_idx = self.D*(i+1)
			W = 1.0/(6.0*abs(det(self.Dm_inv[start_idx:end_idx,:])))
			H = -W*np.dot(self.P[start_idx:end_idx,:], self.Dm_inv[start_idx:end_idx,:].T)
			if self.C == 4:
				self.force[self.F[i,:],:] += [H[:,0],H[:,1],H[:,2], -H[:,0]-H[:,1]-H[:,2]]
			elif self.C == 3:
				self.force[self.F[i,:],:] += [H[:,0],H[:,1],-H[:,0]-H[:,1]]