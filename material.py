import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from math import log

class Linear_Elastic:
	# def __init__(self, F):
	# 	self.F = F
	def __init__(self):
		pass

	def compute_P(self, F):
		P = np.zeros(F.shape)
		D = F.shape[1]
		I = np.eye(D, dtype=float)
		for i in range(int(F.shape[0]/D)):
			P[D*i:D*(i+1),:] = F[D*i:D*(i+1),:] - I
		return P

	# def update_P_Piola(self):
	# 	I = np.eye(self.D)
	# 	for i in range(self.M):
	# 		start_idx = self.D*i
	# 		end_idx = self.D*(i+1)
	# 		self.P[start_idx:end_idx,:] = self.DG[start_idx:end_idx,:] - I

class SVK:
	def __init__(self, lmbda, mu):
		self.lmbda = lmbda
		self.mu = mu

	def compute_P(self, DG):
		P = np.zeros(DG.shape)
		D = DG.shape[1]
		I = np.eye(D, dtype=float)
		for i in range(int(DG.shape[0]/D)):
			start_idx = D*i
			end_idx = D*(i+1)
			f = DG[start_idx:end_idx,:]
			E = np.dot(f.T, f) - I
			P[start_idx:end_idx,:] = np.dot(f, 2*self.mu*E + self.lmbda*np.trace(E)*I)
		return P

	def compute_dP(self, DG, dDG):
		dP = np.zeros(DG.shape)
		D = DG.shape[1]
		I = np.eye(D, dtype=float)
		for i in range(int(DG.shape[0]/D)):
			start_idx = D*i
			end_idx = D*(i+1)
			f = DG[start_idx:end_idx,:]
			df = dDG[start_idx:end_idx,:]
			E = 0.5*(np.dot(f.T, f) - I)
			dE = 0.5*(np.dot(df.T, f)+np.dot(f.T, df))
			dP[start_idx:end_idx,:] = np.dot(df, 2*self.mu*E+self.lmbda*np.trace(E)*I) + np.dot(f, 2*self.mu*dE+self.lmbda*np.trace(dE)*I)
		return dP

class Neohookean:
	def __init__(self, lmbda, mu):
		self.lmbda = lmbda
		self.mu = mu

	def compute_P(self, DG):
		P = np.zeros(DG.shape)
		D = DG.shape[1]
		for i in range(int(DG.shape[0]/D)):
			start_idx = D*i
			end_idx = D*(i+1)
			f = DG[start_idx:end_idx,:]
			P[start_idx:end_idx,:] = self.mu*(f - self.mu*inv(f.T)) + self.lmbda*log(det(f))*inv(f.T)
		return P

	def compute_dP(self, DG, dDG):
		dP = np.zeros(DG.shape)
		D = DG.shape[1]
		for i in range(int(DG.shape[0]/D)):
			start_idx = D*i
			end_idx = D*(i+1)
			f = DG[start_idx:end_idx,:]
			df = dDG[start_idx:end_idx,:]
			dP[start_idx:end_idx,:] = self.mu*df + np.dot((self.mu-self.lmbda*det(f))*inv(f.T), np.dot(df.T, inv(f.T))) + self.lmbda*np.trace(np.dot(inv(f), df))*inv(f.T)
		return dP