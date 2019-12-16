import numpy as np
import igl
import meshio
from numpy.linalg import solve
class Forward_Euler:
	def __init__(self, mesh, dt, T, t):
		self.mesh = mesh
		self.dt = dt
		self.T = T
		self.t = t
		self.acc = np.zeros(mesh.V.shape)
		self.vol = np.zeros(mesh.V.shape)
		self.mass = np.ones(mesh.V.shape[0])

	def solve(self):
		idx = 0
		while self.t < self.T:
			if self.mesh.F.shape[1] == 3:
				cells = {"triangle": self.mesh.F}
			else:
				cells = {"tetra": self.mesh.F}
			meshio.write_points_cells("output/animation_" + str(idx) + ".vtk", self.mesh.V, cells)
			self.t += self.dt 
			self.mesh.compute_force()
			self.acc = self.mesh.force / self.mass[:,None]
			self.vol = self.vol * 0.9 + self.acc * self.dt
			# self.mesh.dV = self.vol * self.dt
			self.mesh.V += self.vol * self.dt
			idx += 1

class Backward_Euler:
	def __init__(self, mesh, dt, T, t):
		self.mesh = mesh
		self.dt = dt
		self.T = T
		self.t = t
		self.acc = np.zeros(mesh.V.shape)
		self.vol = np.zeros(mesh.V.shape)
		self.mass = np.ones(mesh.V.shape[0])

	def solve(self):
		idx = 0
		while self.t < self.T:
			if self.mesh.F.shape[1] == 3:
				cells = {"triangle": self.mesh.F}
			else:
				cells = {"tetra": self.mesh.F}
			meshio.write_points_cells("output/animation_" + str(idx) + ".vtk", self.mesh.V, cells)
			self.t += self.dt 
			self.mesh.compute_force()
			self.mesh.compute_force_differential()
			D = self.mesh.V.shape[1]
			I = np.eye(D, dtype=float)
			for i in range(self.mesh.V.shape[0]):
				K = np.dot(self.mesh.dV[i,:].T, self.mesh.dforce[i,:])
				A = I - self.dt*self.dt*K
				b = self.dt*(self.mesh.force[i,:].T+self.dt*self.vol[i,:].T)
				dv = solve(A,b)
				self.vol[i,:] = dv * self.dt
			self.mesh.dV = self.vol * self.dt
			self.mesh.V += self.vol * self.dt
			idx += 1