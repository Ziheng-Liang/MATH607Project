import numpy as np
import igl
import meshio
from mesh import Mesh
from integrator import *
from mesh_builder import *


# def Piola_kirchhoff(DG, D, M):
# 	for i in range()
	
if __name__ == '__main__':
	U, F = build_dense_cube()
	# U, F = build_square()
	V = U.copy()
	V = V * 0.9
	mesh = Mesh(V, F, U)
	dt = 0.1
	T = 100
	t = 0
	passion = 0.47
	young = 0.001
	# integrator = Forward_Euler(mesh, dt, T, t)
	integrator = Backward_Euler(mesh, dt, T, t)
	integrator.solve()