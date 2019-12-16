import numpy as np

def build_square():
	V = np.array([
		[0,0],
		[0,2],
		[2,2],
		[2,0],
		[1,1]], dtype=float)

	F = np.array([
		[0,1,4],
		[1,2,4],
		[2,3,4],
		[3,0,4]],dtype=int)

	return (V,F)

def build_dense_cube():
	nx = 10
	ny = 10
	nz = 10
	dx = 1.0 / (nx-1)
	dy = 1.0 / (ny-1)
	dz = 1.0 / (nz-1)

	V = np.zeros([nx*ny*nz, 3], dtype=float)
	for k in range(nz):
		for j in range(ny):
			for i in range(nx):
				x = i + j*nx + k*nx*ny
				V[x, :] = np.array([-0.5 + i*dx, -0.5 + j*dy, -0.5 + k*dz])
	n_points = V.shape[0]

	F = np.zeros([5*(nx-1)*(ny-1)*(nz-1), 4], dtype=int)

	for k in range(nz-1):
		for j in range(ny-1):
			for i in range(nx-1):
				cell_id = i + j*(nx-1) + k*(nx-1)*(ny-1)
				x = i + j*nx + k*nx*ny
				if cell_id % 2 == 1:
					F[5*cell_id,:] = [x, x+1, x+nx*ny+1, x+nx+1]
					F[5*cell_id+1,:] = [x, x+nx*ny+1, x+nx*(ny+1), x+nx+1]
					F[5*cell_id+2,:] = [x, x+nx*ny+1, x+ny*ny, x+nx*(ny+1)]
					F[5*cell_id+3,:] = [x, x+nx*(ny+1), x+nx, x+nx+1]
					F[5*cell_id+4,:] = [x+nx+1, x+nx*(ny+1), x+nx*(ny+1)+1, x+nx*ny+1]
				else:
					F[5*cell_id,:] = [x, x+1, x+nx*ny, x+nx]
					F[5*cell_id+1,:] = [x+1, x+nx+1, x+nx*(ny+1)+1, x+nx]
					F[5*cell_id+2,:] = [x+1, x+nx*ny, x+nx, x+nx*(ny+1)+1]
					F[5*cell_id+3,:] = [x+1, x+nx*(ny+1)+1, x+nx*ny+1, x+nx*ny]
					F[5*cell_id+4,:] = [x+nx, x+nx*ny, x+nx*(ny+1), x+nx*(ny+1)+1]
	return (V, F)