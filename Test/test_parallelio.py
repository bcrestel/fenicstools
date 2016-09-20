""" 
Test possibility to save and load Fenics function in parallel with different
number of processes.

Status: works well; file can be saved and loaded with different nb of processes
"""
import dolfin as dl
from dolfin import MPI

mpicomm = dl.mpi_comm_world()
mpirank = MPI.rank(mpicomm)
mpisize = MPI.size(mpicomm)

# save == 1 -> create function and save it to file
# save == 2 -> load file and plot function
save = 2


mesh = dl.UnitSquareMesh(20,20)
V = dl.FunctionSpace(mesh, 'Lagrange', 2)
if save == 1:
    # Save file
    b_expr = dl.Expression(\
    '1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    b = dl.interpolate(b_expr, V)
    dl.plot(b, interactive=True)
    fhdf5 = dl.HDF5File(mpicomm, 'Outputs/b.h5', 'w')
    fhdf5.write(b, 'b')
else:
    # Load file
    b_in = dl.Function(V)
    fhdf5 = dl.HDF5File(mpicomm, 'Outputs/b.h5', 'r')
    fhdf5.read(b_in, 'b')
    dl.plot(b_in, interactive=True)
