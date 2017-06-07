""" Assemble the matrix for the derivative of the lumped mass matrix
Used to be run in parallel for debugging """

import dolfin as dl
from fenicstools.linalg.lumpedmatrixsolver import LumpedMassMatrixPrime

mesh = dl.UnitSquareMesh(20,20)
Va = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vphi = dl.FunctionSpace(mesh, 'Lagrange', 1)
ratioM = 2.0

LMMP = LumpedMassMatrixPrime(Va, Vphi, ratioM)
