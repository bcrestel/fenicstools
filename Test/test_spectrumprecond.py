from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh,'Lagrange', 2)
test, trial = TestFunction(V), TrialFunction(V)

wkformK = inner(nabla_grad(trial), nabla_grad(test))*dx
wkformM = inner(trial, test)*dx
K = assemble(wkformK)
M = assemble(wkformM)

# target medium:
b_target = Expression(\
'1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
b_target_fn = interpolate(b_target, V)
a_target = Expression(\
'1.0 + 0.4*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
a_target_fn = interpolate(a_target, V)

# initial medium:
b_initial = Expression('1.0 + 0.25*sin(pi*x[0])*sin(pi*x[1])')
b_initial_fn = interpolate(b_initial, V)
a_initial = Expression('1.0 + 0.1*sin(pi*x[0])*sin(pi*x[1])')
a_initial_fn = interpolate(a_initial, V)

Ka = K * a_target_fn.vector()
Kb = K * b_target_fn.vector()
Ma = M * a_target_fn.vector()
Mb = M * b_target_fn.vector()
print '|K.a|={}, |M.a|={}'.format(np.linalg.norm(Ka.array()), np.linalg.norm(Ma.array()))
print '|K.b|={}, |M.b|={}'.format(np.linalg.norm(Kb.array()), np.linalg.norm(Mb.array()))
Ka = K * a_initial_fn.vector()
Kb = K * b_initial_fn.vector()
Ma = M * a_initial_fn.vector()
Mb = M * b_initial_fn.vector()
print '|K.a|={}, |M.a|={}'.format(np.linalg.norm(Ka.array()), np.linalg.norm(Ma.array()))
print '|K.b|={}, |M.b|={}'.format(np.linalg.norm(Kb.array()), np.linalg.norm(Mb.array()))

wkform1 = 1e-5 * wkformK + 1e-12 * wkformM
wkform2 = 1e-5 * wkformK + 1e-7 * wkformM
P1 = PETScMatrix()
assemble(wkform1, tensor=P1)
P2 = PETScMatrix()
assemble(wkform2, tensor=P2)
eigensolverP1 = SLEPcEigenSolver(P1)
eigensolverP1.solve()
eigensolverP2 = SLEPcEigenSolver(P2)
eigensolverP2.solve()
nbeigenP1 = eigensolverP1.get_number_converged()
nbeigenP2 = eigensolverP2.get_number_converged()


eigenvaluesP1 = []
for ii in range(nbeigenP1):
    eigenvaluesP1.append(eigensolverP1.get_eigenvalue(ii)[0])
eigenvaluesP1.sort()
eigenvaluesP2 = []
for ii in range(nbeigenP2):
    eigenvaluesP2.append(eigensolverP2.get_eigenvalue(ii)[0])
eigenvaluesP2.sort()

print 'min(eigen(P1))={}, min(eigen(P2))={}'.format(np.amin(eigenvaluesP1),\
np.amin(eigenvaluesP2))
print 'cond(P1)={:.2e}, cond(P2)={:.2e}'.format(np.amax(eigenvaluesP1)/np.amin(eigenvaluesP1),\
np.amax(eigenvaluesP2)/np.amin(eigenvaluesP2))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(eigenvaluesP1, 'o-b')
ax.semilogy(eigenvaluesP2, '*-r')
plt.show()
