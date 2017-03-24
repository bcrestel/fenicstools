"""
Test different Krylov solver used to precondition NCG
"""

import dolfin as dl
from fenicstools.plotfenics import PlotFenics
from fenicstools.jointregularization import crossgradient, normalizedcrossgradient
from fenicstools.linalg.miscroutines import compute_eigfenics

N = 15
mesh = dl.UnitSquareMesh(N,N)
V = dl.FunctionSpace(mesh, 'CG', 1)
mpirank = dl.MPI.rank(mesh.mpi_comm())

cg = crossgradient(V*V)
ncg = normalizedcrossgradient(V*V)

outdir = 'Output-CGvsNCG-' + str(N) + 'x' + str(N) + '/'
plotfenics = PlotFenics(outdir, comm=mesh.mpi_comm())

#ones = dl.interpolate(dl.Expression(("1.0","1.0")), V*V).vector()
#x = dl.Function(V*V).vector()

#solvercg = dl.PETScKrylovSolver("cg", "ml_amg")
#solvercg.parameters["maximum_iterations"] = 2000
#solvercg.parameters["absolute_tolerance"] = 1e-24
#solvercg.parameters["relative_tolerance"] = 1e-24
#solvercg.parameters["error_on_nonconvergence"] = True 
#solvercg.parameters["nonzero_initial_guess"] = False 
#
#solvergmres = dl.PETScKrylovSolver("gmres", "ml_amg")
#solvergmres.parameters["maximum_iterations"] = 2000
#solvergmres.parameters["absolute_tolerance"] = 1e-24
#solvergmres.parameters["relative_tolerance"] = 1e-24
#solvergmres.parameters["error_on_nonconvergence"] = True 
#solvergmres.parameters["nonzero_initial_guess"] = False 

a1true = [
dl.interpolate(dl.Expression('log(10 - ' + 
'(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * 8 )'), V),
dl.interpolate(dl.Expression('log(10 - ' + 
'(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + 
'4*(x[0]<=0.5) + 8*(x[0]>0.5) ))'), V),
dl.interpolate(dl.Expression('log(10 - ' + 
'(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + 
'4*(x[0]<=0.5) + 8*(x[0]>0.5) ))'), V),
dl.interpolate(dl.Expression('log(10 - ' + 
'(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + 
'4*(x[0]<=0.5) + 8*(x[0]>0.5) ))'), V)]
a2true = [
dl.interpolate(dl.Expression('log(10 - ' + 
'(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + 
'8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), V),
 dl.interpolate(dl.Expression('log(10 - ' + 
'(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.4) * (' + 
'8*(x[0]<=0.5) + 4*(x[0]>0.5) ))'), V),
 dl.interpolate(dl.Expression('log(10)'), V),
dl.interpolate(dl.Expression('log(10 - ' + 
'(pow(pow(x[0]-0.5,2)+pow(x[1]-0.5,2),0.5)<0.2) * (' + 
'4*(x[0]<=0.5) + 8*(x[0]>0.5) ))'), V)]

ii=1
for a1, a2 in zip(a1true, a2true):
    if mpirank == 0:    print 'Test ' + str(ii)
    plotfenics.set_varname('a1-case'+str(ii))
    plotfenics.plot_vtk(a1)
    plotfenics.set_varname('a2-case'+str(ii))
    plotfenics.plot_vtk(a2)

    cg.assemble_hessianab(a1.vector(), a2.vector())
    ncg.assemble_hessianab(a1.vector(), a2.vector())

    if mpirank == 0:    print '\teigenvalues Hprecond'
    compute_eigfenics(cg.Hprecond, 
    outdir + 'eig_cg_Hprecond-' + str(ii) + '.txt')
    compute_eigfenics(ncg.Hprecond, 
    outdir + 'eig_ncg_Hprecond-' + str(ii) + '.txt')
    if mpirank == 0:    print '\teigenvalues H'
    compute_eigfenics(cg.H, 
    outdir + 'eig_cg_H-' + str(ii) + '.txt')
    compute_eigfenics(ncg.H, 
    outdir + 'eig_ncg_H-' + str(ii) + '.txt')

    ii += 1
    if mpirank == 0:    print ''


"""
test = dl.TestFunction(V*V)
trial = dl.TrialFunction(V*V)
M = dl.assemble(dl.inner(test, trial)*dl.dx)
rhs = M*ones
#rhs = ncg.H*ones


print '|rhs|={}'.format(dl.norm(rhs))
if mpirank == 0:    print '\tinvert full Hessian with CG'
solvercg.set_operator(ncg.Hprecond+1e-10*M)
#solvercg.set_operator(M)
solvercg.solve(x,rhs)
err = dl.norm(x-ones)/dl.norm(ones)
if mpirank == 0:    print 'errorcg={}'.format(err)

if mpirank == 0:    print '\tinvert full Hessian with GMRES'
solvergmres.set_operator(ncg.H)
solvergmres.solve(x,rhs)
err = dl.norm(x-ones)/dl.norm(ones)
if mpirank == 0:    print 'errorgmres={}'.format(err)
"""


