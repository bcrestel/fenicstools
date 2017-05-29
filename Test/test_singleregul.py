"""
Test SingleRegularization
"""
import numpy as np
import dolfin as dl

from fenicstools.jointregularization import SingleRegularization
from fenicstools.prior import LaplacianPrior
from fenicstools.regularization import TVPD

dl.set_log_active(False)

def test1():
    mesh = dl.UnitSquareMesh(40,40)
    Vm = dl.FunctionSpace(mesh, 'CG', 1)
    ab = dl.Function(Vm*Vm)
    xab = dl.Function(Vm*Vm)
    x = dl.Function(Vm)
    regul = LaplacianPrior({'Vm':Vm, 'gamma':1e-4, 'beta':1e-4})
    regula = LaplacianPrior({'Vm':Vm, 'gamma':1e-4, 'beta':1e-4})
    regulb = LaplacianPrior({'Vm':Vm, 'gamma':1e-4, 'beta':1e-4})
    jointregula = SingleRegularization(regula, 'a')
    jointregulb = SingleRegularization(regulb, 'b')

    for ii in range(4):
        ab.vector()[:] = (ii+1.0)*np.random.randn(2*Vm.dim())
        a, b = ab.split(deepcopy=True)

        print '\nTest a'
        costregul = regul.cost(a)
        costjoint = jointregula.costab(a,b)
        print 'cost={}, diff={}'.format(costregul, np.abs(costregul-costjoint))

        gradregul = regul.grad(a)
        gradjoint = jointregula.gradab(a,b)
        xab.vector().zero()
        xab.vector().axpy(1.0, gradjoint)
        ga, gb = xab.split(deepcopy=True)
        gan = ga.vector().norm('l2')
        gbn = gb.vector().norm('l2')
        diffn = (gradregul-ga.vector()).norm('l2')
        print '|ga|={}, diff={}, |gb|={}'.format(gan, diffn, gbn)

        regul.assemble_hessian(a)
        jointregula.assemble_hessianab(a,b)
        Hvregul = regul.hessian(a.vector())
        Hvjoint = jointregula.hessianab(a.vector(),b.vector())
        xab.vector().zero()
        xab.vector().axpy(1.0, Hvjoint)
        Ha, Hb = xab.split(deepcopy=True)
        Han = Ha.vector().norm('l2')
        Hbn = Hb.vector().norm('l2')
        diffn = (Hvregul-Ha.vector()).norm('l2')
        print '|Ha|={}, diff={}, |Hb|={}'.format(Han, diffn, Hbn)

        solvera = regul.getprecond()
        solveraiter = solvera.solve(x.vector(), a.vector())

        solverab = jointregula.getprecond()
        solverabiter = solverab.solve(xab.vector(), ab.vector())
        xa, xb = xab.split(deepcopy=True)
        diffn = (x.vector()-xa.vector()).norm('l2')
        diffbn = (b.vector()-xb.vector()).norm('l2')
        xan = xa.vector().norm('l2')
        xbn = xb.vector().norm('l2')
        print '|xa|={}, diff={}'.format(xan, diffn)
        print '|xb|={}, diff={}'.format(xbn, diffbn)
        print 'iter={}, diff={}'.format(solveraiter, np.abs(solveraiter-solverabiter))

        print 'Test b'
        costregul = regul.cost(b)
        costjoint = jointregulb.costab(a,b)
        print 'cost={}, diff={}'.format(costregul, np.abs(costregul-costjoint))

        gradregul = regul.grad(b)
        gradjoint = jointregulb.gradab(a,b)
        xab.vector().zero()
        xab.vector().axpy(1.0, gradjoint)
        ga, gb = xab.split(deepcopy=True)
        gan = ga.vector().norm('l2')
        gbn = gb.vector().norm('l2')
        diffn = (gradregul-gb.vector()).norm('l2')
        print '|gb|={}, diff={}, |ga|={}'.format(gbn, diffn, gan)

        regul.assemble_hessian(b)
        jointregulb.assemble_hessianab(a,b)
        Hvregul = regul.hessian(b.vector())
        Hvjoint = jointregulb.hessianab(a.vector(),b.vector())
        xab.vector().zero()
        xab.vector().axpy(1.0, Hvjoint)
        Ha, Hb = xab.split(deepcopy=True)
        Han = Ha.vector().norm('l2')
        Hbn = Hb.vector().norm('l2')
        diffn = (Hvregul-Hb.vector()).norm('l2')
        print '|Hb|={}, diff={}, |Ha|={}'.format(Hbn, diffn, Han)

        solverb = regul.getprecond()
        solverbiter = solverb.solve(x.vector(), b.vector())
        solverab = jointregulb.getprecond()
        solverabiter = solverab.solve(xab.vector(), ab.vector())
        xa, xb = xab.split(deepcopy=True)
        diffn = (x.vector()-xb.vector()).norm('l2')
        diffan = (a.vector()-xa.vector()).norm('l2')
        xan = xa.vector().norm('l2')
        xbn = xb.vector().norm('l2')
        print '|xb|={}, diff={}'.format(xbn, diffn)
        print '|xa|={}, diff={}'.format(xan, diffan)
        print 'iter={}, diff={}'.format(solverbiter, np.abs(solverbiter-solverabiter))


def test2():
    mesh = dl.UnitSquareMesh(40,40)
    Vm = dl.FunctionSpace(mesh, 'CG', 1)
    x1 = dl.Function(Vm)
    x2 = dl.Function(Vm)
    rhs = dl.Function(Vm)
    #regul = LaplacianPrior({'Vm':Vm, 'gamma':1e-4, 'beta':1e-4})
    regul = TVPD({'Vm':Vm, 'k':1e-4, 'eps':1e-3})
    precond = 'hypre_amg'

    for ii in range(1):
        rhs.vector()[:] = (ii+1.0)*np.random.randn(Vm.dim())
        regul.assemble_hessian(rhs)

        solver1 = dl.PETScKrylovSolver('cg', precond)
        solver1.parameters["maximum_iterations"] = 1000
        solver1.parameters["relative_tolerance"] = 1e-24
        solver1.parameters["absolute_tolerance"] = 1e-24
        solver1.parameters["error_on_nonconvergence"] = True 
        solver1.parameters["nonzero_initial_guess"] = False 
        #solver1 = dl.PETScLUSolver()
        solver1.set_operator(regul.precond)
        iter1 = solver1.solve(x1.vector(), rhs.vector())
        x1n = x1.vector().norm('l2')

        solver2 = dl.PETScKrylovSolver('cg', precond)
        solver2.parameters["maximum_iterations"] = 1000
        solver2.parameters["relative_tolerance"] = 1e-24
        solver2.parameters["absolute_tolerance"] = 1e-24
        solver2.parameters["error_on_nonconvergence"] = True 
        solver2.parameters["nonzero_initial_guess"] = False 
        #solver2 = dl.PETScLUSolver()
        solver2.set_operator(regul.precond)
        iter2 = solver2.solve(x2.vector(), rhs.vector())

        diffn = (x1.vector()-x2.vector()).norm('l2')

        print '|x1|={}, |diff|={}'.format(x1n, diffn)
        print 'iter={}, diff_iter={}'.format(iter1, np.abs(iter1-iter2))
    
        solver3 = dl.PETScKrylovSolver('cg', 'petsc_amg')
        solver3.parameters["maximum_iterations"] = 1000
        solver3.parameters["relative_tolerance"] = 1e-24
        solver3.parameters["absolute_tolerance"] = 1e-24
        solver3.parameters["error_on_nonconvergence"] = True 
        solver3.parameters["nonzero_initial_guess"] = False 
        solver3.set_operator(regul.precond)
        iter3 = solver3.solve(x2.vector(), rhs.vector())
        x3n = x2.vector().norm('l2')
        diffn = (x1.vector()-x2.vector()).norm('l2')
        print '|x3|={}, |diff|={}, iter={}'.format(x3n, diffn, iter3)

if __name__ == "__main__":
    test2()
