"""
Compare solution to Newton system when inverting for a single parameters
"""

import numpy as np

import dolfin as dl
from dolfin import MPI

from fenicstools.acousticwave import AcousticWave
from fenicstools.optimsolver import bcktrcklinesearch
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.objectiveacoustic import ObjectiveAcoustic
from fenicstools.prior import LaplacianPrior
from fenicstools.regularization import TVPD
from fenicstools.jointregularization import SingleRegularization, V_TVPD
#from fenicstools.examples.acousticwave.mediumparameters0 import \
from fenicstools.examples.acousticwave.mediumparameters import \
targetmediumparameters, initmediumparameters, loadparameters

from hippylib.cgsolverSteihaug import CGSolverSteihaug

dl.set_log_active(False)


class restrictobjabtoa():
    def __init__(self, obj, regul):
        self.obj = obj
        self.regul = regul

        Vm = self.obj.PDE.Vm
        self.ahat = dl.Function(Vm)
        self.abhat = dl.Function(Vm*Vm)
        self.yab = dl.Function(Vm*Vm)
        self.srchdir = dl.Function(Vm)

        test, trial = dl.TestFunction(Vm), dl.TrialFunction(Vm)
        self.M = dl.assemble(dl.inner(test, trial)*dl.dx)


    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)


    def mult(self, ahat, y):
        self.ahat.vector().zero()
        self.ahat.vector().axpy(1.0, ahat)

        self.abhat.vector().zero()
        dl.assign(self.abhat.sub(0), self.ahat)

        self.obj.mult(self.abhat.vector(), self.yab.vector())

        ya, yb = self.yab.split(deepcopy=True)
        y.zero()
        y.axpy(1.0, ya.vector())
        y.axpy(1.0, self.regul.hessian(ahat))


NOISE = True

Nxy, Dt, fpeak, t0, t1, t2, tf = loadparameters(False)

# Define PDE:
h = 1./Nxy
# dist is in [km]
X, Y = 1, 1
mesh = dl.UnitSquareMesh(Nxy, Nxy)
mpicomm = mesh.mpi_comm()
mpirank = MPI.rank(mpicomm)
Vl = dl.FunctionSpace(mesh, 'Lagrange', 1)
Ricker = RickerWavelet(fpeak, 1e-6)
r = 2   # polynomial degree for state and adj
V = dl.FunctionSpace(mesh, 'Lagrange', r)
Pt = PointSources(V, [[0.5*X,Y]])
srcv = dl.Function(V).vector()
# Boundary conditions:
class ABCdom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < Y)
# Computation:
Wave = AcousticWave({'V':V, 'Vm':Vl}, 
{'print':False, 'lumpM':True, 'timestepper':'backward'})
#Wave.set_abc(mesh, ABCdom(), lumpD=False)
#
at, bt,_,_,_ = targetmediumparameters(Vl, X)
a0, b0,_,_,_ = initmediumparameters(Vl, X)
#
Wave.update({'b':bt, 'a':at, 't0':0.0, 'tf':tf, 'Dt':Dt,\
'u0init':dl.Function(V), 'utinit':dl.Function(V)})
# observation operator:
obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
[[1.0, ii/10.] for ii in range(1,10)] + \
[[ii/10., 0.0] for ii in range(1,10)] + \
[[ii/10., 1.0] for ii in range(1,10)]
tfilterpts = [t0, t1, t2, tf]
obsop = TimeObsPtwise({'V':V, 'Points':obspts}, tfilterpts)

# Regularization
regula = LaplacianPrior({'Vm':Vl, 'gamma':1e-4, 'beta':1e-4, 'm0':at})
regulab = SingleRegularization(regula, 'a', (not mpirank))

# define objective function:
waveobjab = ObjectiveAcoustic(Wave, [Ricker, Pt, srcv], 'a', regulab)
waveobjabnoregul = ObjectiveAcoustic(Wave, [Ricker, Pt, srcv], 'a')
waveobjab.obsop = obsop
waveobjabnoregul.obsop = obsop
# Generate synthetic observations
if mpirank == 0:    print 'generate noisy data'
waveobjab.solvefwd()
DD = waveobjab.Bp[:]
if NOISE:
    SNRdB = 20.0   # [dB], i.e, log10(mu/sigma) = SNRdB/10
    np.random.seed(11)
    for ii, dd in enumerate(DD):
        if mpirank == 0:    print 'source {}'.format(ii)
        nbobspt, dimsol = dd.shape

        #mu = np.abs(dd).mean(axis=1)
        #sigmas = mu/(10**(SNRdB/10.))
        sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*0.01

        rndnoise = np.random.randn(nbobspt*dimsol).reshape((nbobspt, dimsol))
        print 'mpirank={}, sigmas={}, |rndnoise|={}'.format(\
        mpirank, sigmas.sum()/len(sigmas), (rndnoise**2).sum().sum())
        DD[ii] = dd + sigmas.reshape((nbobspt,1))*rndnoise
        MPI.barrier(mpicomm)
waveobjab.dd = DD
waveobjabnoregul.dd = DD
# check:
waveobjab.solvefwd_cost()
costmisfit = waveobjab.cost_misfit
waveobjabnoregul.solvefwd_cost()
costmisfitnoregul = waveobjabnoregul.cost_misfit

# Compare cost functionals from both objective functions
waveobjab.update_PDE({'a':a0, 'b':b0})
waveobjab.solvefwd_cost()
waveobjabnoregul.update_PDE({'a':a0, 'b':b0})
waveobjabnoregul.solvefwd_cost()
if mpirank == 0:    
    print 'misfit at target={:.6e}; at initial state={:.6e}'.format(\
    costmisfit, waveobjab.cost_misfit)
    print '[noregul] misfit at target={:.6e} (df={:.2e}), at initial state={:.6e} (df={:.2e})'.format(\
    costmisfitnoregul, np.abs(costmisfit-costmisfitnoregul)/costmisfit,\
    waveobjabnoregul.cost_misfit, \
    np.abs(waveobjabnoregul.cost_misfit-waveobjab.cost_misfit)/waveobjab.cost_misfit)

mt = dl.Function(Vl*Vl)
dl.assign(mt.sub(0), at)
dl.assign(mt.sub(1), bt)

# check regularizations are the same
regt = regulab.costab(at,bt)
reg0 = regulab.costab(a0,b0)
regta = regula.cost(at)
reg0a = regula.cost(a0)
if mpirank == 0:
    print 'Regularization at target={:.2e}, at initial state={:.2e} [ab]'.format(\
    regt, reg0)
    print 'Regularization at target={:.2e}, at initial state={:.2e} [a]'.format(\
    regta, reg0a)

# check gradients are the same
evaluationpoint = {'a':at, 'b':bt}
waveobjab.update_PDE(evaluationpoint)
waveobjab.solvefwd_cost()
waveobjab.solveadj_constructgrad()
MGa, MGb = waveobjab.MG.split(deepcopy=True)
MGanorm = MGa.vector().norm('l2')
MGbnorm = MGb.vector().norm('l2')
if mpirank == 0:
    print '|MGa|={}, |MGb|={}'.format(MGanorm, MGbnorm)

waveobjabnoregul.update_PDE(evaluationpoint)
waveobjabnoregul.solvefwd_cost()
waveobjabnoregul.solveadj_constructgrad()
MGaa, MGba = waveobjabnoregul.MG.split(deepcopy=True)
MGaa.vector().axpy(1.0, regula.grad(evaluationpoint['a']))
diffa = MGa.vector() - MGaa.vector()
diffb = MGb.vector() - MGba.vector()
MGaanorm = MGaa.vector().norm('l2')
MGbanorm = MGba.vector().norm('l2')
diffanorm = diffa.norm('l2')
diffbnorm = diffb.norm('l2')
if mpirank == 0:
    print '|MGaa|={} (df={:.2e}), |MGba|={} (df={:.2e})'.format(\
    MGaanorm, diffanorm, MGbanorm, diffbnorm)

waveobjab.assemble_hessian()
regula.assemble_hessian(evaluationpoint['a'])

waveobja = restrictobjabtoa(waveobjabnoregul, regula)

print ' Check Hessian are the same'
yy = dl.Function(Vl*Vl)
xx = dl.Function(Vl*Vl)
xx.vector().zero()
xx.vector().axpy(-1.0, waveobjab.MGv)
waveobjab.mult(xx.vector(), yy.vector())
xx.vector().zero()
xx.vector().axpy(1.0, yy.vector())
ya, yb = yy.split(deepcopy=True)
yan = ya.vector().norm('l2')
ybn = yb.vector().norm('l2')

y = dl.Function(Vl)
x = dl.Function(Vl)
x.vector().zero()
x.vector().axpy(-1.0, MGaa.vector())
waveobja.mult(x.vector(), y.vector())
x.vector().zero()
x.vector().axpy(1.0, y.vector())
yn = y.vector().norm('l2')
diff = y.vector() - ya.vector()
diffn = diff.norm('l2')

if mpirank == 0:
    print '|ya|={}, |yb|={}, |yaa|={} (df={:.2e})'.format(yan, ybn, yn, diffn)

waveobjab.mult(mt.vector(), yy.vector())
ya, yb = yy.split(deepcopy=True)
yan = ya.vector().norm('l2')
ybn = yb.vector().norm('l2')

waveobja.mult(at.vector(), y.vector())
yn = y.vector().norm('l2')
diff = y.vector() - ya.vector()
diffn = diff.norm('l2')

if mpirank == 0:
    print '|ya|={}, |yb|={}, |yaa|={} (df={:.2e})'.format(yan, ybn, yn, diffn)

# Solve Newton system using CGSolverSteihaug
#TOLCG = [0.5, 1e-2, 1e-4, 1e-6, 1e-12]
TOLCG = [1e-4]
for tolcg in TOLCG:
    print '\ntolcg={}'.format(tolcg)
    solverab = CGSolverSteihaug()
    solverab.set_operator(waveobjab)
    solverab.set_preconditioner(waveobjab.getprecond())
    solverab.parameters["rel_tolerance"] = tolcg
    solverab.parameters["zero_initial_guess"] = True
    solverab.parameters["print_level"] = 1
    print '|MGv|={:.20e}'.format(waveobjab.MGv.norm('l2'))
    solverab.solve(waveobjab.srchdir.vector(), -1.0*waveobjab.MGv, False)
    print '[ab]: iter={}, final norm={}, <dp,MG>={}'.format(\
    solverab.iter, solverab.final_norm, \
    waveobjab.srchdir.vector().inner(waveobjab.MGv))
    sola, solb = waveobjab.srchdir.split(deepcopy=True)
    solanorm = sola.vector().norm('l2')
    solbnorm = solb.vector().norm('l2')
    if mpirank == 0:    print '|solb|={}'.format(solbnorm)

    solvera = CGSolverSteihaug()
    solvera.set_operator(waveobja)
    solvera.set_preconditioner(regula.getprecond())
    solvera.parameters["rel_tolerance"] = tolcg
    solvera.parameters["zero_initial_guess"] = True
    solvera.parameters["print_level"] = 1
    print '|MGaa|={:.20e}'.format(MGaa.vector().norm('l2'))
    solvera.solve(waveobja.srchdir.vector(), -1.0*MGaa.vector(), False)
    print '[a]: iter={}, final norm={}, <dp,MG>={}'.format(\
    solvera.iter, solvera.final_norm,\
    waveobja.srchdir.vector().inner(MGaa.vector()))

    diffa = sola.vector() - waveobja.srchdir.vector()
    diffanorm = diffa.norm('l2')
    srchanorm = waveobja.srchdir.vector().norm('l2')
    if mpirank == 0:
        print '|sola|={:.16e}, |srcha|={:.16e}, reldiff={:.2e}'.format(\
        solanorm, srchanorm, diffanorm/srchanorm)


# Test each search direction
dpMG = waveobjab.srchdir.vector().inner(waveobjab.MGv)
print '[ab]: <dp,MG>={}'.format(dpMG)
_, LScount, alpha = bcktrcklinesearch(waveobjab)
meda, medb = waveobjab.mediummisfit(mt)
print '[ab]: LScount={}, alpha={}'.format(LScount, alpha)
print '[ab]: meda={:.16e}, medb={:.6e}'.format(meda, medb)
waveobjab.solveadj_constructgrad()
gradnorm = np.sqrt(waveobjab.MGv.inner(waveobjab.Grad.vector()))
print '[ab]: |G|={:.16e}'.format(gradnorm)

waveobjab.update_PDE(evaluationpoint)
waveobjab.solvefwd_cost()
waveobjab.solveadj_constructgrad()
waveobjab.srchdir.vector().zero()
dl.assign(waveobjab.srchdir.sub(0), waveobja.srchdir)
dpMG = waveobjab.srchdir.vector().inner(waveobjab.MGv)
print '[a]: <dp,MG>={}'.format(dpMG)
_, LScount, alpha = bcktrcklinesearch(waveobjab)
meda, medb = waveobjab.mediummisfit(mt)
print '[a]: LScount={}, alpha={}'.format(LScount, alpha)
print '[a]: meda={:.16e}, medb={:.6e}'.format(meda, medb)
waveobjab.solveadj_constructgrad()
gradnorm = np.sqrt(waveobjab.MGv.inner(waveobjab.Grad.vector()))
print '[a]: |G|={:.16e}'.format(gradnorm)



"""
# reproduce CG solver to compare:
rabf = dl.Function(Vl*Vl)
rab = rabf.vector()
rab.zero()
rab.axpy(-1.0, waveobjab.MGv)
xabf = dl.Function(Vl*Vl)
xab = xabf.vector()
xab.zero()
#
r = dl.Function(Vl).vector()
r.zero()
r.axpy(-1.0, MGaa.vector())
x = dl.Function(Vl).vector()
x.zero()

zabf = dl.Function(Vl*Vl)
zab = zabf.vector()
solverab = waveobjab.getprecond()
npcgab = solverab.solve(zab, rab)
#
z = dl.Function(Vl).vector()
solver = regula.getprecond()
npcg = solver.solve(z, r)

dabf = dl.Function(Vl*Vl)
dab = dabf.vector()
dab.zero()
dab.axpy(1.0, zab)
#
d = dl.Function(Vl).vector()
d.zero()
d.axpy(1.0, z)

nomab = dab.inner(rab)
nom = d.inner(r)

waveobjab.mult(dab, zab)
waveobja.mult(d, z)

denab = zab.inner(dab)
den = z.inner(d)

for ii in range(5):
    print '\nii={}'.format(ii)
    alphaab = nomab/denab
    alpha = nom/den

    xab.axpy(alphaab, dab)
    x.axpy(alpha, d)
    xabn = xab.norm('l2')
    xn = x.norm('l2')
    xaba, xabb = xabf.split(deepcopy=True)
    diffn = (xaba.vector()-x).norm('l2')
    print '|x|={:.16e}, reldiff={}'.format(xn, diffn/xn)

    rab.axpy(-alphaab, zab)
    r.axpy(-alpha, z)

    ###############################
    nrab = rab.norm('l2')
    print '|r|ab={:.24e}'.format(nrab)
    raba, rabb = rabf.split(deepcopy=True)
    nra = raba.vector().norm('l2')
    nrb = rabb.vector().norm('l2')

    nr = r.norm('l2')
    print '|r|a={:.24e}'.format(nr)

    diffa = raba.vector() - r
    diffan = diffa.norm('l2')

    print 'alphaab={:.6e} ({:.2e}), nomab={:.16e} ({:.2e}), |raba|={:.6e} ({:.2e}), |rabb|={:.6e}'.format(\
    alphaab, np.abs(alpha-alphaab), nomab, np.abs(nom-nomab), nra, diffan, nrb)
    ###############################

    npcgab = solverab.solve(zab, rab)
    npcg = solver.solve(z, r)
    zaba, zabb = zabf.split(deepcopy=True)
    dzan = (zaba.vector()-z).norm('l2')
    dzbn = zabb.vector().norm('l2')
    print 'Preconditioner: |dza|={}, |zb|={}'.format(dzan, dzbn)
    

    zabn = max(zab.max(), np.abs(zab.min()))
    rabn = max(rab.max(), np.abs(rab.min()))
    zabsc = zab/zabn
    rabsc = rab/rabn
    betanomab = (rabsc.inner(zabsc))*zabn*rabn
    #betanomab = rab.inner(zab)

    betaab = betanomab/nomab
    dab *= betaab
    dab.axpy(1.0, zab)
    #
    zn = max(z.max(), np.abs(z.min()))
    rn = max(r.max(), np.abs(r.min()))
    zsc = z/zn
    rsc = r/rn
    betanom = (rsc.inner(zsc))*zn*rn
    #betanom = r.inner(z)

    print 'betanom={:.16e}, diff={}'.format(betanom, np.abs(betanomab-betanom))
    beta = betanom/nom
    d *= beta
    d.axpy(1.0, z)
    daba, dabb = dabf.split(deepcopy=True)
    ddan = (daba.vector()-d).norm('l2')
    ddbn = dabb.vector().norm('l2')
    print 'd: |dda|={}, |db|={}'.format(ddan, ddbn)

    waveobjab.mult(dab, zab)
    waveobja.mult(d, z)
    zaba, zabb = zabf.split(deepcopy=True)
    dzan = (zaba.vector()-z).norm('l2')
    dzbn = zabb.vector().norm('l2')
    print 'z (Hessian-vect): |dza|={}, |zb|={}'.format(dzan, dzbn)
    
    zabn = max(zab.max(), np.abs(zab.min()))
    dabn = max(dab.max(), np.abs(dab.min()))
    zabsc = zab/zabn
    dabsc = dab/dabn
    denab = (zabsc.inner(dabsc))*zabn*dabn

    zn = max(z.max(), np.abs(z.min()))
    dn = max(d.max(), np.abs(d.min()))
    zsc = z/zn
    dsc = d/dn
    den = (zsc.inner(dsc))*zn*dn

    #denab = zab.inner(dab)
    #den = z.inner(d)

    print 'Diff den={} (den={:.16e})'.format(np.abs(denab-den), den)

    nomab = betanomab
    nom = betanom


diffxa = (waveobja.srchdir.vector() - x).norm('l2')
diffxab = (sola.vector() - xaba.vector()).norm('l2')
xaban = xaba.vector().norm('l2')
print 'diff x: [a]={:.2e}, [ab]={:.2e}'.format(diffxa/xn, diffxab/xaban)
"""





