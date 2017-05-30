"""
Acoustic wave inverse problem with a single low-frequency,
and absorbing boundary conditions on left, bottom, and right.
Check gradient and Hessian for joint inverse problem a and b
"""

import numpy as np

import dolfin as dl
from dolfin import MPI

from fenicstools.acousticwave import AcousticWave
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
regula = LaplacianPrior({'Vm':Vl, 'gamma':1e-4, 'beta':1e-4})
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
        sigmas = np.sqrt((dd**2).sum(axis=1)/dimsol)*0.1

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
# Compute gradient at initial parameters
waveobjab.update_PDE({'a':a0, 'b':b0})
waveobjab.solvefwd_cost()
waveobjabnoregul.update_PDE({'a':a0, 'b':b0})
waveobjabnoregul.solvefwd_cost()
if mpirank == 0:    
    print 'misfit at target={:.4e}; at initial state = {:.4e}'.format(\
    costmisfit, waveobjab.cost_misfit)
    print 'misfit at target={:.4e}; at initial state = {:.4e} [noregul]'.format(\
    costmisfitnoregul, waveobjabnoregul.cost_misfit)

mt = dl.Function(Vl*Vl)
dl.assign(mt.sub(0), at)
dl.assign(mt.sub(1), bt)

m0 = dl.Function(Vl*Vl)
m0.vector().zero()
m0.vector().axpy(1.0, mt.vector())
dl.assign(m0.sub(0), a0)

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
waveobjab.update_PDE({'a':a0, 'b':bt})
waveobjab.solvefwd_cost()
waveobjab.solveadj_constructgrad()
MGa, MGb = waveobjab.MG.split(deepcopy=True)
MGanorm = MGa.vector().norm('l2')
MGbnorm = MGb.vector().norm('l2')
if mpirank == 0:
    print '|MGa|={}, |MGb|={}'.format(MGanorm, MGbnorm)

waveobjabnoregul.update_PDE({'a':a0, 'b':bt})
waveobjabnoregul.solvefwd_cost()
waveobjabnoregul.solveadj_constructgrad()
MGaa, MGba = waveobjabnoregul.MG.split(deepcopy=True)
MGaa.vector().axpy(1.0, regula.grad(a0))
diffa = MGa.vector() - MGaa.vector()
diffb = MGb.vector() - MGba.vector()
MGaanorm = MGaa.vector().norm('l2')
MGbanorm = MGba.vector().norm('l2')
diffanorm = diffa.norm('l2')
diffbnorm = diffb.norm('l2')
if mpirank == 0:
    print '|MGaa|={} ({:.2e}), |MGba|={} ({:.2e})'.format(\
    MGaanorm, diffanorm, MGbanorm, diffbnorm)

waveobjab.assemble_hessian()
regula.assemble_hessian(a0)

waveobja = restrictobjabtoa(waveobjabnoregul, regula)

"""
print ' Check Hessian are the same'
yy = dl.Function(Vl*Vl)
xx = dl.Function(Vl*Vl)
xx.vector().zero()
xx.vector().axpy(-1.0, waveobjab.MGv)
for ii in range(5):
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
for ii in range(5):
    waveobja.mult(x.vector(), y.vector())
    x.vector().zero()
    x.vector().axpy(1.0, y.vector())
yn = y.vector().norm('l2')
diff = y.vector() - ya.vector()
diffn = diff.norm('l2')

if mpirank == 0:
    print '|ya|={}, |yb|={}, |yaa|={} ({:.2e})'.format(yan, ybn, yn, diffn)

waveobjab.mult(mt.vector(), yy.vector())
ya, yb = yy.split(deepcopy=True)
yan = ya.vector().norm('l2')
ybn = yb.vector().norm('l2')

waveobja.mult(at.vector(), y.vector())
yn = y.vector().norm('l2')
diff = y.vector() - ya.vector()
diffn = diff.norm('l2')

if mpirank == 0:
    print '|ya|={}, |yb|={}, |yaa|={} ({:.2e})'.format(yan, ybn, yn, diffn)

# Solve Newton system using CGSolverSteihaug
TOLCG = [0.5, 1e-2, 1e-4, 1e-6, 1e-12]
for tolcg in TOLCG:
    print '\ntolcg={}'.format(tolcg)
    solverab = CGSolverSteihaug()
    solverab.set_operator(waveobjab)
    solverab.set_preconditioner(waveobjab.getprecond())
    solverab.parameters["rel_tolerance"] = tolcg
    solverab.parameters["zero_initial_guess"] = True
    solverab.parameters["print_level"] = 1
    print '|MGv|={:.20e}'.format(waveobjab.MGv.norm('l2'))
    solverab.solve(waveobjab.srchdir.vector(), -1.0*waveobjab.MGv)
    print 'iter={}, final norm={} [ab]'.format(solverab.iter, solverab.final_norm)
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
    solvera.solve(waveobja.srchdir.vector(), -1.0*MGaa.vector())
    print 'iter={}, final norm={} [a]'.format(solvera.iter, solvera.final_norm)

    diffa = sola.vector() - waveobja.srchdir.vector()
    diffanorm = diffa.norm('l2')
    srchanorm = waveobja.srchdir.vector().norm('l2')
    if mpirank == 0:
        print '|sola|={}, |srcha|={}, reldiff={:.2e}'.format(\
        solanorm, srchanorm, diffanorm/srchanorm)
"""



# reproduce CG solver to compare:
rabf = dl.Function(Vl*Vl)
rab = rabf.vector()
rab.zero()
rab.axpy(-1.0, waveobjab.MGv)
xab = dl.Function(Vl*Vl).vector()
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

for ii in range(3):
    print '\nii={}'.format(ii)
    alphaab = nomab/denab
    alpha = nom/den

    xab.axpy(alphaab, dab)
    x.axpy(alpha, d)

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

    print 'alphaab={:.6e} ({:.2e}), nomab={:.6e} ({:.2e}), |raba|={:.6e} ({:.2e}), |rabb|={:.6e}'.format(\
    alphaab, np.abs(alpha-alphaab), nomab, np.abs(nom-nomab), nra, diffan, nrb)
    ###############################

    npcgab = solverab.solve(zab, rab)
    npcg = solver.solve(z, r)
    zaba, zabb = zabf.split(deepcopy=True)
    dzan = (zaba.vector()-z).norm('l2')
    dzbn = zabb.vector().norm('l2')
    print 'Preconditioner: |dza|={}, |zb|={}'.format(dzan, dzbn)
    

    betanomab = rab.inner(zab)
    betaab = betanomab/nomab
    dab *= betaab
    dab.axpy(1.0, zab)
    #
    betanom = r.inner(z)
    print 'Diff betanom={}'.format(np.abs(betanomab-betanom))
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
    
    denab = zab.inner(dab)
    den = z.inner(d)

    #TODO: Test and implement (if this works)
    """
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
    """

    print 'Diff den={} (den={}, denab={})'.format(np.abs(denab-den), den, denab)

    nomab = betanomab
    nom = betanom








