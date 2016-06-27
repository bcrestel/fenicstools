import numpy as np
import dolfin as dl

from fenicstools.miscfenics import checkdt, setfct
from fenicstools.acousticwave import AcousticWave
from fenicstools.sourceterms import PointSources, RickerWavelet
from fenicstools.observationoperator import TimeObsPtwise


def parametersinversionmrhs():
    # Create mesh and define function spaces
    Nxy = 50
    mesh = dl.UnitSquareMesh(Nxy, Nxy)
    Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
    r = 2
    V = dl.FunctionSpace(mesh, 'Lagrange', r)
    Dt = 1.0e-3
    t0, t1, t2, tf = 0.0, 0.5, 2.5, 3.0

    # source:
    Ricker = RickerWavelet(1.0, 1e-10)
    srcloc = [[0.5,1.0], [0.5,0.0], [0.0,0.5], [1.0,0.5]]
    Pt = PointSources(V, srcloc)
    src = dl.Function(V)
    srcv = src.vector()
    mysrc = [Ricker, Pt, srcv]

    # target medium:
    b_target = dl.Expression(\
    '1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    b_target_fn = dl.interpolate(b_target, Vm)
    a_target = dl.Expression(\
    '1.0 + 0.4*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    a_target_fn = dl.interpolate(a_target, Vm)

    # initial medium:
    b_initial = dl.Expression('1.0 + 0.25*sin(pi*x[0])*sin(pi*x[1])')
    b_initial_fn = dl.interpolate(b_initial, Vm)
    a_initial = dl.Expression('1.0 + 0.1*sin(pi*x[0])*sin(pi*x[1])')
    a_initial_fn = dl.interpolate(a_initial, Vm)

    checkdt(Dt, 1./Nxy, r, np.sqrt(2.0), False)

    # observation operator:
    obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
    [[1.0, ii/10.] for ii in range(1,10)] + \
    [[ii/10., 0.0] for ii in range(1,10)] + \
    [[ii/10., 1.0] for ii in range(1,10)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, [t0, t1, t2, tf])

    # define pde operator:
    wavepde = AcousticWave({'V':V, 'Vm':Vm})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
    't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})

    return a_target_fn, a_initial_fn, b_target_fn, b_initial_fn, \
    wavepde, mysrc, obsop



def parametersinversion():
    # Create mesh and define function spaces
    Nxy = 50
    mesh = dl.UnitSquareMesh(Nxy, Nxy)
    Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
    r = 2
    V = dl.FunctionSpace(mesh, 'Lagrange', r)
    Dt = 1.0e-3
    t0, t1, t2, tf = 0.0, 0.5, 2.5, 3.0

    # source:
    Ricker = RickerWavelet(1.0, 1e-10)
    Pt = PointSources(V, [[0.5,1.0]])
    src = dl.Function(V)
    srcv = src.vector()
    mysrc = [Ricker, Pt, srcv]

    # target medium:
    b_target = dl.Expression(\
    '1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    b_target_fn = dl.interpolate(b_target, Vm)
    a_target = dl.Expression(\
    '1.0 + 0.4*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    a_target_fn = dl.interpolate(a_target, Vm)

    # initial medium:
    b_initial = dl.Expression('1.0 + 0.25*sin(pi*x[0])*sin(pi*x[1])')
    b_initial_fn = dl.interpolate(b_initial, Vm)
    a_initial = dl.Expression('1.0 + 0.1*sin(pi*x[0])*sin(pi*x[1])')
    a_initial_fn = dl.interpolate(a_initial, Vm)

    checkdt(Dt, 1./Nxy, r, np.sqrt(2.0), False)

    # observation operator:
    obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
    [[1.0, ii/10.] for ii in range(1,10)] + \
    [[ii/10., 0.0] for ii in range(1,10)] + \
    [[ii/10., 1.0] for ii in range(1,10)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, [t0, t1, t2, tf])

    # define pde operator:
    wavepde = AcousticWave({'V':V, 'Vm':Vm})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
    't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    wavepde.ftime = mysrc

    return a_target_fn, a_initial_fn, b_target_fn, b_initial_fn, wavepde, obsop



def parametersinversionhighfreq():
    # Create mesh and define function spaces
    Nxy = 100
    mesh = dl.UnitSquareMesh(Nxy, Nxy)
    Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
    r = 2
    V = dl.FunctionSpace(mesh, 'Lagrange', r)
    Dt = 5.0e-4
    t0, t1, t2, tf = 0.0, 0.05, 1.5, 1.55

    # source:
    Ricker = RickerWavelet(4.0, 1e-10)
    Pt = PointSources(V, [[0.5,1.0]])
    src = dl.Function(V)
    srcv = src.vector()
    mysrc = [Ricker, Pt, srcv]

    # target medium:
    b_target = dl.Expression(\
    '1.0 + 1.0*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    b_target_fn = dl.interpolate(b_target, Vm)
    a_target = dl.Expression(\
    '1.0 + 0.4*(x[0]<=0.7)*(x[0]>=0.3)*(x[1]<=0.7)*(x[1]>=0.3)')
    a_target_fn = dl.interpolate(a_target, Vm)

    # initial medium:
    b_initial_fn = b_target_fn
    a_initial = dl.Expression('1.0 + 0.1*sin(pi*x[0])*sin(pi*x[1])')
    a_initial_fn = dl.interpolate(a_initial, Vm)

    checkdt(Dt, 1./Nxy, r, np.sqrt(2.0), True)

    # observation operator:
    obspts = [[0.0, ii/10.] for ii in range(1,10)] + \
    [[1.0, ii/10.] for ii in range(1,10)] + \
    [[ii/10., 0.0] for ii in range(1,10)] + \
    [[ii/10., 1.0] for ii in range(1,10)]
    obsop = TimeObsPtwise({'V':V, 'Points':obspts}, [t0, t1, t2, tf])

    # define pde operator:
    wavepde = AcousticWave({'V':V, 'Vm':Vm})
    wavepde.timestepper = 'backward'
    wavepde.lump = True
    wavepde.update({'a':a_target_fn, 'b':b_target_fn, \
    't0':t0, 'tf':tf, 'Dt':Dt, 'u0init':dl.Function(V), 'utinit':dl.Function(V)})
    wavepde.ftime = mysrc

    return a_target_fn, a_initial_fn, b_target_fn, b_initial_fn, wavepde, obsop
