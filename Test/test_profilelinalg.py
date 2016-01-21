from dolfin import UnitSquareMesh, FunctionSpace, Expression, interpolate, \
Function
from fenicstools.acousticwave import AcousticWave

@profile
def run_test():
    q = 3
    Nxy = 100
    tf = 0.1
    h = 1./Nxy
    mesh = UnitSquareMesh(Nxy, Nxy, "crossed")
    V = FunctionSpace(mesh, 'Lagrange', q)
    Vl = FunctionSpace(mesh, 'Lagrange', 1)
    Dt = h/(q*10.)
    u0_expr = Expression(\
    '100*pow(x[i]-.25,2)*pow(x[i]-0.75,2)*(x[i]<=0.75)*(x[i]>=0.25)', i=0)


    Wave = AcousticWave({'V':V, 'Vl':Vl, 'Vr':Vl})
    Wave.lump = True
    Wave.timestepper = 'backward'
    Wave.update({'lambda':1.0, 'rho':1.0, 't0':0.0, 'tf':tf, 'Dt':Dt,\
    'u0init':interpolate(u0_expr, V), 'utinit':Function(V)})
    K = Wave.K
    u = Wave.u0
    u.vector()[:] = 1.0
    b = Wave.u1
    for ii in range(100):
        K*u.vector()
        (K*u.vector()).array()

        b.vector()[:] = (K*u.vector()).array()
        b.vector()[:] = 0.0
        b.vector().axpy(1.0, K*u.vector())

        b.vector()[:] = (K*u.vector()).array() + (K*u.vector()).array()
        b.vector()[:] = (K*u.vector() + K*u.vector()).array()

        b.vector()[:] = 2.*u.vector().array() + u.vector().array() + \
        Dt*u.vector().array()
        b.vector()[:] = 0.0
        b.vector().axpy(2.0, u.vector())
        b.vector().axpy(1.0, u.vector())
        b.vector().axpy(Dt, u.vector())
        b.assign(u)
        b.vector().zero()

if __name__ == "__main__":
    run_test()
