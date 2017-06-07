"""
Test inner and norm for vectors
"""

import dolfin as dl
import numpy as np
from fenicstools.miscfenics import createMixedFS

def test1():
    """
    Test norm consisten with inner
    """
    print 'TEST 1'
    mesh = dl.UnitSquareMesh(50,50)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    
    for ii in range(10):
        u = dl.interpolate(dl.Expression('sin(nn*pi*x[0])*sin(nn*pi*x[1])', \
        nn=ii+1, degree=10), V)

        normn = u.vector().norm('l2')
        normi = np.sqrt(u.vector().inner(u.vector()))

        print 'ii={}, rel_err={}'.format(ii, np.abs(normn-normi)/normn)


def test2():
    """
    Test behaviour norm in MixedFunctionSpace
    """
    print 'TEST 2'
    mesh = dl.UnitSquareMesh(50,50)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    VV = createMixedFS(V, V)
    
    for ii in range(10):
        u = dl.interpolate(dl.Expression(\
        ('sin(nn*pi*x[0])*sin(nn*pi*x[1])','0.0'),\
        nn=ii+1, degree=10), VV)

        normn = u.vector().norm('l2')

        ux, uy = u.split(deepcopy=True)
        normux = ux.vector().norm('l2')
        normuy = uy.vector().norm('l2')
        normuxuy = np.sqrt(normux**2 + normuy**2)

        print 'ii={}, rel_err={}'.format(ii, np.abs(normn-normuxuy)/normn)



def test3():
    """
    Test behaviour inner in MixedFunctionSpace

    inner-product in MixedFunctionSpace not exactl the same
    as sum of both inner-products in underlying FunctionSpaces
    must be due to re-ordering in MixedFunctionSpace that creates different
    summation sequence in inner-product, and therefore different numerical
    truncation
    """
    print 'TEST 3'
    mesh = dl.UnitSquareMesh(50,50)
    V = dl.FunctionSpace(mesh, 'CG', 1)
    VV = createMixedFS(V, V)
    
    for ii in range(10):
        u = dl.interpolate(dl.Expression(\
        ('sin(nn*pi*x[0])*sin(nn*pi*x[1])','0.0'),\
        nn=ii+1, degree=10), VV)
        v = dl.interpolate(dl.Expression(\
        ('nn*x[0]*x[1]','0.0'), nn=ii+1, degree=10), VV)

        #uv = u.vector().inner(v.vector())
        uv = dl.as_backend_type(u.vector()).vec().dot(\
        dl.as_backend_type(v.vector()).vec())

        ux, uy = u.split(deepcopy=True)
        vx, vy = v.split(deepcopy=True)
        #uvx = ux.vector().inner(vx.vector())
        #uvy = uy.vector().inner(vy.vector())
        uvx = dl.as_backend_type(ux.vector()).vec().dot(\
        dl.as_backend_type(vx.vector()).vec())
        uvy = dl.as_backend_type(uy.vector()).vec().dot(\
        dl.as_backend_type(vy.vector()).vec())
        
        uvxuvy = uvx + uvy

        print 'ii={}, rel_err={}'.format(ii, np.abs(uv-uvxuvy)/uv)


if __name__ == "__main__":
    test1()
    test2()
    test3()
