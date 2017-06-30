"""
Test computecfromab, by comparing original c with c computed from a,b
"""
import dolfin as dl

from fenicstools.examples.acousticwave.mediumparameters1 import \
targetmediumparameters
from fenicstools.miscfenics import computecfromab

mesh = dl.UnitSquareMesh(20,20)
V = dl.FunctionSpace(mesh,'CG',1)
a, b, c, _,_ = targetmediumparameters(V, 1.0)

cab = computecfromab(a.vector(), b.vector())

diffc = cab-c.vector()
maxdiffc = diffc.max()
mindiffc = diffc.min()
errc = dl.norm(diffc)/dl.norm(c.vector())
print 'rel err c={:.4e}, max(diff)={:.4e}, min(diff)={:.4e}'.format(\
errc, maxdiffc, mindiffc)
