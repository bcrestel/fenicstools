from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.sourceterms import RickerWavelet

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0.,2.,1000)
RW = RickerWavelet(3., 1e-10)
mesh = dl.UnitSquareMesh(10,10)
V = dl.FunctionSpace(mesh, 'CG', 1)
obs = TimeObsPtwise({'V':V, 'Points':[]})
Rref = RW(times-0.5)
RR = []
dt = []
for ii in range(200):
    dt.append(-(ii-100)/200.)
    RR.append(RW(times-0.5-(ii-100)/200.))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(times, Rref, 'k')
ax1.plot(times, RR[0], 'r--')
ax1.plot(times, RR[-1], 'r--')
fig1.savefig('cycleskipping1.pdf')
dm = []
dm2 = []
for rr in RR:
    dm.append(obs.costfct(Rref, rr, times))
    dm2.append(np.exp(-obs.costfct(Rref, rr, times)**2/2))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(dt, dm, label='l2')
ax2.plot(dt, dm2, label='exp')
fig2.savefig('cycleskipping2.pdf')
ax2.legend()
plt.show()
