from fenicstools.observationoperator import TimeObsPtwise
from fenicstools.sourceterms import RickerWavelet

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt


tt = np.linspace(-1., 16., 1000)
dt = tt[1]-tt[0]
k = np.array([[(i-j)*dt for i in range(1000)] for j in range(1000)])
sigma = 2.0
K = np.exp(-k**2/(2.*sigma**2))

# test case for K:
time = np.linspace(0.0, 1.0, 1000)
noisy = np.sin(2*np.pi*time) + 0.1*np.sin(20*np.pi*time)
noisy = noisy/np.max(np.abs(noisy))
Knoisy = K.dot(noisy)
Knoisy = Knoisy/np.max(np.abs(Knoisy))
figK = plt.figure()
ax = figK.add_subplot(111)
ax.plot(time, noisy, label='noisy')
ax.plot(time, Knoisy, label='K*noisy')
ax.legend()


def misfit(tt, s1, s2):
    dt = tt[1] - tt[0]
    ds = s2 - s1
    factors = np.ones(tt.shape)
    factors[0], factors[-1] = 0.5, 0.5
    return dt*(factors*(ds**2)).sum()

def misfitK(tt, s1, s2):
    dt = tt[1] - tt[0]
    ds = s2 - s1
    return dt*((K.dot(ds)).dot(ds))

def shifted(tt, t0):
    return (tt>=t0)*(tt<=t0+5.)*np.sin(2.*np.pi*(tt-t0))

true = shifted(tt, 5.)
t1 = shifted(tt, 2.5)

MISFIT = []
MISFITK = []
for t0 in tt:
    if t0 > 11.: break
    signal = shifted(tt, t0)
    MISFIT.append(misfit(tt, true, signal))
    MISFITK.append(misfitK(tt, true, signal))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt, true)
ax.plot(tt, t1)

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(MISFIT)
ax.plot(MISFITK)

plt.show()

