import numpy as np
import matplotlib.pyplot as plt

from fenicstools.sourceterms import RickerWavelet

#cut_off = 1e-10
cut_off = 1e-6
Rnb = 1

R1 = RickerWavelet(0.4, cut_off)
T = 10.
fig = R1.plot(np.linspace(0.0, T, 1000), np.linspace(-2.5, 2.5, 1000))
fig.savefig('Plots/Ricker'+str(Rnb)+'.eps')
Rnb += 1
print R1(0.0), R1(T)

R2 = RickerWavelet(1.0, cut_off)
T = 5.0
fig = R2.plot(np.linspace(0.0, T, 1000), np.linspace(-5.0, 5.0, 1000))
fig.savefig('Plots/Ricker'+str(Rnb)+'.eps')
Rnb += 1
print R2(0.0), R2(T)

R2 = RickerWavelet(2.0, cut_off)
T = 2.0
fig = R2.plot(np.linspace(0.0, T, 1000), np.linspace(-10.0, 10.0, 1000))
fig.savefig('Plots/Ricker'+str(Rnb)+'.eps')
Rnb += 1
print R2(0.0), R2(T)

R2 = RickerWavelet(4.0, cut_off)
T = 1.0
fig = R2.plot(np.linspace(0.0, T, 1000), np.linspace(-20.0, 20.0, 1000))
fig.savefig('Plots/Ricker'+str(Rnb)+'.eps')
Rnb += 1
print R2(0.0), R2(T)

R3 = RickerWavelet(10.0, cut_off)
T = 0.4
fig = R3.plot(np.linspace(0.0, T, 1000), np.linspace(-50.0, 50.0, 1000))
fig.savefig('Plots/Ricker'+str(Rnb)+'.eps')
Rnb += 1
print R3(0.0), R3(T)
