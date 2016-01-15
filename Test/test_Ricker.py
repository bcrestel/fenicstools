import numpy as np
import matplotlib.pyplot as plt

from fenicstools.sourceterms import RickerWavelet

cut_off = 1e-10

R1 = RickerWavelet(0.4, cut_off)
T = 10.
fig1 = R1.plot(np.linspace(0.0, T, 1000), np.linspace(-2.5, 2.5, 1000))
fig1.savefig('Plots/Ricker1.eps')
print R1(0.0), R1(T)

R2 = RickerWavelet(4.0, cut_off)
T = 1.0
fig2 = R2.plot(np.linspace(0.0, T, 1000), np.linspace(-20.0, 20.0, 1000))
fig2.savefig('Plots/Ricker2.eps')
print R2(0.0), R2(T)

R3 = RickerWavelet(10.0, cut_off)
T = 0.4
fig3 = R3.plot(np.linspace(0.0, T, 1000), np.linspace(-50.0, 50.0, 1000))
fig3.savefig('Plots/Ricker3.eps')
print R3(0.0), R3(T)
