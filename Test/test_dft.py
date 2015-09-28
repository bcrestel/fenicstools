import numpy as np
import matplotlib.pyplot as plt


"""
Identity function for interval [-.5,.5] has Fourier transform equal to the sinc
function (sin(pi x)/(pi x)) when using the Fourier transform defined in
Osgood's. This is slightly different when using Arbogast definition.
This script checks that the DFT matches the definition in Osgood.
"""

N = 1000
tt = np.linspace(-10,10,N)
Dt = tt[1]-tt[0]
func = lambda t: t>-0.5 and t<0.5
fv = np.vectorize(func)
bb = fv(tt)

ff = np.fft.fft(bb)
ffxi = np.fft.fftfreq(N, d=Dt)
print ffxi[1], 1./(N*Dt)
ffn = ff.real**2 + ff.imag**2
ffn = ffn/ffn.max()

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(np.fft.fftshift(ffxi), np.fft.fftshift(ffn))
xx = np.linspace(-5,5,100)
fsinc = np.sinc(xx)
ax.plot(xx, fsinc**2)
ax.set_xlim(-5,5)

plt.show(block=True)

