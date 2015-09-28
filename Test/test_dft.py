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
ax.set_xlim(-5., 5.)


"""
We look at the Fourier transform of the function sine. We don't compute Fourier
series so we can't expect to get a single frequency but if we look at sine over
several periods, we should get one dominant frequency. We check we recover
that with the DFT.
"""

tt = np.linspace(0., 10., 1000)
f = 2.0
bb = np.sin(2.*np.pi*f*tt)
ff = np.fft.fft(bb)
ffn = ff.real**2 + ff.imag**2
ffxi = np.fft.fftfreq(len(tt), d = tt[1]-tt[0])
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(np.fft.fftshift(ffxi), np.fft.fftshift(ffn))
ax2.plot([f,f],[0.,ffn.max()],'k--')
ax2.plot([-f,-f],[0.,ffn.max()],'k--')
ax2.set_xlim(-5., 5.)
fig2.suptitle('f = {} Hz'.format(f))


plt.show(block=True)

