import numpy as np
import matplotlib.pyplot as plt

from dolfin import TestFunction, Constant, dx, assemble, PointSource, Point


class PointSources():
    """ Create point source at different locations """

    def __init__(self, V, src_loc):
        """ Inputs:
        V = FunctionSpace
        src_loc = iterable that returns coordinates of the point """
        self.V = V
        self.src_loc = src_loc
        test = TestFunction(self.V)
        f = Constant('0')
        L = f*test*dx
        self.b = assemble(L)
        self.PtSrc = []
        for pts in self.src_loc:
            delta = PointSource(self.V, self.list2point(pts))
            bs = self.b.copy()
            delta.apply(bs)
            self.PtSrc.append(self._PointSourcecorrection(bs))


    def _PointSourcecorrection(self, b):
        """ Fix dolfin's PointSource in parallel """
        scale = b.sum()
        if abs(scale - 1.0) > 1e-6:
            return b/scale
        else:
            return b
        

    def list2point(self, list_in):
        """ Turn a list of coord into a Fenics Point
        Inputs:        
        list_in = list containing coordinates of the Point """
        dim = np.size(list_in)
        return Point(dim, np.array(list_in, dtype=float))


    def add_points(self, src_loc):
        """ Create and add more point sources """
        for pts in self.src_loc:
            self.src_loc.append(pts)
            delta = PointSource(self.V, self.list2point(pts))
            bs = self.b.copy()
            delta.apply(bs)
            self.PtSrc.append(self._PointSourcecorrection(bs))


    def __getitem__(self, index):
        """ Overload [] operator """
        return self.PtSrc[index]



class RickerWavelet():
    """ Create function for Ricker wavelet """

    def __init__(self, peak_freq, cut_off=1e-16):
        self.f = peak_freq  # Must be in Hz
        self.t_peak = np.sqrt(-np.log(cut_off))/(np.pi*self.f)


    def __call__(self, tt):
        """ Overload () operator """
        TT = tt - self.t_peak
        return (1.0 - 2.0*np.pi**2*self.f**2*TT**2)*\
        np.exp(-np.pi**2*self.f**2*TT**2)


    def freq(self, xi):
        """ Frequency content (using Osgood def, i.e., in Hz) """
        return 2.0*xi**2*np.exp(-xi**2/self.f**2)/(np.sqrt(np.pi)*self.f**3)


    def plot(self, tt, xi):
        """ Plot Ricker Wavelet along with frequency content """
        Rw = self.__call__(tt)  # time-domain
        Rwf = self.freq(xi) # exact Fourier transf
        ff = np.fft.fft(Rw) # fft(time-domain)
        ffn = np.sqrt(ff.real**2 + ff.imag**2)
        ffxi = np.fft.fftfreq(len(Rw), d=tt[1]-tt[0])
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(tt, Rw)
        ax2 = fig.add_subplot(122)
        ax2.plot(xi, Rwf, label='ex')
        ax2.plot(np.fft.fftshift(ffxi), np.fft.fftshift(ffn)/ffn.max()*Rwf.max(), label='dft')
        ax2.set_xlim(xi.min(), xi.max())
        ax2.legend()
        fig.suptitle('{} Hz'.format(self.f))
        return fig




