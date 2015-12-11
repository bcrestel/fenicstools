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
            bs[:] = self.PointSourcecorrection(bs)
            self.PtSrc.append(bs)


    def PointSourcecorrection(self, b):
        """ Fix dolfin's PointSource in parallel """
        # TODO: TO BE TESTED!!
        scale = b.array().sum()
        if abs(scale) > 1e-12:  
            return b.array()/scale
        else:   return b.array()
        

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
            bs[:] = self.PointSourcecorrection(bs)
            self.PtSrc.append(bs)


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




class TimeFilter():
    """ Create time filter to fade out data misfit (hence src term in adj eqn) """

    def __init__(self, t0, t1, t2, T):
        """ Inputs:
            t0 = initial time
            t1 = beginning of flat section at 1.0
            t2 = end of flat section at 1.0
            T = final time """
        self.t0 = t0
        self.t1 = t1
        self.t1b = 2*t1-t0
        self.t2b = 2*t2-T
        self.t2 = t2
        self.T = T

    def __call__(self, tt):
        """ Overload () operator """
        if tt.__class__ == np.ndarray:
            assert np.min(tt) >= self.t0 and np.max(tt) <= self.T, "Input tt out of bounds [t0, T]"
            output = np.zeros(len(tt))
            # [t0,t1]
            indicest1 = np.intersect1d(np.where(tt > self.t0+1e-16)[0], \
            np.where(tt <= self.t1)[0])
            output[indicest1] = np.exp(-1./((tt[indicest1]-self.t0)*\
            (self.t1b-tt[indicest1]))) / np.exp(-1./(self.t1-self.t0)**2)
            # [t2,T]
            indicest2 = np.intersect1d(np.where(tt >= self.t2)[0], \
            np.where(tt < self.T-1e-16)[0])
            output[indicest2] = np.exp(-1./((tt[indicest2]-self.t2b)*\
            (self.T-tt[indicest2]))) / np.exp(-1./(self.T-self.t2)**2)
            #[t1,t2]
            indicesmid = np.intersect1d(np.where(tt > self.t1)[0], \
            np.where(tt < self.t2)[0])
            output[indicesmid] = 1.0
            return output
        else:
            assert tt >= self.t0 and tt <= self.T, "Input tt out of bounds [t0, T]"
            if tt <= self.t0 + 1e-16: return 0.0
            if tt >= self.T - 1e-16:    return 0.0
            if tt <= self.t1:   
                return np.exp(-1./((tt-self.t0)*(self.t1b-tt)))/np.exp(-1./(self.t1-self.t0)**2)
            if tt >= self.t2:   
                return np.exp(-1./((tt-self.t2b)*(self.T-tt)))/np.exp(-1./(self.T-self.t2)**2)
            return 1.0

    def plot(self, ndt=1000):
        """ Plot the shape of the filter along with its fft """
        tt = np.linspace(self.t0, self.T, ndt)
        xx = self.__call__(tt)
        ff = np.fft.fft(xx) # fft(time-domain)
        ffn = np.sqrt(ff.real**2 + ff.imag**2)
        ffxi = np.fft.fftfreq(len(xx), d=tt[1]-tt[0])
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(tt, xx)
        ax2 = fig.add_subplot(122)
        ax2.plot(np.fft.fftshift(ffxi), np.fft.fftshift(ffn))
        return fig
