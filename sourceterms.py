import numpy as np

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



#TODO: To be tested
class RickerWavelet():
    """ Create function for Ricker wavelet """

    def __init__(self, peak_freq, cut_off=-16):
        self.f = peak_freq
        self.t_peak = np.sqrt(-np.log(10**(cut_off)))/(np.pi*self.f)


    def __call__(self, tt):
        """ Overload () operator """
        TT = tt - self.t_peak
        return (1.0-2.0*(np.pi**2)*(self.f**2)*(TT**2))*\
        np.exp(-(np.pi**2)*(self.f**2)*(TT**2))
