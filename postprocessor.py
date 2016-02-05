import numpy as np

from dolfin import norm, Function, interpolate

class PostProcessor():
    """
    Handles printing of results and stopping criteria for optimization
    """

    # Instantiation
    def __init__(self, meth, Vm, mtrue, maxnbLS=15):
        self.meth = meth
        if self.meth == 'Newt': self.Newt = True
        else:   self.Newt = False
        self.Vm = Vm
        self.mtrue = interpolate(mtrue, self.Vm)
        self.diff = Function(self.Vm)
        self.normmtrue = norm(self.mtrue)   # Note: this is a global value
        self.maxnbLS = maxnbLS
        self._createoutputlines()
        self.index = 0
        self.gradnorminit = None
        self.printrank = 0

    def setnormmtrue(self, normtrue):  self.normmtrue = normmtrue

    def _createoutputlines(self):
        self.titleline = ('{:2s} '+\
        '{:12s} {:12s} {:12s} \t'+\
        '{:10s} {:6s} \t'+\
        '{:12s} {:8s} {:6s} \t'+\
        '{:10s} {:3s} \t').format('it', \
        '    cost', '    datamis', '    regul', \
        '  Mmis-abs', '  rel', \
        '||grad||-abs', '   rel', '  G.p', \
        'LS-length', 'iter')
        if self.Newt:
            self.titleline = self.titleline + \
            '{:10s} {:5s} {:10s}'.format('  CG-tol', 'iter', 'finalres')

        self.dataline0 = '{:2d} {:12.5e} {:12.5e} {:12.5e} \t{:10.2e} {:6.3f}'
        self.dataline = '{:2d} '+\
        '{:12.5e} {:12.5e} {:12.5e} \t'+\
        '{:10.2e} {:6.3f} \t'+\
        '{:12.5e} {:8.2e} {:6.2f} \t'+\
        '{:10.3e} {:3d} \t\t'
        if self.Newt:
            self.dataline = self.dataline + '{:10.1e} {:3d} {:10.1e}'

    def errornorm(self, MM, m):
        self.diff.vector()[:] = (m.vector() - self.mtrue.vector()).array()
        return np.sqrt((MM*self.diff.vector()).inner(self.diff.vector()))

    def getResults0(self, Obj):
        """Get results before first step of iteration"""
        self.index = 0
        # Cost
        self.cost, self.misfit, self.regul= Obj.getcost()
        # Med Misfit
        self.medmisfit = self.errornorm(Obj.getMass(), Obj.getm())
        self.medmisfitrel = self.medmisfit/self.normmtrue
        # Grad
        self.gradnorm = 0.0
        self.gradnorminit = None

    def getResults(self, Obj, LSresults, CGresults, index=None):
        """LSresults = [LSsuccess, LScount, ratio]
        CGresults = [tolcg]"""
        if index == None:   self.index += 1
        else:   self.index = index

        if self.Newt and CGresults == None and self.index > 0:
            raise ValueError(\
            "CGresults must be provided when using Newton method")
        # Cost
        self.cost, self.misfit, self.regul= Obj.getcost()
        # Med Misfit
        self.medmisfit = self.errornorm(Obj.getMass(), Obj.getm())
        self.medmisfitrel = self.medmisfit/self.normmtrue
        # Grad
        self.gradnorm = Obj.getGradnorm()
        if (self.gradnorminit == None) and (self.gradnorm > 0.0):
            self.gradnorminit = self.gradnorm
        if not (self.gradnorminit == None):
            self.gradnormrel = self.gradnorm/self.gradnorminit
        if self.gradnorm > 1e-16:   
            srchdirnorm = Obj.getsrchdirnorm()
            if srchdirnorm > 1e-16:
                self.Gpangle = Obj.getgradxdir()/(self.gradnorm*srchdirnorm)
            else:   self.Gpangle = np.inf
        else:   self.Gpangle = np.inf
        # Line Search
        self.LSsuccess = LSresults[0]
        self.LScount = LSresults[1]
        self.LSratio = LSresults[2]
        # CG
        if self.Newt:
            self.CGiter = CGresults[0]
            self.finalnormCG = CGresults[1]
            self.tolCG = CGresults[3]

    def printResults(self, myrank):
        """Print results on process rank"""
        if self.printrank == myrank:
            if self.index == 0: 
                print self.titleline
                print self.dataline0.format(self.index, \
            self.cost, self.misfit, self.regul, \
            self.medmisfit, self.medmisfitrel)
            elif self.Newt:
                print self.dataline.format(self.index, \
            self.cost, self.misfit, self.regul, \
            self.medmisfit, self.medmisfitrel, \
            self.gradnorm, self.gradnormrel, self.Gpangle, \
            self.LSratio, self.LScount, \
            self.tolCG, self.CGiter, self.finalnormCG)
            else:
                print self.dataline.format(self.index, \
            self.cost, self.misfit, self.regul, \
            self.medmisfit, self.medmisfitrel, \
            self.gradnorm, self.gradnormrel, self.Gpangle, \
            self.LSratio, self.LScount)

    def Stop(self, myrank):
        """Compute stopping criterion and 
        return True if iteration must stop"""
        if not self.LSsuccess:
            if self.printrank == myrank:
                print 'Line Search failed after {0} counts'.format(self.LScount)
            return True
        elif self.gradnormrel < 1e-10:
            if self.printrank == myrank: print 'Optimization converged!'
            return True
        else:   return False

    def alpha_init(self):
        """Evaluate initial length for line search in next iteration"""
        if self.Newt:   return 1.0
        else:
            alpha = self.LSratio
            if self.LScount == 1:    return 10.*alpha
            elif self.LScount < 5:   return 4.*alpha
            else:   return alpha
