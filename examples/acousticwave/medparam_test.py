import dolfin as dl

def targetmediumparameters(Vl, X, myplot=None):
    c = 50.0
    r = 1.0
    l = r*c*c
    a = 1.0/l
    b = 1.0/r
    return a, b, c, l, r


def initmediumparameters(Vl, X, myplot=None):
    return targetmediumparameters(Vl, X)


def loadparameters(LARGE):
    return 20, 1.0e-4, 50.0, 0.0, 0.0, 0.0, 0.1
