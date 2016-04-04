import math
import numpy as np

import dolfin as dl
from fenicstools.imaging import ObjectiveImageDenoising

# Target data:
data = np.loadtxt('image.dat', delimiter=',')
Lx, Ly = float(data.shape[1])/float(data.shape[0]), 1.
class Image(dl.Expression):
    def __init__(self, Lx, Ly, data):
        self.data = data
        self.hx = Lx/float(self.data.shape[1]-1)
        self.hy = Ly/float(self.data.shape[0]-1)

    def eval(self, values, x):
        j = math.floor(x[0]/self.hx)
        i = math.floor(x[1]/self.hy)
        values[0] = self.data[i,j]
trueImage = Image(Lx, Ly, data)

mesh = dl.RectangleMesh(dl.Point(0.,0.), dl.Point(Lx,Ly), 200, 100)

denoise = ObjectiveImageDenoising(mesh, trueImage, 'TV')
denoise.plot(0)
denoise.generatedata(0.6)
denoise.plot(1)
