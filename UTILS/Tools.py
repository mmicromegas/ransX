import numpy as np


# class for tools

class Tools:

    def __init__(self):
        pass

    def getRAdata(self, ransdatarray, q):
        quantity = np.asarray(ransdatarray.item().get(q))
        return quantity

    def errorGeometry(self,ig):
        return " Geometry ig = " + str(ig) + " not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ..."
