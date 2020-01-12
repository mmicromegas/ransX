import numpy as np


# class for tools

class Tools:

    def __init__(self):
        pass

    def getRAdata(self, ransdat, q):
        quantity = np.asarray(ransdat.item().get(q))
        return quantity

