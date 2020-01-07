import numpy as np


# class for tools

class Tools:

    def __init__(self):
        pass

    def getRAdata(self, ransdatarray, q):
        quantity = np.asarray(ransdatarray.item().get(q))
        return quantity

