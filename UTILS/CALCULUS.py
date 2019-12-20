import numpy as np
import sys


# class for calculus functions

class CALCULUS:

    def __init__(self, ig):
        # geometry 
        self.ig = ig

    def deriv(self, f, x):
        """ Compute numerical derivation using 3-point Lagrangian """
        """ interpolation (inspired by IDL function deriv) """
        """ Procedure Hildebrand, Introduction to Numerical Analysis, """
        """ Mc Graw Hill, 1956 """
        """ df/dx = f0*(2x-x1-x2)/(x01*x02)+f1*(x-x0-x2)/(x10*x12)+ """
        """ f2*(2x-x0-x1)/(x20*x21) """
        """ Where: x01 = x0-x1, x02 = x0-x2, x12 = x1-x2, etc. """

        x12 = x - np.roll(x, -1)  # x1 - x2
        x01 = np.roll(x, 1) - x  # x0 - x1
        x02 = np.roll(x, 1) - np.roll(x, -1)  # x0 - x2

        deriv = np.zeros(f.size)

        #       middle points
        deriv = np.roll(f, 1) * (x12 / (x01 * x02)) + f * (1. / x12 - 1. / x01) - np.roll(f, -1) * (x01 / (x02 * x12))

        #       first point
        deriv[0] = f[0] * (x01[1] + x02[1]) / (x01[1] * x02[1]) - f[1] * x02[1] / (x01[1] * x12[1]) + f[2] * x01[1] / (
                    x02[1] * x12[1])

        #       last point
        n = x.size
        n2 = x.size - 1 - 2
        deriv[x.size - 1] = -f[n - 3] * x12[n2] / (x01[n2] * x02[n2]) + f[n - 2] * x02[n2] / (x01[n2] * x12[n2]) - f[
            n - 1] * (x02[n2] + x12[n2]) / (x02[n2] * x12[n2])

        return deriv

    def Div(self, f, rc):
        """Compute the divergence of 'f'"""

        divf = np.zeros(f.shape)

        if (self.ig == 2):
            f = f * rc ** 2
            divf = self.deriv(f, rc) / rc ** 2
        elif (self.ig == 1):
            divf = self.deriv(f, rc)
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        return divf

    def Grad(self, q, rc):
        """Compute gradient"""
        grad = np.zeros(q.shape)
        grad = self.deriv(q, rc)
        return grad

    def dt(self, q, rc, timec, tt):

        # rc = self.xzn0
        # timec = self.timec

        tmp = np.zeros(q.shape)
        dt = np.zeros(rc.shape)

        for i in range(0, rc.size): tmp[:, i] = self.deriv(q[:, i], timec)
        dt[:] = tmp[tt, :]
        return dt
