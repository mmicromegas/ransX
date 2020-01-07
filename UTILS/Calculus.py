import numpy as np
import sys 
import UTILS.Errors as eR


# class for calculus functions

class Calculus(eR.Errors, object):

    def __init__(self, ig):
        super(Calculus)

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

        if self.ig == 2:
            f = f * rc ** 2
            divf = self.deriv(f, rc) / rc ** 2
        elif self.ig == 1:
            divf = self.deriv(f, rc)
        else:
            print("ERROR(Calculus.py):" + self.errorGeometry(self.ig))
            sys.exit()

        return divf

    def Grad(self, q, rc):
        """Compute gradient"""
        grad = self.deriv(q, rc)
        return grad

    def dt(self, q, rc, timec, tt):
        """Compute time derivative"""

        tmp = np.zeros(q.shape)
        dt = np.zeros(rc.shape)

        for i in range(0, rc.size):
            tmp[:, i] = self.deriv(q[:, i], timec)

        dt[:] = tmp[tt, :]
        return dt
