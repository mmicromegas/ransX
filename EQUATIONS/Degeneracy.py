import numpy as np
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Degeneracy(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(Degeneracy, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        psi = self.getRAdata(eht, 'psi')[intc]

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.psi = psi

    def plot_degeneracy(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot degeneracy parameter in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(Degeneracy.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.psi

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('degeneracy parameter')
        plt.plot(grd1, plt1, color='brown', label=r'$\psi$')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\psi$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\psi$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_degeneracy.png')
