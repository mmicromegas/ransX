import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Gravity(calc.Calculus, al.SetAxisLimit, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(Gravity, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0'))

        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        grav = np.asarray(eht.item().get('gg')[intc])

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ig = ig
        self.grav = grav

    def plot_grav(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot gravity in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.grav

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Gravitational Acceleration')
        if (self.ig == 1):
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{g}_x$')
        elif (self.ig == 2):
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{g}_x$')
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

            # define and show x/y LABELS
        if (self.ig == 1):
            setxlabel = r'x (10$^{8}$ cm)'
        elif (self.ig == 2):
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        setylabel = r'$\overline{g}_x$ (cm s$^{-2}$)'

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_grav.png')
