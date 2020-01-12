import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TemperatureGradients(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, data_prefix):
        super(TemperatureGradients, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht,'xzn0')
        nx = self.getRAdata(eht,'nx')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf  

        pp = self.getRAdata(eht,'pp')[intc]
        tt = self.getRAdata(eht,'tt')[intc]
        mu = self.getRAdata(eht,'abar')[intc]
        chim = self.getRAdata(eht,'chim')[intc]
        chit = self.getRAdata(eht,'chit')[intc]
        gamma2 = self.getRAdata(eht,'gamma2')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if (ieos == 1):
            cp = self.getRAdata(eht,'cp')[intc]
            cv = self.getRAdata(eht,'cv')[intc]
            gamma2 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients		
        nabla = self.deriv(lntt, lnpp)
        nabla_ad = (gamma2 - 1.) / gamma2

        if (ieos == 1):
            nabla_mu = np.zeros(nx)
        else:
            nabla_mu = (chim / chit) * self.deriv(lnmu, lnpp)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nabla = nabla
        self.nabla_ad = nabla_ad
        self.nabla_mu = nabla_mu
        self.ig = ig
        self.ieos = ieos

    def plot_nablas(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot temperature gradients in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.nabla
        plt2 = self.nabla_ad
        plt3 = self.nabla_mu

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('temperature gradients')

        if (self.ig == 1):
            plt.plot(grd1, plt1, color='brown', label=r'$\nabla$')
            plt.plot(grd1, plt2, color='red', label=r'$\nabla_{ad}$')
            if (self.ieos == 3):
                plt.plot(grd1, plt3, color='green', label=r'$\nabla_{\mu}$')
            # define x LABEL
            setxlabel = r"x (cm)"
        elif (self.ig == 2):
            plt.plot(grd1, plt1, color='brown', label=r'$\nabla$')
            plt.plot(grd1, plt2, color='red', label=r'$\nabla_{ad}$')
            if (self.ieos == 3):
                plt.plot(grd1, plt3, color='green', label=r'$\nabla_{\mu}$')
            # define x LABEL
            setxlabel = r"r (cm)"
        else:
            print(
                "ERROR (TemperatureGradients.py): geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        # define y LABEL
        setylabel = r"$\nabla$"

        # show x/y labels
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_nablas.png')
