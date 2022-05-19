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

class TemperatureGradients(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, data_prefix):
        super(TemperatureGradients, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf  

        pp = self.getRAdata(eht, 'pp')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        mu = self.getRAdata(eht, 'abar')[intc]
        chim = self.getRAdata(eht, 'chim')[intc]
        chit = self.getRAdata(eht, 'chit')[intc]
        gamma2 = self.getRAdata(eht, 'gamma2')[intc]
        # print(chim,chit,gamma2)

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma2 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients		
        nabla = self.deriv(lntt, lnpp)
        nabla_ad = (gamma2 - 1.) / gamma2

        if ieos == 1:
            nabla_mu = np.zeros(nx)
        else:
            nabla_mu = (chim / chit) * self.deriv(lnmu, lnpp)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nx = nx
        self.nabla = nabla
        self.nabla_ad = nabla_ad
        self.nabla_mu = nabla_mu
        self.ig = ig
        self.ieos = ieos
        self.fext = fext
        self.abar = self.getRAdata(eht, 'abar')[intc]

    def plot_nablas(self, laxis, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
        """Plot temperature gradients in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureGradients.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.nabla
        plt2 = self.nabla_ad
        plt3 = self.nabla_mu

        # create FIGURE
        #plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3]
        self.set_plt_axis(laxis, xbl/xsc, xbr/xsc, ybu/ysc, ybd/ysc, [yval / ysc for yval in to_plot])

        # plot DATA 
        # plt.title('temperature gradients')

        if self.ig == 1:
            plt.plot(grd1/xsc, plt1/ysc, color='brown', label=r'nabla')
            plt.plot(grd1/xsc, plt2/ysc, color='red', label=r'nablaad')
            if self.ieos == 3:
                plt.plot(grd1/xsc, plt3/ysc, color='green', label=r'nablamu')
        elif self.ig == 2:
            plt.plot(grd1/xsc, plt1/ysc, color='brown', label=r'nabla')
            plt.plot(grd1/xsc, plt2/ysc, color='red', label=r'nablaad')
            if self.ieos == 3:
                plt.plot(grd1/xsc, plt3/ysc, color='green', label=r'nablamu')

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        #plt.axvline(super_ad_i, linestyle=':', linewidth=0.7, color='k')
        #plt.axvline(super_ad_o, linestyle=':', linewidth=0.7, color='k')


        ycol = np.linspace(ybu / ysc, ybd / ysc, num=100)
        for i in ycol:
            plt.text(bconv/xsc,i, '.',dict(size=15))
            plt.text(tconv/xsc,i, '.',dict(size=15))

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)

        setylabel = r'(' + "{:.0e}".format(ysc) + ')'
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        #plt.show(block=False)

        # save PLOT
        #if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_nablas.png')
        #if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_nablas.eps')

