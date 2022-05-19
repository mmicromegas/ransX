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

class NuclearEnergyProduction(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(NuclearEnergyProduction, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        enuc = self.getRAdata(eht, 'enuc1')[intc] + self.getRAdata(eht, 'enuc2')[intc]
        # enuc = self.getRAdata(eht, 'enuc1')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uzuz')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]

        urms = (uxux + uyuy + uzuz)**0.5

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.enuc = enuc
        self.dd = dd
        self.ig = ig
        self.urms = urms

    def plot_enuc(self, laxis, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
        """Plot nuclear energy production stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(NuclearEnergyProduction.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.enuc

        # create FIGURE
        #plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        # ybu = 1.e16 # for oburn semilogy
        # ybd = 1.e3 # for oburn semilogy
        self.set_plt_axis(laxis, xbl/xsc, xbr/xsc, ybu/ysc, ybd/ysc, [yval / ysc for yval in to_plot])

        # plot DATA 
        # plt.title('Nuclear energy production')
        plt.title('Heating (source term)')
        #plt.semilogy(grd1,plt1,color='brown',label = r'$\overline{\varepsilon_{nuc}}$')
        #plt.yscale('symlog')
        plt.plot(grd1/xsc, plt1/ysc, color='brown', label=r'+enuc')
        #plt.semilogy(grd1/xsc, plt1/ysc, color='brown', label=r'+enuc')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)

        setylabel = r'erg/g/s (' + "{:.0e}".format(ysc) + ')'
        plt.ylabel(setylabel)

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        ycol = np.linspace(ybu / ysc, ybd / ysc, num=100)
        for i in ycol:
            plt.text(bconv/xsc,i, '.',dict(size=15))
            plt.text(tconv/xsc,i, '.',dict(size=15))

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        #plt.show(block=False)

        # save PLOT
        #plt.savefig('RESULTS/' + self.data_prefix + 'mean_enuc.png')
        #plt.savefig('RESULTS/' + self.data_prefix + 'mean_enuc.eps')

    def plot_enuc_per_volume(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot nuclear energy production stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(NuclearEnergyProduction.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.enuc * self.dd

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        # plt.title('Nuclear energy production')
        plt.title('Heating (source term)')
        # plt.semilogy(grd1,plt1,color='brown',label = r'$\overline{\varepsilon_{nuc}}$')
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{\varepsilon_{nuc}}$')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r'log $\overline{\varepsilon_{enuc}}$ (erg cm$^{-3}$ s$^{-1}$)'
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r'log $\overline{\varepsilon_{enuc}}$ (erg cm$^{-3}$ s$^{-1}$)'
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_enuc_pervolume.png')

    def plot_enuc2(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot nuclear energy production stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(NuclearEnergyProduction.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        to_plt1 = self.enuc
        to_plt3 = self.urms

        # load x GRID
        grd1 = self.xzn0

        if self.ig == 1:
            xlabel_1 = r'x (cm)'
        elif self.ig == 2:
            xlabel_1 = r'r (cm)'

        ylabel_1 = r'$\overline{\varepsilon_{enuc}}$ (erg g$^{-1}$ s$^{-1}$)'
        ylabel_3 = r"$u_{rms}$ (cm s$^{-1})$"

        plabel_1 = r'$\overline{\varepsilon_{enuc}}$'
        plabel_3 = r'$u_{rms}$'

        # calculate indices of grid boundaries
        xzn0 = np.asarray(self.xzn0)
        xlm = np.abs(xzn0 - xbl)
        xrm = np.abs(xzn0 - xbr)
        idxl = int(np.where(xlm == xlm.min())[0][0])
        idxr = int(np.where(xrm == xrm.min())[0][0])

        # create FIGURE
        fig, ax1 = plt.subplots(figsize=(7, 6))

        ax1.axis([xbl, xbr, ybd, ybu])
        ax1.plot(xzn0, to_plt1, color='r', label=plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7, prop={'size': 18})

        ax2 = ax1.twinx()
        ax2.axis([xbl, xbr, -0.5e6, 2.e7])
        ax2.plot(xzn0, to_plt3, color='m', label=plabel_3)
        ax2.set_ylabel(ylabel_3)
        ax2.tick_params('y')
        ax2.legend(loc=1, prop={'size': 18})

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_enuc2.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_enuc2.eps')

