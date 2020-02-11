import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class NuclearEnergyProduction(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(NuclearEnergyProduction, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        enuc = self.getRAdata(eht, 'enuc1')[intc] + self.getRAdata(eht, 'enuc2')[intc]

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.enuc = enuc
        self.dd = dd
        self.ig = ig

    def plot_enuc(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
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
            setylabel = r'log $\overline{\varepsilon_{enuc}}$ (erg g$^{-1}$ s$^{-1}$)'
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r'log $\overline{\varepsilon_{enuc}}$ (erg g$^{-1}$ s$^{-1}$)'
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_enuc.png')

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
