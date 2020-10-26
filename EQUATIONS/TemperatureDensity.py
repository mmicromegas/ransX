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

class TemperatureDensity(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(TemperatureDensity, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = self.getRAdata(eht, 'dd')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dd = dd
        self.tt = tt
        self.ig = ig

    def plot_ttdd(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot temperature and density stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureDensity.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        to_plt1 = np.log10(self.tt)
        to_plt2 = np.log10(self.dd)

        if self.ig == 1:
            xlabel_1 = r'x (cm)'
        elif self.ig == 2:
            xlabel_1 = r'r (cm)'

        ylabel_1 = r'log $\overline{T}$ (K)'
        ylabel_2 = r'log $\overline{\rho}$ (g cm$^{-3}$)'

        plabel_1 = r'$\overline{T}$'
        plabel_2 = r'$\overline{\rho}$'

        # calculate indices of grid boundaries 
        xzn0 = np.asarray(self.xzn0)
        xlm = np.abs(xzn0 - xbl)
        xrm = np.abs(xzn0 - xbr)
        idxl = int(np.where(xlm == xlm.min())[0][0])
        idxr = int(np.where(xrm == xrm.min())[0][0])

        # create FIGURE	
        fig, ax1 = plt.subplots(figsize=(7, 6))

        ax1.axis([xbl, xbr, np.min(to_plt1[idxl:idxr]), np.max(to_plt1[idxl:idxr])])
        ax1.plot(xzn0, to_plt1, color='r', label=plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7, prop={'size': 18})

        ax2 = ax1.twinx()
        ax2.axis([xbl, xbr, np.min(to_plt2[idxl:idxr]), np.max(to_plt2[idxl:idxr])])
        ax2.plot(xzn0, to_plt2, color='b', label=plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1, prop={'size': 18})

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_ttdd.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_ttdd.eps')
