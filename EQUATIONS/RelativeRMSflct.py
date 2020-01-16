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

class RelativeRMSflct(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, data_prefix):
        super(RelativeRMSflct, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf 

        dd = self.getRAdata(eht, 'dd')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        ss = self.getRAdata(eht, 'ss')[intc]
        abar = self.getRAdata(eht, 'abar')[intc]
        zbar = self.getRAdata(eht, 'zbar')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        sound = self.getRAdata(eht, 'sound')[intc]

        ddsq = self.getRAdata(eht, 'ddsq')[intc]
        ttsq = self.getRAdata(eht, 'ttsq')[intc]
        ppsq = self.getRAdata(eht, 'ppsq')[intc]
        sssq = self.getRAdata(eht, 'sssq')[intc]
        abarsq = self.getRAdata(eht, 'abarsq')[intc]
        zbarsq = self.getRAdata(eht, 'zbarsq')[intc]

        self.eht_ddrms = ((ddsq - dd * dd) ** 0.5) / dd
        self.eht_ttrms = ((ttsq - tt * tt) ** 0.5) / tt
        self.eht_pprms = ((ppsq - pp * pp) ** 0.5) / pp
        self.eht_ssrms = ((sssq - ss * ss) ** 0.5) / ss
        self.eht_abarrms = ((np.abs(abarsq - abar * abar)) ** 0.5) / abar
        self.eht_zbarrms = ((np.abs(zbarsq - zbar * zbar)) ** 0.5) / zbar

        # for ideal gas eos
        if (ieos == 1):
            gammac = self.getRAdata(eht, 'gammac')[intc]
            sound = (gammac * pp / dd) ** (0.5)

        self.ms2 = uxux / sound ** 2.  # mach number squared

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ig = ig

    def plot_relative_rms_flct(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot relative rms fluctuations in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.eht_ddrms
        plt2 = self.eht_ttrms
        plt3 = self.eht_pprms
        plt4 = self.ms2
        plt5 = self.eht_ssrms
        plt6 = self.eht_abarrms
        plt7 = self.eht_zbarrms
        plt8 = self.ms2 ** 0.5

        print("OUTPUT(RelativeRMSflct.py): Max Mach Number: ", np.around(max(plt8), decimals=3))

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('relative rms fluctuations')
        plt.semilogy(grd1, plt1, color='brown', label=r"$\rho$")
        plt.semilogy(grd1, plt2, color='r', label=r"$T$")
        plt.semilogy(grd1, plt3, color='g', label=r"$P$")
        plt.semilogy(grd1, plt8, color='purple', label=r"$M_s$ (Mach)")
        #plt.semilogy(grd1, plt4, color='b', label=r"$M_s^2 = u_r^2/c_s^2$")
        # plt.semilogy(grd1,plt5,color='m',label = r"$S$")
        # plt.semilogy(grd1,plt6,color='k',linestyle='--',label = r"$\overline{A}$")
        # plt.semilogy(grd1,plt7,color='c',linestyle='--',label = r"$\overline{Z}$")

        if (self.ig == 1):
            setxlabel = r"x (cm)"
        elif (self.ig == 2):
            setxlabel = r"r (cm)"
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        # define y LABEL
        setylabel = r"$q'_{rms} \ / \ \overline{q}$"

        # show x/y LABELS
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_rel_rms_fluctuations.png')
