# class for RANS KineticEnergyFlux #

import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class KineticEnergyFlux(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, data_prefix):
        super(KineticEnergyFlux, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]
        ddekux = self.getRAdata(eht, 'ddekux')[intc]

        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]

        uyux = self.getRAdata(eht, 'uxuy')[intc]
        uzux = self.getRAdata(eht, 'uxuz')[intc]

        uxuxux = self.getRAdata(eht, 'uxuxux')[intc]
        uyuyux = self.getRAdata(eht, 'uyuyux')[intc]
        uzuzux = self.getRAdata(eht, 'uzuzux')[intc]

        mm = self.getRAdata(eht, 'mm')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ek = 0.5 * (dduxux + dduyuy + dduzuz) / dd

        #####################
        # KINETIC ENERGY FLUX
        #####################

        fekx = ddekux - dd * fht_ek * fht_ux

        #fekx2 = dd*self.thirdOrder(eht,intc,"ux","ux","ux") + \
        #        dd*self.thirdOrder(eht,intc,"uy","uy","ux") + \
        #        dd*self.thirdOrder(eht,intc,"uz","uz","ux")

        fekx2 = dd*(uxuxux - ux*uxux - ux*uxux - ux*uxux + ux*ux*ux) + \
                dd*(uyuyux - uy*uyux - uy*uyux - ux*uyuy + uy*uy*ux) + \
                dd*(uzuzux - uz*uzux - uz*uzux - ux*uzuz + uz*uz*ux)


        #########################
        # END KINETIC ENERGY FLUX
        #########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dd = dd
        self.fht_ek = fht_ek
        self.fekx = fekx
        self.fekx2 = fekx2
        self.fext = fext

    def plot_keflx(self, laxis, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot rho stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(KineticEnergyFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fekx
        plt2 = self.fekx2

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2]
        self.set_plt_axis(laxis, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('turbulent kinetic energy flux')
        plt.plot(grd1, plt1, color='brown', label=r"$\overline{\rho e''_k u''_x}$")
        plt.plot(grd1, plt2, color='red', linestyle='--',label=r"$\overline{\rho} \ \overline{u'_i u'_i u'_x}$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$F_K$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$F_K$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # check supported file output extension
        if self.fext != "png" and self.fext != "eps":
            print("ERROR(KineticEnergyFlux.py):" + self.errorOutputFileExtension(self.fext))
            sys.exit()

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_keflx.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_keflx.eps')