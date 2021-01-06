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

class TurbulentMassFlux(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, intc, data_prefix, lc):
        super(TurbulentMassFlux, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]
        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        gg = self.getRAdata(eht, 'gg')[intc]
        sv = self.getRAdata(eht, 'sv')[intc]
        mm = self.getRAdata(eht, 'mm')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        ttux = self.getRAdata(eht, 'ttux')[intc]
        ppux = self.getRAdata(eht, 'ppux')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]

        x0001 = self.getRAdata(eht, 'x0001')[intc]
        x0002 = self.getRAdata(eht, 'x0002')[intc]
        x0001ux = self.getRAdata(eht, 'x0001ux')[intc]
        x0002ux = self.getRAdata(eht, 'x0002ux')[intc]
        # self.alphac0001 = self.getRAdata(eht, 'alphac0001')[intc]
        # self.alphac0002 = self.getRAdata(eht, 'alphac0002')[intc]

        # a is turbulent mass flux
        eht_a = ux - ddux / dd

        # temperature flux
        eht_ftt = ttux - tt*ux
        # acoustic flux
        eht_fpp = ppux - pp*ux

        # composition fluxes
        self.eht_fx0001 = x0001ux - x0001*ux
        self.eht_fx0002 = x0002ux - x0002*ux

        # mass flux and temperature flux relation for incompressible/Businessq case
        self.eht_a_tempflx = - (dd/tt)*eht_ftt

        self.eht_a_pressflx = + (dd/pp)*eht_fpp

        # self.eht_a_compflx = -self.alphac0001*self.eht_fx0001 + self.alphac0002*self.eht_fx0002
        #print(self.alphac0001)
        #print("************")
        #print(self.eht_fx0001)
        #print("************")
        #print(self.alphac0002)
        #print("************")
        #print(self.eht_fx0002)

        # print(self.eht_a_grad_model)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.eht_a = eht_a
        self.ux = ux
        self.dd = dd

    def plot_a(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean turbulent mass flux in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentMassFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = -self.dd*self.eht_a
        plt2 = self.eht_a_tempflx
        plt3 = self.eht_a_pressflx
        #plt4 = self.eht_a_compflx
        #plt4 = self.alphac0001*self.eht_fx0001
        #plt4 = self.alphac0002*self.eht_fx0002

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        # plt.title(r'turbulent mass flux'+ ' c = ' + str(self.coeff))
        plt.title(r'turbulent mass flux')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r"$+\overline{\rho' u'_x}$")
            plt.plot(grd1, plt2, color='g', linestyle='--', label=r"$-(\overline{\rho} / \overline{T}) \ \overline{T'u'_x}$")
            plt.plot(grd1, plt3, color='r', linestyle='--', label=r"$+(\overline{\rho} / \overline{P}) \ \overline{P'u'_x}$")
            # plt.plot(grd1, plt4, color='b', linestyle='--', label=r"$+\alpha_c^i \ \overline{X_i'u'_x}$")
            plt.plot(grd1, plt2+plt3, color='r', linestyle='dotted', label=r"$sum$")
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r"$a$")

        # horizontal line at y = 0
        plt.axhline(0.0, linestyle='dotted', linewidth=0.7, color='k')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"+$\overline{\rho' u'_x}$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\overline{\rho}$ $\overline{u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        plt.savefig('RESULTS/' + self.data_prefix + 'mean_a.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_a.eps')

        # create FIGURE
        # plt.figure(figsize=(7, 6))

        # plt1 = self.alphac0001
        # plt2 = self.alphac0002

        # format AXIS, make sure it is exponential
        # plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        # to_plot = [plt1]
        # self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plt.plot(grd1, plt1)
        # plt.plot(grd1, plt2)

        # print(self.alphac0001)
        # plt.show(block=False)

