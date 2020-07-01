import matplotlib.pyplot as plt
import UTILS.SetAxisLimit as uSal
import UTILS.Errors as eR
import EQUATIONS.TurbulentKineticEnergyCalculation as tkeCalc
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.cm as cm

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentKineticEnergyEquation(uSal.SetAxisLimit, eR.Errors, object):

    def __init__(self, filename, ig, intc, kolmdissrate, bconv, tconv, data_prefix):
        super(TurbulentKineticEnergyEquation, self).__init__()

        # instantiate turbulent kinetic energy object
        tkeF = tkeCalc.TurbulentKineticEnergyCalculation(filename, ig, intc)

        # load all fields
        tkefields = tkeF.getTKEfield()

        self.xzn0 = tkefields['xzn0']

        # LHS -dq/dt
        self.minus_dt_dd_tke = tkefields['minus_dt_dd_tke']

        # LHS -dq/dt
        self.minus_dt_dd_tke = tkefields['minus_dt_dd_tke']

        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = tkefields['minus_div_eht_dd_fht_ux_tke']

        # -div kinetic energy flux
        self.minus_div_fekx = tkefields['minus_div_fekx']

        # -div acoustic flux		
        self.minus_div_fpx = tkefields['minus_div_fpx']

        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = tkefields['plus_ax']

        # +buoyancy work
        self.plus_wb = tkefields['plus_wb']

        # +pressure dilatation
        self.plus_wp = tkefields['plus_wp']

        # -R grad u
        self.minus_r_grad_u = tkefields['minus_r_grad_u']
        # -res		
        self.minus_resTkeEquation = tkefields['minus_resTkeEquation']

        #######################################
        # END TURBULENT KINETIC ENERGY EQUATION 
        #######################################

        # - kolm_rate u'3/lc
        self.minus_kolmrate = -kolmdissrate

        # convection boundaries
        self.tconv = tconv
        self.bconv = bconv

        # assign more global data to be shared across whole class
        self.data_prefix = data_prefix
        self.ig = ig
        self.dd = tkefields['dd']
        self.tke = tkefields['tke']

        self.nx = tkefields['nx']
        self.t_timec = tkefields['t_timec']
        self.t_tke = tkefields['t_tke']

    def plot_tke(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot turbulent kinetic energy stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot 		
        plt1 = self.tke
        # plt2 = self.eht_tke

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('turbulent kinetic energy')
        plt.plot(grd1, plt1, color='brown', label=r"$\frac{1}{2} \widetilde{u''_i u''_i}$")
        # plt.plot(grd1, plt2, color='r', linestyle='--', label=r"$\frac{1}{2} \overline{u'_i u'_i}$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\widetilde{k}$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\widetilde{k}$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_tke.png')

    def plot_tke_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot turbulent kinetic energy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_tke
        lhs1 = self.minus_div_eht_dd_fht_ux_tke

        rhs0 = self.plus_wb
        rhs1 = self.plus_wp
        rhs2 = self.minus_div_fekx
        rhs3 = self.minus_div_fpx
        rhs4 = self.minus_r_grad_u

        res = self.minus_resTkeEquation

        rhs5 = self.minus_kolmrate * self.dd

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # model constant for tke dissipation
        Cm = 0.5

        # plot DATA 
        plt.title(r'TKE equation C$_m$ = ' + str(Cm))
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t (\overline{\rho} \widetilde{k})$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{k})$")
            plt.plot(grd1, rhs0, color='r', label=r'$+W_b$')
            plt.plot(grd1, rhs1, color='c', label=r'$+W_p$')
            plt.plot(grd1, rhs2, color='#802A2A', label=r"$-\nabla_x f_k$")
            plt.plot(grd1, rhs3, color='m', label=r"$-\nabla_x f_P$")
            plt.plot(grd1, rhs4, color='b', label=r"$-\widetilde{R}_{xi}\partial_x \widetilde{u_i}$")
            plt.plot(grd1, Cm * rhs5, color='k', linewidth=0.7, label=r"$-C_m \overline{\rho} u^{'3}_{rms}/l_c$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_k$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t (\overline{\rho} \widetilde{k})$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{k})$")
            plt.plot(grd1, rhs0, color='r', label=r'$+W_b$')
            plt.plot(grd1, rhs1, color='c', label=r'$+W_p$')
            plt.plot(grd1, rhs2, color='#802A2A', label=r"$-\nabla_r f_k$")
            plt.plot(grd1, rhs3, color='m', label=r"$-\nabla_r f_P$")
            plt.plot(grd1, rhs4, color='b', label=r"$-\widetilde{R}_{ri}\partial_r \widetilde{u_i}$")
            plt.plot(grd1, Cm * rhs5, color='k', linewidth=0.7, label=r"$-C_m \overline{\rho} u^{'3}_{rms}/l_c$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_k$")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tke_eq.png')

    def plot_TKE_space_time(self, LAXIS, xbl, xbr, ybu, ybd, ilg):

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        t_timec = self.t_timec

        # load x GRID
        nx = self.nx
        grd1 = self.xzn0

        # load DATA to plot
        #plt1 = np.log10(self.t_tke.T)
        plt1 = self.t_tke.T

        #indRES = np.where((grd1 < 1.2e9) & (grd1 > 2.e8))[0]

        #pltMax = np.max(plt1[indRES])
        #pltMin = np.min(plt1[indRES])

        pltMax = np.max(plt1)
        pltMax = 8.e11 # for the thpulse
        #pltMax = 2.e12 # for neshell nucb10x
        #pltMax = 1.e14
        pltMin = np.min(plt1)

        #pltMin = 7.
        #pltMax = 14.

        # create FIGURE
        # plt.figure(figsize=(7, 6))

        fig, ax = plt.subplots(figsize=(14, 7))
        # fig.suptitle("rhoX (" + self.setNucNoUp(str(element)) + ") (g cm-3)")
        fig.suptitle("TKE")
        im = ax.imshow(plt1, interpolation='bilinear', cmap=cm.autumn,
                       origin='lower', extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]], aspect='auto',
                       vmax=pltMax, vmin=pltMin)

        #extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'time (s)'
            setylabel = r"r ($10^8$ cm)"
            ax.set_xlabel(setxlabel)
            ax.set_ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'time (s)'
            setylabel = r"r ($10^8$ cm)"
            ax.set_xlabel(setxlabel)
            ax.set_ylabel(setylabel)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_TKE_space_time.png')

    def tke_dissipation(self):
        return self.minus_resTkeEquation

    def tke(self):
        return self.tke
