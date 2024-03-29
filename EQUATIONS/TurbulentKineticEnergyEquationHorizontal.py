import matplotlib.pyplot as plt
from scipy import integrate
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

class TurbulentKineticEnergyEquationHorizontal(uSal.SetAxisLimit, eR.Errors, object):

    def __init__(self, filename, ig, intc, nsdim, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o, data_prefix):
        super(TurbulentKineticEnergyEquationHorizontal, self).__init__()

        # instantiate turbulent kinetic energy object
        tkeF = tkeCalc.TurbulentKineticEnergyCalculation(filename, ig, intc)

        # load all fields
        tkefields = tkeF.getTKEfield()

        self.xzn0 = tkefields['xzn0']
        self.yzn0 = tkefields['yzn0']
        self.zzn0 = tkefields['zzn0']

        # LHS -dq/dt
        self.minus_dt_dd_tke = tkefields['minus_dt_dd_tke']

        self.minus_dt_dd_tkex = tkefields['minus_dt_dd_tkex']
        self.minus_dt_dd_tkey = tkefields['minus_dt_dd_tkey']
        self.minus_dt_dd_tkez = tkefields['minus_dt_dd_tkez']

        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = tkefields['minus_div_eht_dd_fht_ux_tke']

        self.minus_div_eht_dd_fht_ux_tkex = tkefields['minus_div_eht_dd_fht_ux_tkex']
        self.minus_div_eht_dd_fht_ux_tkey = tkefields['minus_div_eht_dd_fht_ux_tkey']
        self.minus_div_eht_dd_fht_ux_tkez = tkefields['minus_div_eht_dd_fht_ux_tkez']

        # -div kinetic energy flux
        self.minus_div_fekx = tkefields['minus_div_fekx']

        self.minus_div_fekxx = tkefields['minus_div_fekxx']
        self.minus_div_fekxy = tkefields['minus_div_fekxy']
        self.minus_div_fekxz = tkefields['minus_div_fekxz']

        # -div acoustic flux		
        self.minus_div_fpx = tkefields['minus_div_fpx']

        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = tkefields['plus_ax']

        # +buoyancy work
        self.plus_wb = tkefields['plus_wb']

        # +pressure dilatation
        self.plus_wp = tkefields['plus_wp']

        self.plus_wpx = tkefields['plus_wpx']
        self.plus_wpy = tkefields['plus_wpy']
        self.plus_wpz = tkefields['plus_wpz']

        # -R grad u
        self.minus_r_grad_u = tkefields['minus_r_grad_u']

        self.minus_rxx_grad_ux = tkefields['minus_rxx_grad_ux']
        self.minus_rxy_grad_uy = tkefields['minus_rxy_grad_uy']
        self.minus_rxz_grad_uz = tkefields['minus_rxz_grad_uz']

        # -res
        self.minus_resTkeEquation = tkefields['minus_resTkeEquation']

        self.minus_resTkeEquationX = tkefields['minus_resTkeEquationX']
        self.minus_resTkeEquationY = tkefields['minus_resTkeEquationY']
        self.minus_resTkeEquationZ = tkefields['minus_resTkeEquationZ']

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
        self.tkex = tkefields['tkex']
        self.tkey = tkefields['tkey']
        self.tkez = tkefields['tkez']

        self.nx = tkefields['nx']
        self.t_timec = tkefields['t_timec']
        self.t_tke = tkefields['t_tke']
        self.t_tkex = tkefields['t_tkex']
        self.t_tkey = tkefields['t_tkey']
        self.t_tkez = tkefields['t_tkez']

        self.nsdim = nsdim

        self.super_ad_i = super_ad_i
        self.super_ad_o = super_ad_o

    def plot_tkeHorizontal(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot turbulent kinetic energy stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquationHorizontal.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot 		
        plt1 = self.tkey + self.tkez

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('horizontal turbulent kinetic energy')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r"$\frac{1}{2} (\widetilde{u''_y u''_y} + \widetilde{u''_y u''_y})$")

        if self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r"$\frac{1}{2} (\widetilde{u''_\theta u''_\theta} + \widetilde{u''_\phi u''_\phi})$")

        # plt.plot(grd1, plt2, color='r', linestyle='--', label=r"$\frac{1}{2} \overline{u'_i u'_i}$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\widetilde{k}^r$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\widetilde{k}^r$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_tkeHorizontal.png')

    def plot_tkeHorizontal_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot radial component of turbulent kinetic energy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquationHorizontal.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_tkey + self.minus_dt_dd_tkez
        lhs1 = self.minus_div_eht_dd_fht_ux_tkey + self.minus_div_eht_dd_fht_ux_tkez

        #rhs0 = self.plus_wb
        rhs1 = self.plus_wpy + self.plus_wpz
        rhs2 = self.minus_div_fekxy + self.minus_div_fekxz
        #rhs3 = self.minus_div_fpx
        rhs4 = self.minus_rxy_grad_uy + self.minus_rxz_grad_uz

        res = self.minus_resTkeEquationY + self.minus_resTkeEquationZ

        rhs5 = self.minus_kolmrate * self.dd

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs1, rhs2, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # model constant for tke dissipation
        Cm = 2./3.

        # plot DATA
        if self.nsdim != 2:
            # plt.title(r"TKE horizontal equation C$_m$ = " + str(Cm) + " " + str(self.nsdim) + "D")
            plt.title(r"TKE horizontal equation " + str(self.nsdim) + "D")
        else:
            plt.title(r"TKE horizontal equation " + str(self.nsdim) + "D")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t (\overline{\rho} \widetilde{k}^h)$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{k}^h)$")
            # plt.plot(grd1, rhs0, color='r', label=r'$+W_b$')
            plt.plot(grd1, rhs1, color='c', label=r'$+W_p^h$')
            plt.plot(grd1, rhs2, color='#802A2A', label=r"$-\nabla_x f_k^h$")
            # plt.plot(grd1, rhs3, color='m', label=r"$-\nabla_x f_P$")
            plt.plot(grd1, rhs4, color='b', label=r"$-\widetilde{R}_{xy}\partial_x \widetilde{u}_y -\widetilde{R}_{xz}\partial_x \widetilde{u}_z$")
            if self.nsdim != 2:
                plt.plot(grd1, Cm * rhs5, color='k', linewidth=0.7, label=r"$-2/3 \overline{\rho} u^{'3}_{rms}/l_d$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_{kh}$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t (\overline{\rho} \widetilde{k}^h)$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{k}^h)$")
            # plt.plot(grd1, rhs0, color='r', label=r'$+W_b$')
            plt.plot(grd1, rhs1, color='c', label=r'$+W_p^h$')
            plt.plot(grd1, rhs2, color='#802A2A', label=r"$-\nabla_r f_k^h$")
            # plt.plot(grd1, rhs3, color='m', label=r"$-\nabla_r f_P$")
            plt.plot(grd1, rhs4, color='b', label=r"$-\widetilde{R}_{r\theta}\partial_r \widetilde{u_\theta} -\widetilde{R}_{r\phi}\partial_r \widetilde{u_\phi}$")
            # plt.plot(grd1, Cm * rhs5, color='k', linewidth=0.7, label=r"$-C_m \overline{\rho} u^{'3}_{rms}/l_c$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_k^r$")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        # plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        # plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

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
        plt.savefig('RESULTS/' + self.data_prefix + 'tke_horizontal_eq.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'tke_horizontal_eq.eps')

    def plot_TKEhorizontal_space_time(self, LAXIS, xbl, xbr, ybu, ybd, ilg):

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        t_timec = self.t_timec

        # load x GRID
        nx = self.nx
        grd1 = self.xzn0

        # load DATA to plot
        # plt1 = np.log10(self.t_tke.T)
        plt1 = self.t_tkey.T + self.t_tkez.T

        # indRES = np.where((grd1 < 1.2e9) & (grd1 > 2.e8))[0]

        # pltMax = np.max(plt1[indRES])
        # pltMin = np.min(plt1[indRES])

        pltMax = np.max(plt1)
        # pltMax = 8.e11 # for the thpulse
        # pltMax = 1.e14
        # pltMax = 4.e12
        # pltMax = 2.e12 # for neshell nucb10x
        pltMax = 6.e14
        pltMin = np.min(plt1)

        plt1 = np.log10(plt1)

        pltMin = 11.5
        pltMax = 13.


        # create FIGURE
        # plt.figure(figsize=(7, 6))

        fig, ax = plt.subplots(figsize=(14, 7))
        # fig.suptitle("rhoX (" + self.setNucNoUp(str(element)) + ") (g cm-3)")
        # fig.suptitle("TKE 2D")
        im = ax.imshow(plt1, interpolation='bilinear', cmap=cm.autumn,
                       origin='lower', extent=[t_timec[0], t_timec[-1], grd1[0], grd1[-1]], aspect='auto',
                       vmax=pltMax, vmin=pltMin)

        # extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'time (s)'
            setylabel = r"x ($10^8$ cm)"
            ax.set_xlabel(setxlabel)
            ax.set_ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'time (s)'
            # setylabel = r"r ($10^8$ cm)"
            setylabel = r"r (cm)"
            ax.set_xlabel(setxlabel)
            ax.set_ylabel(setylabel)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_TKEhorizontal_space_time.png')

    def plot_tkeHorizontal_equation_integral_budget(self, laxis, xbl, xbr, ybu, ybd):
        """Plot integral budgets of tke equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquationHorizontal.py):" + self.errorGeometry(self.ig))
            sys.exit()

        term1 = self.minus_dt_dd_tkey + self.minus_dt_dd_tkez
        term2 = self.minus_div_eht_dd_fht_ux_tkey + self.minus_div_eht_dd_fht_ux_tkez
        # term3 = self.plus_wb
        term4 = self.plus_wpy + self.plus_wpz
        term5 = self.minus_div_fekxy + self.minus_div_fekxz
        # term6 = self.minus_div_fpx
        term7 = self.minus_rxy_grad_uy + self.minus_rxz_grad_uz
        term8 = self.minus_resTkeEquationY + self.minus_resTkeEquationZ

        # hack for the ccp setup getting rid of bndry noise
        # fct1 = 0.5e-1
        # fct2 = 1.e-1
        # xbl = xbl + fct1*xbl
        # xbr = xbr - fct2*xbl
        # print(xbl,xbr)

        # calculate INDICES for grid boundaries
        if laxis == 1 or laxis == 2:
            idxl, idxr = self.idx_bndry(xbl, xbr)
        else:
            idxl = 0
            idxr = self.nx - 1

        term1_sel = term1[idxl:idxr]
        term2_sel = term2[idxl:idxr]
        # term3_sel = term3[idxl:idxr]
        term4_sel = term4[idxl:idxr]
        term5_sel = term5[idxl:idxr]
        # term6_sel = term6[idxl:idxr]
        term7_sel = term7[idxl:idxr]
        term8_sel = term8[idxl:idxr]

        rc = self.xzn0[idxl:idxr]

        # handle geometry
        Sr = 0.
        if self.ig == 1 and self.nsdim == 3:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.zzn0[-1] - self.zzn0[0])
        elif self.ig == 1 and self.nsdim == 2:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.yzn0[-1] - self.yzn0[0])
        elif self.ig == 2:
            Sr = 4. * np.pi * rc ** 2

        int_term1 = integrate.simps(term1_sel * Sr, rc)
        int_term2 = integrate.simps(term2_sel * Sr, rc)
        # int_term3 = integrate.simps(term3_sel * Sr, rc)
        int_term4 = integrate.simps(term4_sel * Sr, rc)
        int_term5 = integrate.simps(term5_sel * Sr, rc)
        # int_term6 = integrate.simps(term6_sel * Sr, rc)
        int_term7 = integrate.simps(term7_sel * Sr, rc)
        int_term8 = integrate.simps(term8_sel * Sr, rc)

        fig = plt.figure(figsize=(7, 6))

        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd, ybu])

        fc = 1.

        # note the change: I'm only supplying y data.
        y = [int_term1 / fc, int_term2 / fc, int_term4 / fc,
             int_term5 / fc, int_term7 / fc, int_term8 / fc]

        # calculate how many bars there will be
        N = len(y)

        # Generate a list of numbers, from 0 to N
        # This will serve as the (arbitrary) x-axis, which
        # we will then re-label manually.
        ind = range(N)

        # See note below on the breakdown of this command
        ax.bar(ind, y, facecolor='#0000FF',
               align='center', ecolor='black')

        # Create a y label
        ax.set_ylabel(r'ergs s$^{-1}$')

        if self.nsdim != 2:
            ax.set_title(r"TKE hor equation integral budget " + str(self.nsdim) + "D")
        else:
            ax.set_title(r"TKE hor equation integral budget " + str(self.nsdim) + "D")

        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        if self.ig == 1:
            group_labels = [r'$-\partial_t (\overline{\rho} \widetilde{k}^h)$',
                            r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{k}^h)$",
                            r'$+W_p^h$', r"$-\nabla_x f_k^h$",
                            r"$-\widetilde{R}_{xy}\partial_x \widetilde{u}_y -\widetilde{R}_{xz}\partial_x \widetilde{u}_z$", 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=16)
        elif self.ig == 2:
            group_labels = [r'$-\partial_t (\overline{\rho} \widetilde{k}^h)$',
                            r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{k}^h)$",
                            r'$+W_p^h$', r"$-\nabla_r f_k^h$",
                            r"$-\widetilde{R}_{r\theta}\partial_r \widetilde{u_\theta} -\widetilde{R}_{r\phi}\partial_r \widetilde{u_\phi}$", 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=16)

        # auto-rotate the x axis labels
        fig.autofmt_xdate()

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tke_eqHorizontal_bar.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'tke_eqHorizontal_bar.eps')

    def tke_dissipation(self):
        return self.minus_resTkeEquation

    def tke(self):
        return self.tke
