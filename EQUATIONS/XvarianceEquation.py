import numpy as np
import sys
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XvarianceEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, inuc, element, tauL, bconv, tconv, intc, nsdim, data_prefix):
        super(XvarianceEquation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        xi = self.getRAdata(eht, 'x' + inuc)[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]
        ddxisq = self.getRAdata(eht, 'ddx' + inuc + 'sq')[intc]
        ddxisqux = self.getRAdata(eht, 'ddx' + inuc + 'squx')[intc]

        ddxiuxux = self.getRAdata(eht, 'ddx' + inuc + 'uxux')[intc]
        ddxiuyuy = self.getRAdata(eht, 'ddx' + inuc + 'uyuy')[intc]
        ddxiuzuz = self.getRAdata(eht, 'ddx' + inuc + 'uzuz')[intc]

        ddxixidot = self.getRAdata(eht, 'ddx' + inuc + 'x' + inuc + 'dot')[intc]
        ddxidotux = self.getRAdata(eht, 'ddx' + inuc + 'dotux')[intc]

        xigradxpp = self.getRAdata(eht, 'x' + inuc + 'gradxpp')[intc]

        ######################
        # Xi VARIANCE EQUATION 
        ######################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddux = self.getRAdata(eht, 'ddux')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_ddxisq = self.getRAdata(eht, 'ddx' + inuc + 'sq')
        t_ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')

        # construct equation-specific mean fields
        t_eht_dd_sigmai = t_ddxisq - t_ddxi * t_ddxi / t_dd

        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        sigmai = (ddxisq - ddxi * ddxi / dd) / dd
        fsigmai = ddxisqux - 2. * ddxiux * ddxi / dd - ddxisq * ddux / dd + 2. * ddxi * ddxi * ddux / (dd * dd)
        fxi = ddxiux - ddxi * ddux / dd

        # LHS -dq/dt 
        self.minus_dt_eht_dd_sigmai = -self.dt(t_eht_dd_sigmai, xzn0, t_timec, intc)

        # LHS -div(dduxsigmai)
        self.minus_div_eht_dd_fht_ux_sigmai = -self.Div(dd * fht_ux * sigmai, xzn0)

        # RHS -div fsigmai
        self.minus_div_fsigmai = -self.Div(fsigmai, xzn0)

        # RHS -2 fxi gradx fht_xi
        self.minus_two_fxi_gradx_fht_xi = -2. * fxi * self.Grad(fht_xi, xzn0)

        # RHS +2 xiff eht_dd xidot
        self.plus_two_xiff_eht_dd_xidot = +2. * (ddxixidot - (ddxi / dd) * ddxidot)

        # -res
        self.minus_resXiVariance = -(self.minus_dt_eht_dd_sigmai + self.minus_div_eht_dd_fht_ux_sigmai +
                                     self.minus_div_fsigmai + self.minus_two_fxi_gradx_fht_xi +
                                     self.plus_two_xiff_eht_dd_xidot)

        ##########################
        # END Xi VARIANCE EQUATION 		
        ##########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.inuc = inuc
        self.element = element
        self.tauL = tauL
        self.dd = dd
        self.sigmai = sigmai

        self.bconv = bconv
        self.tconv = tconv
        self.ig = ig
        self.nsdim = nsdim

    def plot_Xvariance(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xvariance stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XvarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load and calculate DATA to plot
        plt1 = self.sigmai

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format Y AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xvariance for ' + self.element + str(self.nsdim) + "D")
        plt.semilogy(grd1, plt1, color='b', label=r"$\sigma_i$")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\widetilde{X''_i X''_i}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\widetilde{X''_i X''_i}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xvariance_' + element + '.png')

    def plot_Xvariance_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xi variance equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XvarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        tauL = self.tauL

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_sigmai
        lhs1 = self.minus_div_eht_dd_fht_ux_sigmai

        rhs0 = self.minus_div_fsigmai
        rhs1 = self.minus_two_fxi_gradx_fht_xi
        rhs2 = self.plus_two_xiff_eht_dd_xidot

        res = self.minus_resXiVariance

        self.minus_variancediss = -self.dd * self.sigmai / tauL
        rhs3 = self.minus_variancediss

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # model constant for variance dissipation
        Cm = 0.1

        # plot DATA 
        # plt.title(r'Xvariance equation for ' + self.element + ' C$_m$ = ' + str(Cm))
        plt.title(r'Xvariance equation for ' + self.element + " " + str(self.nsdim) + "D")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='cyan', label=r'$-\partial_t (\overline{\rho} \sigma)$')
            plt.plot(grd1, lhs1, color='purple', label=r'$-\nabla_x (\overline{\rho} \widetilde{u}_x \sigma)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f^\sigma$')
            plt.plot(grd1, rhs1, color='g', label=r'$-2 f_i \partial_x \widetilde{X}$')
            plt.plot(grd1, rhs2, color='r', label=r'$+2 \overline{\rho X'' \dot{X}}$')
            # plt.plot(grd1, Cm * rhs3, color='k', linewidth=0.8, label=r'$- C_m \ \overline{\rho} \sigma / \tau_L$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='cyan', label=r'$-\partial_t (\overline{\rho} \sigma)$')
            plt.plot(grd1, lhs1, color='purple', label=r'$-\nabla_r (\overline{\rho} \widetilde{u}_r \sigma)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_r f^\sigma$')
            plt.plot(grd1, rhs1, color='g', label=r'$-2 f_i \partial_r \widetilde{X}$')
            plt.plot(grd1, rhs2, color='r', label=r'$+2 \overline{\rho X'' \dot{X}}$')
            plt.plot(grd1, Cm * rhs3, color='k', linewidth=0.8, label=r'$- C \overline{\rho} \sigma / \tau_L$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XvarianceEquation_' + element + '.png')
