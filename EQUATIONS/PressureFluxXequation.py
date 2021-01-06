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

class PressureFluxXequation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, ieos, intc, tke_diss, data_prefix):
        super(PressureFluxXequation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        ppux = self.getRAdata(eht, 'ppux')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]

        uxuy = self.getRAdata(eht, 'uxuy')[intc]
        uxuz = self.getRAdata(eht, 'uxuz')[intc]

        dduxuxux = self.getRAdata(eht, 'dduxuxux')[intc]
        dduxuyuy = self.getRAdata(eht, 'dduxuyuy')[intc]
        dduxuzuz = self.getRAdata(eht, 'dduxuzuz')[intc]

        ddppux = self.getRAdata(eht, 'ddppux')[intc]
        ppuxux = self.getRAdata(eht, 'ppuxux')[intc]
        ppuyuy = self.getRAdata(eht, 'ppuyuy')[intc]
        ppuzuz = self.getRAdata(eht, 'ppuzuz')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        dddivu = self.getRAdata(eht, 'dddivu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        uxppdivu = self.getRAdata(eht, 'uxppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        dduxenuc1 = self.getRAdata(eht, 'dduxenuc1')[intc]
        dduxenuc2 = self.getRAdata(eht, 'dduxenuc2')[intc]

        gamma1 = self.getRAdata(eht, 'gamma1')[intc]
        gamma3 = self.getRAdata(eht, 'gamma3')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1

        gradxpp_o_dd = self.getRAdata(eht, 'gradxpp_o_dd')[intc]
        ppgradxpp_o_dd = self.getRAdata(eht, 'ppgradxpp_o_dd')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_ux = self.getRAdata(eht, 'ux')
        t_pp = self.getRAdata(eht, 'pp')
        t_ppux = self.getRAdata(eht, 'ppux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ppux = ddppux / dd
        fht_divu = dddivu / dd
        eht_uxf_uxff = uxux - ux * ux
        eht_ppf_uxff_divuff = uxppdivu - fht_ppux * divu - pp * uxdivu - pp * fht_ux * divu - ppux * fht_divu + \
                              fht_ppux * fht_divu + pp * ux * fht_divu + pp * fht_ux * fht_divu

        fppx = ppux - pp * ux
        fppxx = ppuxux - ppux * ux - pp * uxux + pp * ux * ux - ppux * fht_ux + pp * ux * fht_ux

        ########################
        # PRESSURE FLUX EQUATION
        ########################

        # time-series of pressure flux 
        t_fppx = t_ppux - t_pp * t_ux

        # LHS -dq/dt 		
        self.minus_dt_fppx = -self.dt(t_fppx, xzn0, t_timec, intc)

        # LHS -fht_ux gradx fppx
        self.minus_fht_ux_gradx_fppx = -fht_ux * self.Grad(fppx, xzn0)

        # RHS -div pressure flux in x
        self.minus_div_fppxx = -self.Div(fppxx, xzn0)

        # RHS -fppx_gradx_ux
        self.minus_fppx_gradx_ux = -fppx * self.Grad(ux, xzn0)

        # RHS +eht_uxf_uxff_gradx_pp
        self.plus_eht_uxf_uxff_gradx_pp = +eht_uxf_uxff * self.Grad(pp, xzn0)

        # RHS +gamma1_eht_uxf_pp_divu
        self.plus_gamma1_eht_uxf_pp_divu = +gamma1 * (uxppdivu - ux * ppdivu)

        # RHS +gamma3_minus_one_eht_uxf_dd_enuc 		
        self.plus_gamma3_minus_one_eht_uxf_dd_enuc = +(gamma3 - 1.) * (
                (dduxenuc1 - ux * ddenuc1) + (dduxenuc2 - ux * ddenuc2))

        # RHS +eht_ppf_uxff_divuff 	
        self.plus_eht_ppf_uxff_divuff = +eht_ppf_uxff_divuff

        # RHS -eht_ppf_GrM_o_dd
        if self.ig == 1:
            self.minus_eht_ppf_GrM_o_dd = np.zeros(nx)
        elif self.ig == 2:
            self.minus_eht_ppf_GrM_o_dd = -1. * (-ppuyuy / xzn0 - ppuzuz / xzn0 + pp * (uyuy / xzn0 + uzuz / xzn0))

        # RHS -eht_ppf_gradx_pp_o_dd 		
        self.minus_eht_ppf_gradx_pp_o_dd = -(ppgradxpp_o_dd	- pp*gradxpp_o_dd)

        # this term is approx. zero, just replace the gradx pp with rho gg		
        # self.minus_eht_ppf_gradx_pp_o_dd = np.zeros(nx)

        # -res  
        self.minus_resPPfluxEquation = -(self.minus_dt_fppx + self.minus_fht_ux_gradx_fppx + self.minus_div_fppxx +
                                         self.minus_fppx_gradx_ux + self.plus_eht_uxf_uxff_gradx_pp +
                                         self.plus_gamma1_eht_uxf_pp_divu +
                                         self.plus_gamma3_minus_one_eht_uxf_dd_enuc + self.plus_eht_ppf_uxff_divuff +
                                         self.minus_eht_ppf_GrM_o_dd +
                                         self.minus_eht_ppf_gradx_pp_o_dd)

        ########################
        # PRESSURE FLUX EQUATION
        ########################

        # acoustic flux model (uxuxux approx. dduxuxux/dd)
        eht_uxuxux = dduxuxux/dd - 3.*ux*uxux + 2.*ux*ux*ux
        eht_uxuyuy = dduxuyuy/dd - 2.*uy*uxuy - ux*uyuy + 2.*ux*uy*uy
        eht_uxuzuz = dduxuzuz/dd - 2.*uy*uxuz - ux*uzuz + 2.*ux*uz*uz

        a = 1./2.
        self.minus_a_eht_uxuiui = -a*dd*(eht_uxuxux+eht_uxuyuy+eht_uxuzuz)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fppx = fppx

    def plot_fppx(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean pressure flux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureFluxXEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fppx
        plt2 = self.minus_a_eht_uxuiui

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'pressure flux x')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r"f$_{px}$")
            plt.plot(grd1, plt2, color='red', label=r"-a$\overline{u'_x u'_i u'_i}$")
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'f$_{pr}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$f_{px}$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$f_{pr}$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_fppx.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_fppx.eps')

    def plot_fppx_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot acoustic flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureFluxXEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fppx
        lhs1 = self.minus_fht_ux_gradx_fppx

        rhs0 = self.minus_div_fppxx
        rhs1 = self.minus_fppx_gradx_ux
        rhs2 = self.plus_eht_uxf_uxff_gradx_pp
        rhs3 = self.plus_gamma1_eht_uxf_pp_divu
        rhs4 = self.plus_gamma3_minus_one_eht_uxf_dd_enuc
        rhs5 = self.plus_eht_ppf_uxff_divuff
        rhs6 = self.minus_eht_ppf_GrM_o_dd
        rhs7 = self.minus_eht_ppf_gradx_pp_o_dd

        res = self.minus_resPPfluxEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('acoustic flux x equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_{px}$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_x \partial_x f_{px}$")
            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_p^x $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_{px} \partial_x \overline{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$+\overline{u'_x u''_x} \partial_x \overline{P}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$+\Gamma_1 \overline{u'_x P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$+(\Gamma_3-1)\overline{u'_x \rho \epsilon_{nuc}}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$+\overline{P'u''_x d''}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{P'\partial_x P/ \rho}$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_p$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_{pr}$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_r \partial_r f_{pr}$")
            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_p^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_{pr} \partial_r \overline{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$+\overline{u'_r u''_r} \partial_r \overline{P}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$+\Gamma_1 \overline{u'_r P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$+(\Gamma_3-1)\overline{u'_r \rho \epsilon_{nuc}}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$+\overline{P'u''_rd''}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\overline{P' G_r^M/ \rho}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{P'\partial_r P/ \rho} \sim 0$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_p$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg cm$^{-2}$ s$^{-2}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg cm$^{-2}$ s$^{-2}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'fppx_eq.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'fppx_eq.eps')
