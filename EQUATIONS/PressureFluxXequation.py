import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class PressureFluxXequation(calc.CALCULUS, al.ALIMIT, object):

    def __init__(self, filename, ig, ieos, intc, tke_diss, data_prefix):
        super(PressureFluxXequation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0'))
        nx = np.asarray(eht.item().get('nx'))

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])
        pp = np.asarray(eht.item().get('pp')[intc])

        ddux = np.asarray(eht.item().get('ddux')[intc])
        ppux = np.asarray(eht.item().get('ppux')[intc])

        uxux = np.asarray(eht.item().get('uxux')[intc])
        uyuy = np.asarray(eht.item().get('uyuy')[intc])
        uzuz = np.asarray(eht.item().get('uzuz')[intc])

        ddppux = np.asarray(eht.item().get('ddppux')[intc])
        ppuxux = np.asarray(eht.item().get('ppuxux')[intc])
        ppuyuy = np.asarray(eht.item().get('ppuyuy')[intc])
        ppuzuz = np.asarray(eht.item().get('ppuzuz')[intc])

        divu = np.asarray(eht.item().get('divu')[intc])
        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])
        dddivu = np.asarray(eht.item().get('dddivu')[intc])
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])
        uxppdivu = np.asarray(eht.item().get('uxppdivu')[intc])

        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])

        dduxenuc1 = np.asarray(eht.item().get('dduxenuc1')[intc])
        dduxenuc2 = np.asarray(eht.item().get('dduxenuc2')[intc])

        gamma1 = np.asarray(eht.item().get('gamma1')[intc])
        gamma3 = np.asarray(eht.item().get('gamma3')[intc])

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if (ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1

        gradxpp_o_dd = np.asarray(eht.item().get('gradxpp_o_dd')[intc])
        ppgradxpp_o_dd = np.asarray(eht.item().get('ppgradxpp_o_dd')[intc])

        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))
        t_ux = np.asarray(eht.item().get('ux'))
        t_pp = np.asarray(eht.item().get('pp'))
        t_ppux = np.asarray(eht.item().get('ppux'))

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
        self.minus_eht_ppf_GrM_o_dd = -1. * (-ppuyuy / xzn0 - ppuzuz / xzn0 + pp * (uyuy / xzn0 + uzuz / xzn0))

        # RHS -eht_ppf_gradx_pp_o_dd 		
        # self.minus_eht_ppf_gradx_pp_o_dd = -(ppgradxpp_o_dd	- pp*gradxpp_o_dd)

        # this term is approx. zero, just replace the gradx pp with rho gg		
        self.minus_eht_ppf_gradx_pp_o_dd = np.zeros(nx)

        # -res  
        self.minus_resPPfluxEquation = -(self.minus_dt_fppx + self.minus_fht_ux_gradx_fppx + self.minus_div_fppxx + \
                                         self.minus_fppx_gradx_ux + self.plus_eht_uxf_uxff_gradx_pp + self.plus_gamma1_eht_uxf_pp_divu + \
                                         self.plus_gamma3_minus_one_eht_uxf_dd_enuc + self.plus_eht_ppf_uxff_divuff + self.minus_eht_ppf_GrM_o_dd + \
                                         self.minus_eht_ppf_gradx_pp_o_dd)

        ########################		
        # PRESSURE FLUX EQUATION
        ########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fppx = fppx

    def plot_fppx(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mean pressure flux stratification in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fppx

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'pressure flux x')
        plt.plot(grd1, plt1, color='brown', label=r'f$_{pr}$')

        # define and show x/y LABELS
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

    def plot_fppx_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot acoustic flux equation in the model"""

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
        plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_{pr}$")
        plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_r \partial_r f_{pr}$)")

        plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_p^r $")
        plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_{pr} \partial_r \overline{u}_r$")
        plt.plot(grd1, rhs2, color='r', label=r"$+\overline{u'_r u''_r} \partial_r \overline{P}$")
        plt.plot(grd1, rhs3, color='firebrick', label=r"$+\Gamma_1 \overline{u'_r P d}$")
        plt.plot(grd1, rhs4, color='c', label=r"$+(\Gamma_3-1)\overline{u'_r \rho \epsilon_{nuc}}$")
        plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$+\overline{P'u''_rd''}$")
        plt.plot(grd1, rhs6, color='b', label=r"$+\overline{P' G_r^M/ \rho}$")
        plt.plot(grd1, rhs7, color='m', label=r"$+\overline{P'\partial_r P/ \rho} \sim 0$")

        plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_p$")

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-2}$ s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'fppx_eq.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'fppx_eq.eps')
