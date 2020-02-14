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

class PressureFluxYequation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, tke_diss, data_prefix):
        super(PressureFluxYequation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

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
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        ppux = self.getRAdata(eht, 'ppux')[intc]
        ppuy = self.getRAdata(eht, 'ppuy')[intc]
        ppuz = self.getRAdata(eht, 'ppuz')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]
        uxuy = self.getRAdata(eht, 'uxuy')[intc]
        uxuz = self.getRAdata(eht, 'uxuz')[intc]

        ddppux = self.getRAdata(eht, 'ddppux')[intc]
        ddppuy = self.getRAdata(eht, 'ddppuy')[intc]
        ddppuz = self.getRAdata(eht, 'ddppuz')[intc]

        ppuxux = self.getRAdata(eht, 'ppuxux')[intc]
        ppuyuy = self.getRAdata(eht, 'ppuyuy')[intc]
        ppuzuz = self.getRAdata(eht, 'ppuzuz')[intc]
        ppuzuy = self.getRAdata(eht, 'ppuzuy')[intc]
        ppuzux = self.getRAdata(eht, 'ppuzux')[intc]
        ppuyux = self.getRAdata(eht, 'ppuyux')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]

        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        uydivu = self.getRAdata(eht, 'uydivu')[intc]
        uzdivu = self.getRAdata(eht, 'uzdivu')[intc]

        dddivu = self.getRAdata(eht, 'dddivu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        uxppdivu = self.getRAdata(eht, 'uxppdivu')[intc]
        uyppdivu = self.getRAdata(eht, 'uyppdivu')[intc]
        uzppdivu = self.getRAdata(eht, 'uzppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        dduxenuc1 = self.getRAdata(eht, 'dduxenuc1')[intc]
        dduyenuc1 = self.getRAdata(eht, 'dduyenuc1')[intc]
        dduzenuc1 = self.getRAdata(eht, 'dduzenuc1')[intc]

        dduxenuc2 = self.getRAdata(eht, 'dduxenuc2')[intc]
        dduyenuc2 = self.getRAdata(eht, 'dduyenuc2')[intc]
        dduzenuc2 = self.getRAdata(eht, 'dduzenuc2')[intc]

        gamma1 = self.getRAdata(eht, 'gamma1')[intc]
        gamma3 = self.getRAdata(eht, 'gamma3')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1

        uzuzcoty = self.getRAdata(eht, 'uzuzcoty')[intc]
        ppuzuzcoty = self.getRAdata(eht, 'ppuzuzcoty')[intc]

        gradxpp_o_dd = self.getRAdata(eht, 'gradxpp_o_dd')[intc]
        ppgradxpp_o_dd = self.getRAdata(eht, 'ppgradxpp_o_dd')[intc]

        gradypp_o_dd = self.getRAdata(eht, 'gradypp_o_dd')[intc]
        ppgradypp_o_dd = self.getRAdata(eht, 'ppgradypp_o_dd')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_uy = self.getRAdata(eht, 'uy')
        t_pp = self.getRAdata(eht, 'pp')
        t_ppuy = self.getRAdata(eht, 'ppuy')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_uy = dduy / dd

        fht_ppux = ddppux / dd
        fht_ppuy = ddppuy / dd

        fht_divu = dddivu / dd
        eht_uyf_uxff = uxuy - ux * uy

        eht_ppf_uyff_divuff = uyppdivu - fht_ppuy * divu - pp * uydivu - pp * fht_uy * divu - ppuy * fht_divu + \
                              fht_ppuy * fht_divu + pp * uy * fht_divu + pp * fht_uy * fht_divu

        fppx = ppux - pp * ux
        fppy = ppuy - pp * uy
        fppyx = ppuyux - ppuy * ux - pp * uxuy + pp * uy * ux - ppuy * fht_ux + pp * uy * fht_ux

        ########################
        # PRESSURE FLUX EQUATION
        ########################

        # time-series of pressure flux 
        t_fppy = t_ppuy - t_pp * t_uy

        # LHS -dq/dt 		
        self.minus_dt_fppy = -self.dt(t_fppy, xzn0, t_timec, intc)

        # LHS -fht_ux gradx fppy
        self.minus_fht_ux_gradx_fppy = -fht_ux * self.Grad(fppy, xzn0)

        # RHS -div pressure flux in y
        self.minus_div_fppyx = -self.Div(fppyx, xzn0)

        # RHS -fppx_gradx_uy
        self.minus_fppx_gradx_uy = -fppx * self.Grad(uy, xzn0)

        # RHS +eht_uyf_uxff_gradx_pp
        self.plus_eht_uyf_uxff_gradx_pp = +eht_uyf_uxff * self.Grad(pp, xzn0)

        # RHS +gamma1_eht_uyf_pp_divu
        self.plus_gamma1_eht_uyf_pp_divu = +gamma1 * (uyppdivu - uy * ppdivu)

        # RHS +gamma3_minus_one_eht_uyf_dd_enuc 		
        self.plus_gamma3_minus_one_eht_uyf_dd_enuc = +(gamma3 - 1.) * (
                (dduyenuc1 - uy * ddenuc1) + (dduyenuc2 - uy * ddenuc2))

        # RHS +eht_ppf_uyff_divuff 	
        self.plus_eht_ppf_uyff_divuff = +eht_ppf_uyff_divuff

        # RHS -eht_ppf_GtM_o_dd
        if self.ig == 1:
            self.minus_eht_ppf_GtM_o_dd = np.zeros(nx)
        elif self.ig == 2:
            self.minus_eht_ppf_GtM_o_dd = -1. * (ppuyux / xzn0 - ppuzuzcoty / xzn0 - pp * (uxuy / xzn0 - uzuzcoty / xzn0))

        # RHS -eht_ppf_grady_pp_o_ddrr
        if self.ig == 1:
            # will be multiplied by xzn0 in the plot method
            self.minus_eht_ppf_grady_pp_o_ddrr = -(ppgradypp_o_dd / xzn0 - pp * gradypp_o_dd / xzn0)
        elif self.ig == 2:
            self.minus_eht_ppf_grady_pp_o_ddrr = -(ppgradypp_o_dd / xzn0 - pp * gradypp_o_dd / xzn0)

        # -res  
        self.minus_resPPfluxEquation = -(self.minus_dt_fppy + self.minus_fht_ux_gradx_fppy + self.minus_div_fppyx +
                                         self.minus_fppx_gradx_uy + self.plus_eht_uyf_uxff_gradx_pp +
                                         self.plus_gamma1_eht_uyf_pp_divu + self.plus_gamma3_minus_one_eht_uyf_dd_enuc +
                                         self.plus_eht_ppf_uyff_divuff + self.minus_eht_ppf_GtM_o_dd +
                                         self.minus_eht_ppf_grady_pp_o_ddrr)

        ########################
        # PRESSURE FLUX EQUATION
        ########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fppy = fppy

    def plot_fppy(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean pressure flux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureFluxYEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fppy

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'pressure flux y')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'f$_{py}$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'f$_{p\theta}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$f_{py}$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$f_{p\theta}$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_fppy.png')

    def plot_fppy_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot acoustic flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureFluxYEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fppy
        lhs1 = self.minus_fht_ux_gradx_fppy

        rhs0 = self.minus_div_fppyx
        rhs1 = self.minus_fppx_gradx_uy
        rhs2 = self.plus_eht_uyf_uxff_gradx_pp
        rhs3 = self.plus_gamma1_eht_uyf_pp_divu
        rhs4 = self.plus_gamma3_minus_one_eht_uyf_dd_enuc
        rhs5 = self.plus_eht_ppf_uyff_divuff
        rhs6 = self.minus_eht_ppf_GtM_o_dd
        if self.ig == 1:
            rhs7 = self.xzn0*self.minus_eht_ppf_grady_pp_o_ddrr
        elif self.ig == 2:
            rhs7 = self.minus_eht_ppf_grady_pp_o_ddrr

        res = self.minus_resPPfluxEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('acoustic flux y equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_{py}$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_x \partial_r f_{py}$")
            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_p^x $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_{py} \partial_x \overline{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$+\overline{u'_y u''_x} \partial_x \overline{P}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$+\Gamma_1 \overline{u'_\theta P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$+(\Gamma_3-1)\overline{u'_y \rho \epsilon_{nuc}}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$+\overline{P'u''_y d''}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{P'\partial_y P/ \rho}$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_p$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_{p\theta}$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_r \partial_r f_{p\theta}$")
            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_p^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_{p\theta} \partial_r \overline{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$+\overline{u'_\theta u''_r} \partial_r \overline{P}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$+\Gamma_1 \overline{u'_\theta P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$+(\Gamma_3-1)\overline{u'_\theta \rho \epsilon_{nuc}}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$+\overline{P'u''_\theta d''}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\overline{P' G_\theta^M/ \rho}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{P'\partial_\theta P/ \rho r}$")
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
        plt.savefig('RESULTS/' + self.data_prefix + 'fppy_eq.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'fppy_eq.eps')
