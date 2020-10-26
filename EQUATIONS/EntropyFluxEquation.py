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

class EntropyFluxEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, tke_diss, data_prefix):
        super(EntropyFluxEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        ss = self.getRAdata(eht, 'ss')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]
        ddss = self.getRAdata(eht, 'ddss')[intc]
        ddgg = self.getRAdata(eht, 'ddgg')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        ddssux = self.getRAdata(eht, 'ddssux')[intc]
        ddssuy = self.getRAdata(eht, 'ddssuy')[intc]
        ddssuz = self.getRAdata(eht, 'ddssuz')[intc]

        ssddgg = self.getRAdata(eht, 'ssddgg')[intc]

        ddssuxux = self.getRAdata(eht, 'ddssuxux')[intc]
        ddssuyuy = self.getRAdata(eht, 'ddssuyuy')[intc]
        ddssuzuz = self.getRAdata(eht, 'ddssuzuz')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ddenuc1_o_tt = self.getRAdata(eht, 'ddenuc1_o_tt')[intc]
        ddenuc2_o_tt = self.getRAdata(eht, 'ddenuc2_o_tt')[intc]

        dduxenuc1_o_tt = self.getRAdata(eht, 'dduxenuc1_o_tt')[intc]
        dduxenuc2_o_tt = self.getRAdata(eht, 'dduxenuc2_o_tt')[intc]

        ssgradxpp = self.getRAdata(eht, 'ssgradxpp')[intc]

        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        uxppdivu = self.getRAdata(eht, 'uxppdivu')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddux = self.getRAdata(eht, 'ddux')
        t_ddss = self.getRAdata(eht, 'ddss')
        t_ddssux = self.getRAdata(eht, 'ddssux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ss = ddss / dd
        rxx = dduxux - ddux * ddux / dd

        f_ss = ddssux - ddux * ddss / dd
        fr_ss = ddssuxux - ddss * dduxux / dd - 2. * fht_ux * ddssux + 2. * dd * fht_ux * fht_ss * fht_ux

        ssff = ss - ddss / dd
        ssff_gradx_ppf = ssgradxpp - ss * self.Grad(pp, xzn0)

        uxff_dd_enuc_T = (dduxenuc1_o_tt + dduxenuc2_o_tt) - fht_ux * (ddenuc1_o_tt + ddenuc2_o_tt)

        uxff_epsilonk_approx = (ux - ddux / dd) * tke_diss

        Grss = -(ddssuyuy - ddss * dduyuy / dd - 2. * (dduy / dd) * (ddssuy / dd) + 2. * ddss * dduy * dduy / (
                dd * dd * dd)) / xzn0 - \
               (ddssuzuz - ddss * dduzuz / dd - 2. * (dduz / dd) * (ddssuz / dd) + 2. * ddss * dduz * dduz / (
                       dd * dd * dd)) / xzn0

        ssff_GrM = -(ddssuyuy - (ddss / dd) * dduyuy) / xzn0 - (ddssuzuz - (ddss / dd) * dduzuz) / xzn0

        #######################
        # ENTROPY FLUX EQUATION
        #######################

        # time-series of entropy flux 
        t_f_ss = t_ddssux / t_dd - t_ddss * t_ddux / (t_dd * t_dd)

        # LHS -dq/dt 		
        self.minus_dt_f_ss = -self.dt(t_f_ss, xzn0, t_timec, intc)

        # LHS -div fht_ux f_ss
        self.minus_div_fht_ux_f_ss = -self.Div(fht_ux * f_ss, xzn0)

        # RHS -div flux internal energy flux
        self.minus_div_fr_ss = -self.Div(fr_ss, xzn0)

        # RHS -f_ss_gradx_fht_ux
        self.minus_f_ss_gradx_fht_ux = -f_ss * self.Grad(fht_ux, xzn0)

        # RHS -rxx_gradx_fht_ss
        self.minus_rxx_gradx_fht_ss = -rxx * self.Grad(fht_ss, xzn0)

        # RHS -eht_ssff_gradx_eht_pp
        self.minus_eht_ssff_gradx_eht_pp = -(ss - ddss / dd) * self.Grad(pp, xzn0)

        # RHS -eht_ssff_gradx_ppf
        self.minus_eht_ssff_gradx_ppf = -(ssgradxpp - (ddss / dd) * self.Grad(pp, xzn0))

        # RHS eht_uxff_dd_nuc_T	
        self.plus_eht_uxff_dd_nuc_T = (dduxenuc1_o_tt + dduxenuc2_o_tt) - fht_ux * (ddenuc1_o_tt + ddenuc2_o_tt)

        # RHS eht_uxff_div_ftt_T (not calculated)
        eht_uxff_div_f_o_tt_T = np.zeros(nx)
        self.plus_eht_uxff_div_ftt_T = eht_uxff_div_f_o_tt_T

        # RHS eht_uxff_epsilonk_approx_T	
        self.plus_eht_uxff_epsilonk_approx_T = (ux - fht_ux) * tke_diss / tt

        # RHS Gss
        if self.ig == 1:
            self.plus_Gss = np.zeros(nx)
        elif self.ig == 2:
            self.plus_Gss = -Grss - ssff_GrM

        # -res  
        self.minus_resSSfluxEquation = -(self.minus_dt_f_ss + self.minus_div_fht_ux_f_ss +
                                         self.minus_div_fr_ss + self.minus_f_ss_gradx_fht_ux +
                                         self.minus_rxx_gradx_fht_ss +
                                         self.minus_eht_ssff_gradx_eht_pp + self.minus_eht_ssff_gradx_ppf +
                                         self.plus_eht_uxff_dd_nuc_T + self.plus_eht_uxff_div_ftt_T +
                                         self.plus_eht_uxff_epsilonk_approx_T +
                                         self.plus_Gss)

        # RHS -eht_ssddgg
        self.minus_eht_ssddgg = -ssddgg

        # RHS +fht_ss_eht_ddgg		
        self.plus_fht_ss_eht_ddgg = +fht_ss * ddgg

        # -res2 
        self.minus_resSSfluxEquation2 = -(self.minus_dt_f_ss + self.minus_div_fht_ux_f_ss +
                                          self.minus_div_fr_ss + self.minus_f_ss_gradx_fht_ux +
                                          self.minus_rxx_gradx_fht_ss +
                                          self.minus_eht_ssddgg + self.plus_fht_ss_eht_ddgg + self.minus_eht_ssff_gradx_ppf +
                                          self.plus_eht_uxff_dd_nuc_T + self.plus_eht_uxff_div_ftt_T +
                                          self.plus_eht_uxff_epsilonk_approx_T +
                                          self.plus_Gss)

        ###########################
        # END ENTROPY FLUX EQUATION
        ###########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.f_ss = f_ss

    def plot_fss(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean Favrian entropy flux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EntropyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.f_ss

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'entropy flux')
        plt.plot(grd1, plt1, color='brown', label=r'f$_s$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$f_s$ (erg K$^{-1}$ cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$f_s$ (erg K$^{-1}$ cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_fss.png')

    def plot_fss_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot entropy flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EntropyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_f_ss
        lhs1 = self.minus_div_fht_ux_f_ss

        rhs0 = self.minus_div_fr_ss
        rhs1 = self.minus_f_ss_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_ss
        rhs3 = self.minus_eht_ssff_gradx_eht_pp
        rhs4 = self.minus_eht_ssff_gradx_ppf
        rhs5 = self.plus_eht_uxff_dd_nuc_T
        rhs6 = self.plus_eht_uxff_div_ftt_T
        rhs7 = self.plus_eht_uxff_epsilonk_approx_T
        rhs8 = self.plus_Gss

        res = self.minus_resSSfluxEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('entropy flux equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_s$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\widetilde{u}_x f_s$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_s^x $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_s \partial_x \widetilde{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{xx} \partial_x \widetilde{s}$")
            plt.plot(grd1, rhs3, color='c', label=r"$-\overline{s''} \ \partial_x \overline{P}$")
            plt.plot(grd1, rhs4, color='mediumseagreen', label=r"$- \overline{s''\partial_x P'}$")
            plt.plot(grd1, rhs5, color='b', label=r"$+\overline{u''_x \rho \varepsilon_{nuc} /T}$")
            plt.plot(grd1, rhs6, color='m', label=r"$+\overline{u''_x \nabla \cdot T /T}$")
            plt.plot(grd1, rhs7, color='g', label=r"$+\overline{u''_x \varepsilon_k /T}$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_fs$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_s$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\widetilde{u}_r f_s$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_s^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_s \partial_r \widetilde{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{rr} \partial_r \widetilde{s}$")
            plt.plot(grd1, rhs3, color='c', label=r"$-\overline{s''} \ \partial_r \overline{P}$")
            plt.plot(grd1, rhs4, color='mediumseagreen', label=r"$- \overline{s''\partial_r P'}$")
            plt.plot(grd1, rhs5, color='b', label=r"$+\overline{u''_r \rho \varepsilon_{nuc} /T}$")
            plt.plot(grd1, rhs6, color='m', label=r"$+\overline{u''_r \nabla \cdot T /T}$")
            plt.plot(grd1, rhs7, color='g', label=r"$+\overline{u''_r \varepsilon_k /T}$")
            plt.plot(grd1, rhs8, color='y', label=r"$+G_s$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_fs$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg K$^{-1}$ cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg K$^{-1}$ cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'fss_eq.png')

    def plot_fss_equation2(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot entropy flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EntropyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_f_ss
        lhs1 = self.minus_div_fht_ux_f_ss

        rhs0 = self.minus_div_fr_ss
        rhs1 = self.minus_f_ss_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_ss
        rhs3 = self.minus_eht_ssddgg
        rhs4 = self.plus_fht_ss_eht_ddgg
        rhs9 = self.minus_eht_ssff_gradx_ppf
        rhs5 = self.plus_eht_uxff_dd_nuc_T
        rhs6 = self.plus_eht_uxff_div_ftt_T
        rhs7 = self.plus_eht_uxff_epsilonk_approx_T
        rhs8 = self.plus_Gss

        res = self.minus_resSSfluxEquation2

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, rhs9, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('entropy flux equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_s$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\widetilde{u}_x f_s$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_s^x $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_s \partial_x \widetilde{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{xx} \partial_x \widetilde{s}$")
            plt.plot(grd1, rhs3, color='c', label=r"$-\overline{s \rho g_x}$")
            plt.plot(grd1, rhs4, color='mediumseagreen', label=r"$+\widetilde{s} \overline{\rho g_x}$")
            plt.plot(grd1, rhs9, color='brown', label=r"$- \overline{s''\partial_x P'}$")
            plt.plot(grd1, rhs5, color='b', label=r"$+\overline{u''_x \rho \varepsilon_{nuc} /T}$")
            plt.plot(grd1, rhs6, color='m', label=r"$+\overline{u''_x \nabla \cdot T /T}$")
            plt.plot(grd1, rhs7, color='g', label=r"$+\overline{u''_x \varepsilon_k /T}$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_fs$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_s$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\widetilde{u}_r f_s$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_s^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_s \partial_r \widetilde{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{rr} \partial_r \widetilde{s}$")
            plt.plot(grd1, rhs3, color='c', label=r"$-\overline{s \rho g_r}$")
            plt.plot(grd1, rhs4, color='mediumseagreen', label=r"$+\widetilde{s} \overline{\rho g_r}$")
            plt.plot(grd1, rhs9, color='brown', label=r"$- \overline{s''\partial_r P'}$")
            plt.plot(grd1, rhs5, color='b', label=r"$+\overline{u''_r \rho \varepsilon_{nuc} /T}$")
            plt.plot(grd1, rhs6, color='m', label=r"$+\overline{u''_r \nabla \cdot T /T}$")
            plt.plot(grd1, rhs7, color='g', label=r"$+\overline{u''_r \varepsilon_k /T}$")
            plt.plot(grd1, rhs8, color='y', label=r"$+G_s$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_fs$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg K$^{-1}$ cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg K$^{-1}$ cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'fss_eq2.png')
