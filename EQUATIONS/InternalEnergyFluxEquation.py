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

class InternalEnergyFluxEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, tke_diss, data_prefix):
        super(InternalEnergyFluxEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        ei = self.getRAdata(eht, 'ei')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]
        ddei = self.getRAdata(eht, 'ddei')[intc]
        ddgg = self.getRAdata(eht, 'ddgg')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        ddeiux = self.getRAdata(eht, 'ddeiux')[intc]
        ddeiuy = self.getRAdata(eht, 'ddeiuy')[intc]
        ddeiuz = self.getRAdata(eht, 'ddeiuz')[intc]
        eiddgg = self.getRAdata(eht, 'eiddgg')[intc]

        ddeiuxux = self.getRAdata(eht, 'ddeiuxux')[intc]
        ddeiuyuy = self.getRAdata(eht, 'ddeiuyuy')[intc]
        ddeiuzuz = self.getRAdata(eht, 'ddeiuzuz')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        dduxenuc1 = self.getRAdata(eht, 'dduxenuc1')[intc]
        dduxenuc2 = self.getRAdata(eht, 'dduxenuc2')[intc]

        eigradxpp = self.getRAdata(eht, 'eigradxpp')[intc]

        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        uxppdivu = self.getRAdata(eht, 'uxppdivu')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddux = self.getRAdata(eht, 'ddux')
        t_ddei = self.getRAdata(eht, 'ddei')
        t_ddeiux = self.getRAdata(eht, 'ddeiux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ei = ddei / dd
        rxx = dduxux - ddux * ddux / dd

        fei = ddeiux - ddux * ddei / dd
        feix = ddeiuxux - ddei * dduxux / dd - 2. * fht_ux * ddeiux + 2. * dd * fht_ux * fht_ei * fht_ux

        eht_eiff = ei - ddei / dd
        eht_eiff_gradx_ppf = eigradxpp - ei * self.Grad(pp, xzn0)

        eht_uxff_dd_enuc = (dduxenuc1 + dduxenuc2) - fht_ux * (ddenuc1 + ddenuc2)

        eht_uxff_epsilonk_approx = (ux - ddux / dd) * tke_diss

        Grei = -(ddeiuyuy - ddei * dduyuy / dd - 2. * (dduy / dd) * (ddeiuy / dd) + 2. * ddei * dduy * dduy / (
                dd * dd * dd)) / xzn0 - \
               (ddeiuzuz - ddei * dduzuz / dd - 2. * (dduz / dd) * (ddeiuz / dd) + 2. * ddei * dduz * dduz / (
                       dd * dd * dd)) / xzn0

        eiff_GrM = -(ddeiuyuy - (ddei / dd) * dduyuy) / xzn0 - (ddeiuzuz - (ddei / dd) * dduzuz) / xzn0

        ###############################
        # INTERNAL ENERGY FLUX EQUATION
        ###############################

        # time-series of internal energy flux 
        t_fei = t_ddeiux / t_dd - t_ddei * t_ddux / (t_dd * t_dd)

        # LHS -dq/dt 		
        self.minus_dt_fei = -self.dt(t_fei, xzn0, t_timec, intc)

        # LHS -div fht_ux fei
        self.minus_div_fht_ux_fei = -self.Div(fht_ux * fei, xzn0)

        # RHS -div internal energy flux
        self.minus_div_feix = -self.Div(feix, xzn0)

        # RHS -fei_gradx_fht_ux
        self.minus_fei_gradx_fht_ux = -fei * self.Grad(fht_ux, xzn0)

        # RHS -rxx_gradx_fht_ei
        self.minus_rxx_gradx_fht_ei = -rxx * self.Grad(fht_ei, xzn0)

        # RHS -eht_eiff_gradx_eht_pp
        self.minus_eht_eiff_gradx_eht_pp = -eht_eiff * self.Grad(pp, xzn0)

        # RHS -eht_eiff_gradx_ppf
        self.minus_eht_eiff_gradx_ppf = -(eht_eiff_gradx_ppf)

        # RHS -eht_uxff_pp_divu
        self.minus_eht_uxff_pp_divu = -(uxppdivu - fht_ux * ppdivu)

        # RHS eht_uxff_dd_nuc	
        self.plus_eht_uxff_dd_nuc = +(eht_uxff_dd_enuc)

        # RHS eht_uxff_div_fth (not calculated)
        # fth is flux due to thermal transport (conduction/radiation)
        eht_uxff_div_fth = np.zeros(nx)
        self.plus_eht_uxff_div_fth = +eht_uxff_div_fth

        # RHS eht_uxff_epsilonk_approx	
        self.plus_eht_uxff_epsilonk_approx = +eht_uxff_epsilonk_approx

        # RHS Gei
        if self.ig == 1:
            self.plus_Gei = np.zeros(nx)
        elif self.ig == 2:
            self.plus_Gei = -Grei - eiff_GrM
        else:
            print("ERROR(InternalEnergyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # -res  
        self.minus_resEiFluxEquation = -(self.minus_dt_fei + self.minus_div_fht_ux_fei +
                                         self.minus_div_feix + self.minus_fei_gradx_fht_ux +
                                         self.minus_rxx_gradx_fht_ei +
                                         self.minus_eht_eiff_gradx_eht_pp +
                                         self.minus_eht_eiff_gradx_ppf +
                                         self.minus_eht_uxff_pp_divu +
                                         self.plus_eht_uxff_dd_nuc +
                                         self.plus_eht_uxff_div_fth +
                                         self.plus_eht_uxff_epsilonk_approx +
                                         self.plus_Gei)

        ###################################
        # END INTERNAL ENERGY FLUX EQUATION
        ###################################

        # RHS -eht_eiddgg
        self.minus_eht_eiddgg = -eiddgg

        # RHS +fht_ei_eht_ddgg		
        self.plus_fht_ei_eht_ddgg = +fht_ei * ddgg

        # -res  
        self.minus_resEiFluxEquation2 = -(self.minus_dt_fei + self.minus_div_fht_ux_fei +
                                          self.minus_div_feix + self.minus_fei_gradx_fht_ux +
                                          self.minus_rxx_gradx_fht_ei +
                                          self.minus_eht_eiddgg + self.plus_fht_ei_eht_ddgg +
                                          self.minus_eht_uxff_pp_divu +
                                          self.plus_eht_uxff_dd_nuc + self.plus_eht_uxff_div_fth +
                                          self.plus_eht_uxff_epsilonk_approx +
                                          self.plus_Gei)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fei = fei

    def plot_fei(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean Favrian internal energy flux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(InternalEnergyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fei

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'internal energy flux')
        plt.plot(grd1, plt1, color='brown', label=r'f$_I$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$f_I$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$f_I$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_fei.png')

    def plot_fei_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot internal energy flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(InternalEnergyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fei
        lhs1 = self.minus_div_fht_ux_fei

        rhs0 = self.minus_div_feix
        rhs1 = self.minus_fei_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_ei
        rhs3 = self.minus_eht_uxff_pp_divu
        rhs4 = self.minus_eht_eiff_gradx_eht_pp
        rhs5 = self.minus_eht_eiff_gradx_ppf
        rhs6 = self.plus_eht_uxff_dd_nuc
        rhs7 = self.plus_eht_uxff_div_fth
        rhs8 = self.plus_eht_uxff_epsilonk_approx
        rhs9 = self.plus_Gei

        res = self.minus_resEiFluxEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, rhs9, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('internal energy flux equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_I$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\widetilde{u}_x f_I$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_i^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_i \partial_x \widetilde{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{xx} \partial_x \widetilde{\epsilon_I}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\overline{u''_x P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$-\overline{\epsilon''_I}\partial_x \overline{P}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$-\overline{\epsilon''_I \partial_x P'}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\overline{u''_x \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{u''_x \nabla \cdot T}$ (not calc.)")
            plt.plot(grd1, rhs8, color='g', label=r"$+\overline{u''_x \varepsilon_k }$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\epsilon$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_I$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\widetilde{u}_r f_I$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_i^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_i \partial_r \widetilde{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{rr} \partial_r \widetilde{\epsilon_I}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\overline{u''_r P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$-\overline{\epsilon''_I}\partial_r \overline{P}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$-\overline{\epsilon''_I \partial_r P'}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\overline{u''_r \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{u''_r \nabla \cdot T}$ (not calc.)")
            plt.plot(grd1, rhs8, color='g', label=r"$+\overline{u''_r \varepsilon_k }$")
            plt.plot(grd1, rhs9, color='y', label=r"$+G_{\epsilon}$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\epsilon$")

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
        plt.savefig('RESULTS/' + self.data_prefix + 'fei_eq.png')

    def plot_fei_equation2(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot internal energy flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(InternalEnergyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fei
        lhs1 = self.minus_div_fht_ux_fei

        rhs0 = self.minus_div_feix
        rhs1 = self.minus_fei_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_ei
        rhs3 = self.minus_eht_uxff_pp_divu

        # rhs4 = self.minus_eht_eiff_gradx_eht_pp
        # rhs5 = self.minus_eht_eiff_gradx_ppf

        rhs4 = self.minus_eht_eiddgg
        rhs5 = self.plus_fht_ei_eht_ddgg

        rhs4_plus_rhs5 = rhs4 + rhs5

        rhs6 = self.plus_eht_uxff_dd_nuc
        rhs7 = self.plus_eht_uxff_div_fth
        rhs8 = self.plus_eht_uxff_epsilonk_approx
        rhs9 = self.plus_Gei

        res = self.minus_resEiFluxEquation2

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4_plus_rhs5, rhs6, rhs7, rhs8, rhs9, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title('internal energy flux equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_I$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\widetilde{u}_x f_I$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_i^x $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_i \partial_x \widetilde{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{xx} \partial_x \widetilde{\epsilon_I}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\overline{u''_x P d}$")
            # plt.plot(grd1,rhs4,color='c',label = r"$-\overline{\epsilon_I \rho g_x}$")
            # plt.plot(grd1,rhs5,color='mediumseagreen',label = r"$-\widetilde{\epsilon_I}\overline{\rho g_x}$")
            plt.plot(grd1, rhs4_plus_rhs5, color='c',
                     label=r"$-\overline{\epsilon_I \rho g_x}-\widetilde{\epsilon_I}\overline{\rho g_x}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\overline{u''_x \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{u''_x \nabla \cdot T}$ (not calc.)")
            plt.plot(grd1, rhs8, color='g', label=r"$+\overline{u''_x \varepsilon_k }$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\epsilon$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_I$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\widetilde{u}_r f_I$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_i^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_i \partial_r \widetilde{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{rr} \partial_r \widetilde{\epsilon_I}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\overline{u''_r P d}$")
            # plt.plot(grd1,rhs4,color='c',label = r"$-\overline{\epsilon_I \rho g_r}$")
            # plt.plot(grd1,rhs5,color='mediumseagreen',label = r"$-\widetilde{\epsilon_I}\overline{\rho g_r}$")
            plt.plot(grd1, rhs4_plus_rhs5, color='c',
                     label=r"$-\overline{\epsilon_I \rho g_r}-\widetilde{\epsilon_I}\overline{\rho g_r}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\overline{u''_r \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\overline{u''_r \nabla \cdot T}$ (not calc.)")
            plt.plot(grd1, rhs8, color='g', label=r"$+\overline{u''_r \varepsilon_k }$")
            plt.plot(grd1, rhs9, color='y', label=r"$+G_{\epsilon}$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\epsilon$")

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
        plt.savefig('RESULTS/' + self.data_prefix + 'fei_eq2.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'fei_eq2.eps')
