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

class EnthalpyFluxEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, tke_diss, data_prefix):
        super(EnthalpyFluxEquation, self).__init__(ig)

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
        hh = self.getRAdata(eht, 'hh')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]
        ddhh = self.getRAdata(eht, 'ddhh')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        ddhhux = self.getRAdata(eht, 'ddhhux')[intc]
        ddhhuy = self.getRAdata(eht, 'ddhhuy')[intc]
        ddhhuz = self.getRAdata(eht, 'ddhhuz')[intc]

        ddhhuxux = self.getRAdata(eht, 'ddhhuxux')[intc]
        ddhhuyuy = self.getRAdata(eht, 'ddhhuyuy')[intc]
        ddhhuzuz = self.getRAdata(eht, 'ddhhuzuz')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        dduxenuc1 = self.getRAdata(eht, 'dduxenuc1')[intc]
        dduxenuc2 = self.getRAdata(eht, 'dduxenuc2')[intc]

        hhgradxpp = self.getRAdata(eht, 'hhgradxpp')[intc]

        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        uxppdivu = self.getRAdata(eht, 'uxppdivu')[intc]

        gamma1 = self.getRAdata(eht, 'gamma1')[intc]
        gamma3 = self.getRAdata(eht, 'gamma3')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddux = self.getRAdata(eht, 'ddux')
        t_ddhh = self.getRAdata(eht, 'ddhh')
        t_ddhhux = self.getRAdata(eht, 'ddhhux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_hh = ddhh / dd
        rxx = dduxux - ddux * ddux / dd

        fhh = ddhhux - ddux * ddhh / dd
        fhhx = ddhhuxux - ddhh * dduxux / dd - 2. * fht_ux * ddhhux + 2. * dd * fht_ux * fht_hh * fht_ux

        eht_hhff = hh - ddhh / dd
        eht_hhff_gradx_ppf = hhgradxpp - fht_hh * self.Grad(pp, xzn0)

        eht_uxff_dd_enuc = (dduxenuc1 + dduxenuc2) - fht_ux * (ddenuc1 + ddenuc2)

        eht_uxff_epsilonk_approx = (ux - ddux / dd) * tke_diss

        Grhh = -(ddhhuyuy - ddhh * dduyuy / dd - 2. * (dduy / dd) * (ddhhuy / dd) + 2. * ddhh * dduy * dduy /
                 (dd * dd * dd)) / xzn0 - \
               (ddhhuzuz - ddhh * dduzuz / dd - 2. * (dduz / dd) * (ddhhuz / dd) + 2. * ddhh * dduz * dduz /
                (dd * dd * dd)) / xzn0

        hhff_GrM = -(ddhhuyuy - (ddhh / dd) * dduyuy) / xzn0 - (ddhhuzuz - (ddhh / dd) * dduzuz) / xzn0

        ########################
        # ENTHALPY FLUX EQUATION
        ########################

        # time-series of enthalpy flux 
        t_fhh = t_ddhhux / t_dd - t_ddhh * t_ddux / (t_dd * t_dd)

        # LHS -dq/dt 		
        self.minus_dt_fhh = -self.dt(t_fhh, xzn0, t_timec, intc)

        # LHS -div fht_ux fhh
        self.minus_div_fht_ux_fhh = -self.Div(fht_ux * fhh, xzn0)

        # RHS -div enthalpy flux
        self.minus_div_fhhx = -self.Div(fhhx, xzn0)

        # RHS -fhh_gradx_fht_ux
        self.minus_fhh_gradx_fht_ux = -fhh * self.Grad(fht_ux, xzn0)

        # RHS -rxx_gradx_fht_hh
        self.minus_rxx_gradx_fht_hh = -rxx * self.Grad(fht_hh, xzn0)

        # RHS -eht_hhff_gradx_eht_pp
        self.minus_eht_hhff_gradx_eht_pp = -eht_hhff * self.Grad(pp, xzn0)

        # RHS -eht_hhff_gradx_ppf
        self.minus_eht_hhff_gradx_ppf = -(eht_hhff_gradx_ppf)

        # RHS -gamma1_eht_uxff_pp_divu
        self.minus_gamma1_eht_uxff_pp_divu = -gamma1 * (uxppdivu - fht_ux * ppdivu)

        # RHS +gamma3_eht_uxff_dd_nuc	
        self.plus_gamma3_eht_uxff_dd_nuc = +gamma3 * (eht_uxff_dd_enuc)

        # RHS +gamma3_eht_uxff_div_fth (not calculated)
        # fth is flux due to thermal transport (conduction/radiation)
        eht_uxff_div_fth = np.zeros(nx)
        self.plus_gamma3_eht_uxff_div_fth = +gamma3 * eht_uxff_div_fth

        # RHS gamma3_eht_uxff_epsilonk_approx	
        self.plus_gamma3_eht_uxff_epsilonk_approx = +gamma3 * eht_uxff_epsilonk_approx

        # RHS Ghh
        if self.ig == 1:
            self.plus_Ghh = np.zeros(nx)
        elif self.ig == 2:
            self.plus_Ghh = -Grhh - hhff_GrM
        else:
            print("ERROR(EnthalpyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # -res  
        self.minus_resHHfluxEquation = -(self.minus_dt_fhh + self.minus_div_fht_ux_fhh +
                                         self.minus_div_fhhx + self.minus_fhh_gradx_fht_ux + self.minus_rxx_gradx_fht_hh +
                                         self.minus_eht_hhff_gradx_eht_pp + self.minus_eht_hhff_gradx_ppf +
                                         self.minus_gamma1_eht_uxff_pp_divu + self.plus_gamma3_eht_uxff_dd_nuc +
                                         self.plus_gamma3_eht_uxff_div_fth + self.plus_gamma3_eht_uxff_epsilonk_approx +
                                         self.plus_Ghh)

        ############################
        # END ENTHALPY FLUX EQUATION
        ############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fhh = fhh

    def plot_fhh(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean Favrian enthalpy flux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EnthalpyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fhh

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'enthalpy flux')
        plt.plot(grd1, plt1, color='brown', label=r'f$_h$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$f_h$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$f_h$ (erg cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_fhh.png')

    def plot_fhh_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot internal energy flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EnthalpyFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fhh
        lhs1 = self.minus_div_fht_ux_fhh

        rhs0 = self.minus_div_fhhx
        rhs1 = self.minus_fhh_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_hh
        rhs3 = self.minus_gamma1_eht_uxff_pp_divu
        rhs4 = self.minus_eht_hhff_gradx_eht_pp
        rhs5 = self.minus_eht_hhff_gradx_ppf
        rhs6 = self.plus_gamma3_eht_uxff_dd_nuc
        rhs7 = self.plus_gamma3_eht_uxff_div_fth
        rhs8 = self.plus_gamma3_eht_uxff_epsilonk_approx
        rhs9 = self.plus_Ghh

        res = self.minus_resHHfluxEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, rhs9, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('enthalpy flux equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_h$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\widetilde{u}_x f_h$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_h^x $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_h \partial_x \widetilde{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{xx} \partial_x \widetilde{h}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\Gamma_1 \overline{u''_x P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$-\overline{h''}\partial_x \overline{P}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$-\overline{h'' \partial_x P'}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\Gamma_3 \overline{u''_x \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\Gamma_3 \overline{u''_x \nabla \cdot T}$")
            plt.plot(grd1, rhs8, color='g', label=r"$+\Gamma_3 \overline{u''_x \varepsilon_k }$")
            #plt.plot(grd1, rhs9, color='y', label=r"$+G_h$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_h$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_h$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\widetilde{u}_r f_h$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_h^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_h \partial_r \widetilde{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\widetilde{R}_{rr} \partial_r \widetilde{h}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\Gamma_1 \overline{u''_r P d}$")
            plt.plot(grd1, rhs4, color='c', label=r"$-\overline{h''}\partial_r \overline{P}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$-\overline{h'' \partial_r P'}$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\Gamma_3 \overline{u''_r \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, rhs7, color='m', label=r"$+\Gamma_3 \overline{u''_r \nabla \cdot T}$")
            plt.plot(grd1, rhs8, color='g', label=r"$+\Gamma_3 \overline{u''_r \varepsilon_k }$")
            plt.plot(grd1, rhs9, color='y', label=r"$+G_h$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_h$")

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
        plt.savefig('RESULTS/' + self.data_prefix + 'fhh_eq.png')
