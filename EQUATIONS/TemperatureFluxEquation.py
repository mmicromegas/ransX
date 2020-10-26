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

class TemperatureFluxEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, tke_diss, data_prefix):
        super(TemperatureFluxEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        cv = self.getRAdata(eht, 'cv')[intc]

        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]

        ttux = self.getRAdata(eht, 'ttux')[intc]
        ttuy = self.getRAdata(eht, 'ttuy')[intc]
        ttuz = self.getRAdata(eht, 'ttuz')[intc]

        ttuxux = self.getRAdata(eht, 'ttuxux')[intc]
        ttuyuy = self.getRAdata(eht, 'ttuyuy')[intc]
        ttuzuz = self.getRAdata(eht, 'ttuzuz')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        ddttuxux = self.getRAdata(eht, 'ddttuxux')[intc]
        ddttuyuy = self.getRAdata(eht, 'ddttuyuy')[intc]
        ddttuzuz = self.getRAdata(eht, 'ddttuzuz')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        dddivu = self.getRAdata(eht, 'dddivu')[intc]
        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        ttdivu = self.getRAdata(eht, 'ttdivu')[intc]

        ttgradxpp_o_dd = self.getRAdata(eht, 'ttgradxpp_o_dd')[intc]
        gradxpp_o_dd = self.getRAdata(eht, 'gradxpp_o_dd')[intc]

        uxttdivu = self.getRAdata(eht, 'uxttdivu')[intc]

        uxenuc1_o_cv = self.getRAdata(eht, 'uxenuc1_o_cv')[intc]
        uxenuc2_o_cv = self.getRAdata(eht, 'uxenuc2_o_cv')[intc]

        enuc1_o_cv = self.getRAdata(eht, 'enuc1_o_cv')[intc]
        enuc2_o_cv = self.getRAdata(eht, 'enuc2_o_cv')[intc]

        gamma3 = self.getRAdata(eht, 'gamma3')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma3 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_tt = self.getRAdata(eht, 'tt')
        t_ux = self.getRAdata(eht, 'ux')
        t_ttux = self.getRAdata(eht, 'ttux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_d = dddivu / dd

        ftt = ttux - tt * ux
        fttx = ttuxux - 2. * ux * ttux - tt * uxux - 2. * tt * ux * ux

        eht_uxf_uxff = uxux - ux * ux
        eht_uxf_dff = uxdivu - ux * divu
        eht_ttf_dff = ttdivu - tt * divu

        eht_ttf_gradx_pp_o_dd = ttgradxpp_o_dd - tt * gradxpp_o_dd

        eht_uxf_ttf_dff = uxttdivu - uxdivu * tt - ux * ttdivu + ux * tt * divu - ftt * fht_d + ux * tt * fht_d

        eht_uxf_enuc_o_cv = (uxenuc1_o_cv + uxenuc2_o_cv) - ux * (enuc1_o_cv + enuc2_o_cv)

        eht_uxff_epsilonk_approx_o_cv = (ux - fht_ux) * tke_diss / cv

        Grtt = -(ttuyuy - 2. * uy * ttuy - tt * uyuy - 2. * tt * uy * uy) / xzn0 - \
               (ttuzuz - 2. * uz * ttuz - tt * uzuz - 2. * tt * uz * uz) / xzn0

        ttf_GrM = -(ddttuyuy - tt * dduyuy) / xzn0 - (ddttuzuz - tt * dduzuz) / xzn0

        ###########################
        # TEMPERATURE FLUX EQUATION
        ###########################

        # time-series of temperature flux 
        t_ftt = t_ttux - t_tt * t_ux

        # LHS -dq/dt 		
        self.minus_dt_ftt = -self.dt(t_ftt, xzn0, t_timec, intc)

        # LHS -fht_ux gradx ftt
        self.minus_fht_ux_gradx_ftt = -fht_ux * self.Grad(ftt, xzn0)

        # RHS -div flux temperature flux
        self.minus_div_fttx = -self.Div(fttx, xzn0)

        # RHS -ftt_gradx_fht_ux
        self.minus_ftt_gradx_fht_ux = -ftt * self.Grad(fht_ux, xzn0)

        # RHS -eht_uxf_uxff_gradx_tt
        self.minus_eht_uxf_uxff_gradx_tt = -eht_uxf_uxff * self.Grad(tt, xzn0)

        # RHS -eht_eht_ttf_gradx_pp_o_dd
        # self.minus_eht_ttf_gradx_pp_o_dd = -(eht_ttf_gradx_pp_o_dd)
        self.minus_eht_ttf_gradx_pp_o_dd = np.zeros(nx)  # replace gradx pp with rho gg and you get the 0

        # RHS -gamma3_minus_one_tt_eht_uxf_dff
        self.minus_gamma3_minus_one_tt_eht_uxf_dff = -(gamma3 - 1.) * tt * eht_uxf_dff

        # RHS -gamma3_minus_one_fht_d_ftt
        self.minus_gamma3_minus_one_fht_d_ftt = -(gamma3 - 1.) * fht_d * ftt

        # RHS -gamma3_eht_uxf_ttf_dff
        self.minus_gamma3_eht_uxf_ttf_dff = -gamma3 * eht_uxf_ttf_dff

        # RHS eht_uxf_enuc_o_cv	
        self.plus_eht_uxf_enuc_o_cv = (uxenuc1_o_cv + uxenuc2_o_cv) - ux * (enuc1_o_cv + enuc2_o_cv)

        # RHS eht_uxf_div_fth_o_cv (not calculated)
        # fth is flux due to thermal transport (conduction/radiation)
        eht_uxf_div_fth_o_cv = np.zeros(nx)
        self.plus_eht_uxf_div_fth_o_cv = eht_uxf_div_fth_o_cv

        # RHS Gtt
        # self.plus_Gtt = -Grtt-ttf_GrM
        self.plus_Gtt = np.zeros(nx)

        # -res  
        self.minus_resTTfluxEquation = -(self.minus_dt_ftt + self.minus_fht_ux_gradx_ftt + self.minus_div_fttx + self.minus_ftt_gradx_fht_ux + self.minus_eht_uxf_uxff_gradx_tt + self.minus_eht_ttf_gradx_pp_o_dd + self.minus_gamma3_minus_one_tt_eht_uxf_dff + self.minus_gamma3_minus_one_fht_d_ftt + self.minus_gamma3_eht_uxf_ttf_dff + self.plus_eht_uxf_enuc_o_cv + self.plus_eht_uxf_div_fth_o_cv + self.plus_Gtt)

        ###############################
        # END TEMPERATURE FLUX EQUATION
        ###############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ftt = ftt
        self.fext = fext

    def plot_ftt(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot temperature flux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ftt

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'temperature flux')
        plt.plot(grd1, plt1, color='brown', label=r'f$_T$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$f_T$ (K cm s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$f_T$ (K cm s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ftt.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ftt.eps')

    def plot_ftt_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot temperature flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureFluxEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_ftt
        lhs1 = self.minus_fht_ux_gradx_ftt

        rhs0 = self.minus_div_fttx
        rhs1 = self.minus_ftt_gradx_fht_ux
        rhs2 = self.minus_eht_uxf_uxff_gradx_tt
        rhs3 = self.minus_eht_ttf_gradx_pp_o_dd
        rhs4 = self.minus_gamma3_minus_one_tt_eht_uxf_dff
        rhs5 = self.minus_gamma3_minus_one_fht_d_ftt
        rhs6 = self.minus_gamma3_eht_uxf_ttf_dff
        rhs7 = self.plus_eht_uxf_enuc_o_cv
        rhs8 = self.plus_eht_uxf_div_fth_o_cv
        rhs9 = self.plus_Gtt

        res = self.minus_resTTfluxEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs3, rhs2, rhs4, rhs5, rhs6, rhs7, rhs8, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('temperature flux equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_T$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_x \partial_x f_T$")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_T^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_T \partial_x \widetilde{u}_x$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\overline{u'_x u''_x} \partial_x \overline{T}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\overline{T'\partial_r P / \rho}$")
            plt.plot(grd1, rhs4, color='c', label=r"$-(\Gamma_3 -1)\overline{T} \ \overline{u'_x d''}$")
            # plt.plot(grd1,rhs2+rhs4,color='r',label = r"$-\overline{u'_r u''_r} \partial_r \overline{T}-(\Gamma_3 -1)\overline{T} \ \overline{u'_r d''}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$-(\Gamma_3 -1)\widetilde{d} f_T$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\Gamma_3 \overline{u'_x T' d''}$")
            plt.plot(grd1, rhs7, color='g', label=r"$+\overline{u'_x \varepsilon_{nuc} / c_v }$")
            plt.plot(grd1, rhs8, color='m', label=r"$+\overline{u'_x \nabla \cdot T / c_v}$ (not calc.)")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_T$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t f_T$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_r \partial_r f_T$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_T^r $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-f_T \partial_r \widetilde{u}_r$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\overline{u'_r u''_r} \partial_r \overline{T}$")
            plt.plot(grd1, rhs3, color='firebrick', label=r"$-\overline{T'\partial_r P / \rho}$")
            plt.plot(grd1, rhs4, color='c', label=r"$-(\Gamma_3 -1)\overline{T} \ \overline{u'_r d''}$")
            # plt.plot(grd1,rhs2+rhs4,color='r',label = r"$-\overline{u'_r u''_r} \partial_r \overline{T}-(\Gamma_3 -1)\overline{T} \ \overline{u'_r d''}$")
            plt.plot(grd1, rhs5, color='mediumseagreen', label=r"$-(\Gamma_3 -1)\widetilde{d} f_T$")
            plt.plot(grd1, rhs6, color='b', label=r"$+\Gamma_3 \overline{u'_r T' d''}$")
            plt.plot(grd1, rhs7, color='g', label=r"$+\overline{u'_r \varepsilon_{nuc} / c_v }$")
            plt.plot(grd1, rhs8, color='m', label=r"$+\overline{u'_r \nabla \cdot T / c_v}$ (not calc.)")
            plt.plot(grd1, rhs9, color='y', label=r"$+G_T$ (not calc.)")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_T$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"K cm$^{-2}$ s$^{-2}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"K cm$^{-2}$ s$^{-2}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'ftt_eq.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'ftt_eq.eps')