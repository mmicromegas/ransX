import numpy as np
from scipy import integrate
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

class TemperatureVarianceEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, tke_diss, tauL, data_prefix):
        super(TemperatureVarianceEquation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]

        ttsq = self.getRAdata(eht, 'ttsq')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
        ttux = self.getRAdata(eht, 'ttux')[intc]
        ttttux = self.getRAdata(eht, 'ttttux')[intc]
        dddivu = self.getRAdata(eht, 'dddivu')[intc]
        ttdivu = self.getRAdata(eht, 'ttdivu')[intc]

        enuc1_o_cv = self.getRAdata(eht, 'enuc1_o_cv')[intc]
        enuc2_o_cv = self.getRAdata(eht, 'enuc2_o_cv')[intc]

        ttenuc1_o_cv = self.getRAdata(eht, 'ttenuc1_o_cv')[intc]
        ttenuc2_o_cv = self.getRAdata(eht, 'ttenuc2_o_cv')[intc]

        ttttdivu = self.getRAdata(eht, 'ttttdivu')[intc]
        ttdivu = self.getRAdata(eht, 'ttdivu')[intc]

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
        t_tt = self.getRAdata(eht, 'tt')
        t_ttsq = self.getRAdata(eht, 'ttsq')

        t_sigma_tt = t_ttsq - t_tt * t_tt

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        ftt = ttux - tt * ux
        fht_d = dddivu / dd
        sigma_tt = ttsq - tt * tt

        eht_ttf_dff = ttdivu - tt * divu
        eht_ttf_ttf_dff = ttttdivu - 2. * ttdivu * tt + tt * tt * divu - ttsq * dddivu / dd + tt * tt * dddivu / dd

        f_sigma_tt = ttttux - 2. * ttux * tt + tt * tt * ux - ttsq * ddux / dd + tt * tt * ddux / dd

        disstke = tke_diss

        ########################################
        # TEMPERATURE VARIANCE SIGMA TT EQUATION 
        ########################################

        # LHS -dt sigma_tt 		
        self.minus_dt_sigma_tt = -self.dt(t_sigma_tt, xzn0, t_timec, intc)

        # LHS -fht_ux gradx sigma_tt
        self.minus_fht_ux_gradx_sigma_tt = -fht_ux * self.Grad(sigma_tt, xzn0)

        # RHS -div f_sigma_tt
        self.minus_div_f_sigma_tt = -self.Div(f_sigma_tt, xzn0)

        # RHS -2_gamma3_minus_one_tt_ttf_dff
        self.minus_two_gamma3_minus_one_tt_ttf_ttff = -2. * (gamma3 - 1.) * tt * eht_ttf_dff

        # RHS minus_two_ftt_gradx_tt
        self.minus_two_ftt_gradx_tt = -2. * ftt * self.Grad(tt, xzn0)

        # RHS minus_two_gamma3_minus_one_fht_d_sigma_tt
        self.minus_two_gamma3_minus_one_fht_d_sigma_tt = -2. * (gamma3 - 1.) * fht_d * sigma_tt

        # RHS plus_three_minus_two_gamma3_eht_ttf_ttf_dff	
        self.plus_three_minus_two_gamma3_eht_ttf_ttf_dff = +(3. - 2. * gamma3) * eht_ttf_ttf_dff

        # RHS plus_two_ttf_dd_enuc_o_cv	
        self.plus_two_ttf_dd_enuc_o_cv = +2. * ((ttenuc1_o_cv + ttenuc2_o_cv) - tt * (enuc1_o_cv + enuc2_o_cv))

        # -res
        self.minus_resSigmaTTequation = -(
                self.minus_dt_sigma_tt + self.minus_fht_ux_gradx_sigma_tt + self.minus_div_f_sigma_tt +
                self.minus_two_gamma3_minus_one_tt_ttf_ttff + self.minus_two_ftt_gradx_tt +
                self.minus_two_gamma3_minus_one_fht_d_sigma_tt + self.minus_two_gamma3_minus_one_fht_d_sigma_tt +
                self.plus_three_minus_two_gamma3_eht_ttf_ttf_dff + self.plus_two_ttf_dd_enuc_o_cv)

        # Kolmogorov dissipation, tauL is Kolmogorov damping timescale 		 
        self.minus_sigmaTTkolmdiss = -sigma_tt / tauL

        ############################################
        # END TEMPERATURE VARIANCE SIGMA TT EQUATION 
        ############################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.sigma_tt = sigma_tt
        self.fext = fext

    def plot_sigma_tt(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean temperature variance stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureVarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.sigma_tt

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'temperature variance')
        plt.plot(grd1, plt1, color='brown', label=r'$\sigma_{T}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\sigma_{T}$ (K$^2)$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\sigma_{T}$ (K$^2)$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_sigma_tt.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_sigma_tt.eps')

    def plot_sigma_tt_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """ temperature variance equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureVarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_sigma_tt
        lhs1 = self.minus_fht_ux_gradx_sigma_tt

        rhs0 = self.minus_div_f_sigma_tt
        rhs1 = self.minus_two_gamma3_minus_one_tt_ttf_ttff
        rhs2 = self.minus_two_ftt_gradx_tt
        rhs3 = self.minus_two_gamma3_minus_one_fht_d_sigma_tt
        rhs4 = self.plus_three_minus_two_gamma3_eht_ttf_ttf_dff
        rhs5 = self.plus_two_ttf_dd_enuc_o_cv

        rhs7 = rhs1 + rhs2

        res = self.minus_resSigmaTTequation

        rhs6 = self.minus_sigmaTTkolmdiss

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        # to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs4, rhs5, rhs6, res]
        to_plot = [lhs0, lhs1, rhs0, rhs4, rhs5, rhs6, rhs7, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # model constant for variance dissipation
        Cm = 0.01

        # plot DATA 
        plt.title(r"tt variance equation C$_m$ = " + str(Cm))
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t \sigma_T$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_x \partial_x \sigma_T$")

            plt.plot(grd1, rhs0, color='r', label=r"$-\nabla f_{\sigma T}$")
            # plt.plot(grd1, rhs1, color='c', label=r"$-2 (\Gamma_3-1) \overline{T} \ \overline{T'd''}$")
            # plt.plot(grd1, rhs2, color='#802A2A', label=r"$-2 f_T \partial_x \overline{T}$")
            plt.plot(grd1, rhs7, color='#802A2A',label = r"$-2 f_T \partial_r \overline{T}-2 (\Gamma_3-1) \overline{T} \ \overline{T'd''}$")
            plt.plot(grd1, rhs3, color='m', label=r"$-2 (\Gamma_3-1) \widetilde{d} \sigma_T$")
            plt.plot(grd1, rhs4, color='g', label=r"$+(3 - 2\Gamma_3)\overline{T'T'd''}$")
            plt.plot(grd1, rhs5, color='olive', label=r"$+2\overline{T' \varepsilon_{nuc}/c_v}$")
            plt.plot(grd1, Cm*rhs6, color='k', linewidth=0.8, label=r"$-C_m \sigma_T / \tau_L$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\sigma$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t \sigma_T$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\widetilde{u}_r \partial_r \sigma_T$")

            plt.plot(grd1, rhs0, color='r', label=r"$-\nabla f_{\sigma T}$")
            # plt.plot(grd1, rhs1, color='c', label=r"$-2 (\Gamma_3-1) \overline{T} \ \overline{T'd''}$")
            # plt.plot(grd1, rhs2, color='#802A2A', label=r"$-2 f_T \partial_r \overline{T}$")
            plt.plot(grd1, rhs7, color='#802A2A',label = r"$-2 f_T \partial_r \overline{T}-2 (\Gamma_3-1) \overline{T} \ \overline{T'd''}$")
            plt.plot(grd1, rhs3, color='m', label=r"$-2 (\Gamma_3-1) \widetilde{d} \sigma_T$")
            plt.plot(grd1, rhs4, color='g', label=r"$+(3 - 2\Gamma_3)\overline{T'T'd''}$")
            plt.plot(grd1, rhs5, color='olive', label=r"$+2\overline{T' \varepsilon_{nuc}/c_v}$")
            plt.plot(grd1, Cm*rhs6, color='k', linewidth=0.8, label=r"$-C_m \sigma_T / \tau_L$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\sigma$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\sigma_{T}$ (K$^2$ s$^{-1})$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\sigma_{T}$ (K$^2$ s$^{-1})$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'sigma_tt_eq.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'sigma_tt_eq.eps')