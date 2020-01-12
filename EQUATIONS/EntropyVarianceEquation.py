import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al
import UTILS.Tools as uT
import UTILS.Errors as eR

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class EntropyVarianceEquation(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, tke_diss, tauL, data_prefix):
        super(EntropyVarianceEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht,'xzn0')
        nx = self.getRAdata(eht,'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht,'dd')[intc]
        ux = self.getRAdata(eht,'ux')[intc]
        pp = self.getRAdata(eht,'pp')[intc]
        tt = self.getRAdata(eht,'tt')[intc]
        ss = self.getRAdata(eht,'ss')[intc]

        ddux = self.getRAdata(eht,'ddux')[intc]
        ddei = self.getRAdata(eht,'ddei')[intc]
        ddss = self.getRAdata(eht,'ddss')[intc]
        ddssux = self.getRAdata(eht,'ddssux')[intc]
        ddsssq = self.getRAdata(eht,'ddsssq')[intc]

        ddssssux = self.getRAdata(eht,'ddssssux')[intc]

        ddenuc1_o_tt = self.getRAdata(eht,'ddenuc1_o_tt')[intc]
        ddenuc2_o_tt = self.getRAdata(eht,'ddenuc2_o_tt')[intc]

        ddssenuc1_o_tt = self.getRAdata(eht,'ddssenuc1_o_tt')[intc]
        ddssenuc2_o_tt = self.getRAdata(eht,'ddssenuc2_o_tt')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht,'timec')
        t_dd = self.getRAdata(eht,'dd')
        t_ddss = self.getRAdata(eht,'ddss')
        t_ddsssq = self.getRAdata(eht,'ddsssq')

        t_sigma_ss = (t_ddsssq / t_dd) - (t_ddss * t_ddss) / (t_dd * t_dd)

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ss = ddss / dd
        f_ss = ddssux - ddux * ddss / dd
        sigma_ss = (ddsssq / dd) - (ddss * ddss) / (dd * dd)

        f_sigma_ss = dd * (ddssssux / dd - 2. * ddss * ddssux / (dd * dd) - ddux * ddsssq / (dd * dd) + \
                           2. * (ddss * ddss * ddux) / (dd * dd * dd))

        disstke_o_tt = tke_diss / tt

        ###########################
        # ENTROPY VARIANCE EQUATION
        ###########################

        # LHS -dt dd sigma_ss 		
        self.minus_dt_eht_dd_sigma_ss = -self.dt(t_dd * t_sigma_ss, xzn0, t_timec, intc)

        # LHS -div dd fht_ux sigma_ss
        self.minus_div_eht_dd_fht_ux_sigma_ss = -self.Div(dd * fht_ux * sigma_ss, xzn0)

        # RHS -div f_sigma_ss
        self.minus_div_f_sigma_ss = -self.Div(f_sigma_ss, xzn0)

        # RHS minus_two_f_ss_gradx_fht_ss
        self.minus_two_f_ss_gradx_fht_ss = -2. * f_ss * self.Grad(fht_ss, xzn0)

        # RHS minus_two_ssff_div_ftt_T
        self.minus_two_ssff_div_ftt_T = +np.zeros(nx)

        # RHS plus_two_ssff_enuc_T	
        self.plus_two_ssff_enuc_T = +2. * ((ddssenuc1_o_tt + ddssenuc2_o_tt) - \
                                           (ddss / dd) * (ddenuc1_o_tt + ddenuc2_o_tt))

        # RHS plus_two_ssff_epsilonk_o_tt_approx		
        self.plus_two_ssff_epsilonk_o_tt_approx = +2. * (ss - ddss / dd) * disstke_o_tt

        # -res 
        self.minus_resSigmaSSequation = -(self.minus_dt_eht_dd_sigma_ss + self.minus_div_eht_dd_fht_ux_sigma_ss + \
                                          self.minus_div_f_sigma_ss + self.minus_two_f_ss_gradx_fht_ss + self.minus_two_ssff_div_ftt_T + \
                                          self.plus_two_ssff_enuc_T + self.plus_two_ssff_epsilonk_o_tt_approx)

        # Kolmogorov dissipation, tauL is Kolmogorov damping timescale 		 
        self.minus_sigmaSSkolmdiss = -dd * sigma_ss / tauL

        ###############################		
        # END ENTROPY VARIANCE EQUATION
        ###############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.sigma_ss = sigma_ss

    def plot_sigma_ss(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mean Favrian entropy variance stratification in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.sigma_ss

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'entropy variance')
        plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{\sigma}_s$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_s$ (erg$^2$ g$^{-2}$ K$^{-2})$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_sigma_ss.png')

    def plot_sigma_ss_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot entropy variance equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_sigma_ss
        lhs1 = self.minus_div_eht_dd_fht_ux_sigma_ss

        rhs0 = self.minus_div_f_sigma_ss
        rhs1 = self.minus_two_f_ss_gradx_fht_ss
        rhs2 = self.minus_two_ssff_div_ftt_T
        rhs3 = self.plus_two_ssff_enuc_T
        rhs4 = self.plus_two_ssff_epsilonk_o_tt_approx

        res = self.minus_resSigmaSSequation

        rhs5 = self.minus_sigmaSSkolmdiss

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('entropy variance equation')
        plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \sigma_s)$")
        plt.plot(grd1, lhs1, color='g', label=r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \sigma_s $)")

        plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_{\sigma_s} $")
        plt.plot(grd1, rhs1, color='r', label=r"$2 f_s \partial_r \widetilde{s}$")
        plt.plot(grd1, rhs2, color='c', label=r"$-\overline{\nabla_r f_T /T}$ (not incl.)")
        plt.plot(grd1, rhs3, color='b', label=r"$+\overline{2 \rho s'' \epsilon_{nuc}/T}$")
        plt.plot(grd1, rhs4, color='m', label=r"$+2\overline{\rho s'' \varepsilon_{k}/T}$ approx.")
        plt.plot(grd1, rhs5, color='k', linewidth=0.8, label=r"$-\overline{\rho} \sigma_s / \tau_L$")

        plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_{\sigma_s}$")

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg$^2$ g$^{-1}$ K$^{-2}$ cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'sigma_ss_eq.png')
