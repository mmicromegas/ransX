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

class PressureVarianceEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, tke_diss, tauL, data_prefix):
        super(PressureVarianceEquation, self).__init__(ig)

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

        ppsq = self.getRAdata(eht, 'ppsq')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
        ppux = self.getRAdata(eht, 'ppux')[intc]
        ppppux = self.getRAdata(eht, 'ppppux')[intc]
        dddivu = self.getRAdata(eht, 'dddivu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        ppddenuc1 = self.getRAdata(eht, 'ppddenuc1')[intc]
        ppddenuc2 = self.getRAdata(eht, 'ppddenuc2')[intc]

        ppppdivu = self.getRAdata(eht, 'ppppdivu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

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
        t_pp = self.getRAdata(eht, 'pp')
        t_ppsq = self.getRAdata(eht, 'ppsq')

        t_sigma_pp = t_ppsq - t_pp * t_pp

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fpp = ppux - pp * ux
        fht_d = dddivu / dd
        sigma_pp = ppsq - pp * pp

        eht_ppf_dff = ppdivu - pp * divu
        eht_ppf_ppf_dff = ppppdivu - 2. * ppdivu * pp + pp * pp * divu - ppsq * dddivu / dd + pp * pp * dddivu / dd

        f_sigma_pp = ppppux - 2. * ppux * pp + pp * pp * ux - ppsq * ddux / dd + pp * pp * ddux / dd

        disstke = tke_diss

        #####################################
        # PRESSURE VARIANCE SIGMA PP EQUATION 
        #####################################

        # LHS -dt sigma_pp 		
        self.minus_dt_sigma_pp = -self.dt(t_sigma_pp, xzn0, t_timec, intc)

        # LHS -fht_ux gradx sigma_pp
        self.minus_fht_ux_gradx_sigma_pp = -fht_ux * self.Grad(sigma_pp, xzn0)

        # RHS -div f_sigma_pp
        self.minus_div_f_sigma_pp = -self.Div(f_sigma_pp, xzn0)

        # RHS -2_gamma1_pp_ppf_ddff
        self.minus_two_gamma1_pp_ppf_ddff = -2. * gamma1 * pp * eht_ppf_dff

        # RHS minus_two_fpp_gradx_pp
        self.minus_two_fpp_gradx_pp = -2. * fpp * self.Grad(pp, xzn0)

        # RHS minus_two_gamma1_fht_d_sigma_pp
        self.minus_two_gamma1_fht_d_sigma_pp = -2. * gamma1 * fht_d * sigma_pp

        # RHS minus_two_gamma1_minus_one_eht_ppf_ppf_dff	
        self.minus_two_gamma1_minus_one_eht_ppf_ppf_dff = -(2. * gamma1 - 1.) * eht_ppf_ppf_dff

        # RHS plus_two_gamma3_minus_one_ppf_dd_enuc	
        self.plus_two_gamma3_minus_one_ppf_dd_enuc = +(2. * gamma3 - 1.) * (
                (ppddenuc1 + ppddenuc2) - pp * (ddenuc1 + ddenuc2))

        # -res
        self.minus_resSigmaPPequation = -(
                self.minus_dt_sigma_pp + self.minus_fht_ux_gradx_sigma_pp + self.minus_div_f_sigma_pp +
                self.minus_two_gamma1_pp_ppf_ddff + self.minus_two_fpp_gradx_pp + self.minus_two_gamma1_fht_d_sigma_pp +
                self.minus_two_gamma1_minus_one_eht_ppf_ppf_dff + self.plus_two_gamma3_minus_one_ppf_dd_enuc)

        # Kolmogorov dissipation, tauL is Kolmogorov damping timescale 		 
        self.minus_sigmaPPkolmdiss = -sigma_pp / tauL

        #########################################
        # END PRESSURE VARIANCE SIGMA PP EQUATION 
        #########################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.sigma_pp = sigma_pp

    def plot_sigma_pp(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mean pressure variance stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureVarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.sigma_pp

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'pressure variance')
        plt.plot(grd1, plt1, color='brown', label=r'$\sigma_{P}$')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\sigma_{P}$ (erg$^2$ cm$^{-6}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\sigma_{P}$ (erg$^2$ cm$^{-6}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_sigma_pp.png')

    def plot_sigma_pp_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """ pressure variance equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureVarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_sigma_pp
        lhs1 = self.minus_fht_ux_gradx_sigma_pp

        rhs0 = self.minus_div_f_sigma_pp
        rhs1 = self.minus_two_gamma1_pp_ppf_ddff
        rhs2 = self.minus_two_fpp_gradx_pp
        rhs3 = self.minus_two_gamma1_fht_d_sigma_pp
        rhs4 = self.minus_two_gamma1_minus_one_eht_ppf_ppf_dff
        rhs5 = self.plus_two_gamma3_minus_one_ppf_dd_enuc

        res = self.minus_resSigmaPPequation

        rhs6 = self.minus_sigmaPPkolmdiss

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs4, rhs5, rhs6, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # model for variance dissipation
        Cm = 0.5

        # plot DATA 
        plt.title(r"pp variance equation C$_m$ = " + str(Cm))
        if self.ig == 1:
            plt.plot(grd1, -lhs0, color='#FF6EB4', label=r'$-\partial_t \sigma_P$')
            plt.plot(grd1, -lhs1, color='k', label=r"$-\widetilde{u}_x \partial_x \sigma_P$")
            plt.plot(grd1, rhs0, color='r', label=r"$-\nabla f_{\sigma P}$")
            plt.plot(grd1, rhs1, color='c', label=r"$-2 \Gamma_1 \overline{P} \ \overline{P'd''}$")
            plt.plot(grd1, rhs2, color='#802A2A', label=r"$-2 f_P \partial_x \overline{P}$")
            # plt.plot(grd1,rhs1+rhs2,color='#802A2A',label = r"$-2 f_P \partial_r \overline{P}-2 \Gamma_1 \overline{P} \ \overline{P'd''}$")
            plt.plot(grd1, rhs3, color='m', label=r"$+2 \Gamma_1 \widetilde{d} \sigma_P$")
            plt.plot(grd1, rhs4, color='g', label=r"$-2(\Gamma_1 -1)\overline{P'P'd''}$")
            plt.plot(grd1, rhs5, color='olive', label=r"$+2(\Gamma_3 -1)\overline{P' \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, Cm*rhs6, color='k', linewidth=0.8, label=r"$-C_m \sigma_P / \tau_L$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\sigma$")
        elif self.ig == 2:
            plt.plot(grd1, -lhs0, color='#FF6EB4', label=r'$-\partial_t \sigma_P$')
            plt.plot(grd1, -lhs1, color='k', label=r"$-\widetilde{u}_r \partial_r \sigma_P$")
            plt.plot(grd1, rhs0, color='r', label=r"$-\nabla f_{\sigma P}$")
            plt.plot(grd1, rhs1, color='c', label=r"$-2 \Gamma_1 \overline{P} \ \overline{P'd''}$")
            plt.plot(grd1, rhs2, color='#802A2A', label=r"$-2 f_P \partial_r \overline{P}$")
            # plt.plot(grd1,rhs1+rhs2,color='#802A2A',label = r"$-2 f_P \partial_r \overline{P}-2 \Gamma_1 \overline{P} \ \overline{P'd''}$")
            plt.plot(grd1, rhs3, color='m', label=r"$+2 \Gamma_1 \widetilde{d} \sigma_P$")
            plt.plot(grd1, rhs4, color='g', label=r"$-2(\Gamma_1 -1)\overline{P'P'd''}$")
            plt.plot(grd1, rhs5, color='olive', label=r"$+2(\Gamma_3 -1)\overline{P' \rho \varepsilon_{nuc}}$")
            plt.plot(grd1, Cm*rhs6, color='k', linewidth=0.8, label=r"$-\sigma_P / \tau_L$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\sigma$")

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\sigma_{P}$ (erg$^2$ cm$^{-3}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\sigma_{P}$ (erg$^2$ cm$^{-3}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'sigma_pp_eq.png')
