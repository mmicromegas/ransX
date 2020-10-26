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

class EntropyEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, nsdim, tke_diss, data_prefix):
        super(EntropyEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddss = self.getRAdata(eht, 'ddss')[intc]
        ddssux = self.getRAdata(eht, 'ddssux')[intc]

        ddenuc1_o_tt = self.getRAdata(eht, 'ddenuc1_o_tt')[intc]
        ddenuc2_o_tt = self.getRAdata(eht, 'ddenuc2_o_tt')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddss = self.getRAdata(eht, 'ddss')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ss = ddss / dd
        f_ss = ddssux - ddux * ddss / dd

        ##################
        # ENTROPY EQUATION 
        ##################

        # LHS -dq/dt 		
        self.minus_dt_eht_dd_fht_ss = -self.dt(t_ddss, xzn0, t_timec, intc)

        # LHS -div eht_dd fht_ux fht_ss		
        self.minus_div_eht_dd_fht_ux_fht_ss = -self.Div(dd * fht_ux * fht_ss, xzn0)

        # RHS -div fss
        self.minus_div_fss = -self.Div(f_ss, xzn0)

        # RHS -div ftt / T (not included)
        self.minus_div_ftt_T = -np.zeros(nx)

        # RHS +rho enuc / T
        self.plus_eht_dd_enuc_T = ddenuc1_o_tt + ddenuc2_o_tt

        # RHS approx. +diss tke / T
        # self.plus_disstke_T_approx = tke_diss / tt
        self.plus_disstke_T_approx = np.zeros(nx)

        # -res		
        self.minus_resSequation = -(self.minus_dt_eht_dd_fht_ss + self.minus_div_eht_dd_fht_ux_fht_ss +
                                    self.minus_div_fss + self.minus_div_ftt_T + self.plus_eht_dd_enuc_T +
                                    self.plus_disstke_T_approx)

        ######################
        # END ENTROPY EQUATION 
        ######################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fht_ss = fht_ss
        self.fext = fext
        self.nsdim = nsdim

    def plot_ss(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean Favrian entropy stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EntropyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fht_ss

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'entropy')
        plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{s}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\widetilde{s}$ (erg g$^{-1}$ K$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\widetilde{s}$ (erg g$^{-1}$ K$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ss.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ss.eps')

    def plot_ss_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot entropy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EntropyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_fht_ss
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_ss

        rhs0 = self.minus_div_fss
        rhs1 = self.minus_div_ftt_T
        rhs2 = self.plus_eht_dd_enuc_T
        rhs3 = self.plus_disstke_T_approx

        res = self.minus_resSequation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('entropy equation ' + str(self.nsdim) + "D")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \widetilde{s} )$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\overline{\rho}\widetilde{u}_x \widetilde{s}$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_s $")
            plt.plot(grd1, rhs1, color='c', label=r"$-\overline{\nabla_x f_T /T}$ (not incl.)")
            plt.plot(grd1, rhs2, color='b', label=r"$+\overline{\rho\epsilon_{nuc}/T}$")
            plt.plot(grd1, rhs3, color='m', label=r"$+\overline{\varepsilon_k/T}$ set to 0")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_s$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \widetilde{s} )$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{s}$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_s $")
            plt.plot(grd1, rhs1, color='c', label=r"$-\overline{\nabla_r f_T /T}$ (not incl.)")
            plt.plot(grd1, rhs2, color='b', label=r"$+\overline{\rho\epsilon_{nuc}/T}$")
            plt.plot(grd1, rhs3, color='m', label=r"$+\overline{\varepsilon_k/T}$ set to 0")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_s$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg cm$^{-3}$ K$^{-1}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg cm$^{-3}$ K$^{-1}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'ss_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'ss_eq.eps')