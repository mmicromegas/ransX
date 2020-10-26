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

class PressureEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, nsdim, tke_diss, data_prefix):
        super(PressureEquation, self).__init__(ig)

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

        ddux = self.getRAdata(eht, 'ddux')[intc]
        ppux = self.getRAdata(eht, 'ppux')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

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
        t_dd = self.getRAdata(eht, 'dd')
        t_gamma1 = self.getRAdata(eht, 'gamma1')

        # A = p / rho ** gamma
        onedu = 1.820940e06
        onepu = 4.607893e23
        t_A = (t_pp / onepu) / ((t_dd / onedu) ** (5. / 3.))
        dAdt = self.dt(t_A, xzn0, t_timec, intc)

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fpp = ppux - pp * ux

        ###################
        # PRESSURE EQUATION 
        ###################

        # LHS -dq/dt 		
        self.minus_dt_pp = -self.dt(t_pp, xzn0, t_timec, intc)

        # LHS -fht_ux grad P		
        self.minus_eht_ux_grad_pp = -ux * self.Grad(pp, xzn0)

        # RHS -div fpp
        self.minus_div_fpp = -self.Div(fpp, xzn0)

        # RHS -gamma1 P d = - gamma1 pp Div ux
        self.minus_gamma1_pp_div_ux = -gamma1 * pp * self.Div(ux, xzn0)

        # RHS +(1-gamma1) Wp = +(1-gamma1) eht_ppf_df
        self.plus_one_minus_gamma1_eht_ppf_df = +(1. - gamma1) * (ppdivu - pp * divu)

        # RHS source +(gamma3-1)*dd enuc
        self.plus_gamma3_minus_one_dd_fht_enuc = (gamma3 - 1.) * (ddenuc1 + ddenuc2)

        # RHS +(gamma3-1) dissipated turbulent kinetic energy
        self.plus_gamma3_minus_one_disstke = +(gamma3 - 1.) * tke_diss

        # -res
        self.minus_resPPequation = -(self.minus_dt_pp + self.minus_eht_ux_grad_pp +
                                     self.minus_div_fpp + self.minus_gamma1_pp_div_ux +
                                     self.plus_one_minus_gamma1_eht_ppf_df +
                                     self.plus_gamma3_minus_one_dd_fht_enuc +
                                     self.plus_gamma3_minus_one_disstke)

        #######################
        # END PRESSURE EQUATION 
        #######################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.pp = pp
        self.dAdt = dAdt
        self.fext = fext
        self.nsdim = nsdim

    def plot_pp(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean pressure stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.pp

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'pressure')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{P}$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{P}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define/show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\overline{P}$ (erg cm$^{-3}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\overline{P}$ (erg cm$^{-3}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_pp.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_pp.eps')

    def plot_pp_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot pressure equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(PressureEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_pp
        lhs1 = self.minus_eht_ux_grad_pp

        rhs0 = self.minus_div_fpp
        rhs1 = self.minus_gamma1_pp_div_ux
        rhs2 = self.plus_one_minus_gamma1_eht_ppf_df
        rhs3 = self.plus_gamma3_minus_one_dd_fht_enuc
        rhs4 = self.plus_gamma3_minus_one_disstke

        res = self.minus_resPPequation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('pressure equation ' + str(self.nsdim) + " D")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{P})$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\overline{u}_x \partial_x \overline{P}$")
            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_p $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-\Gamma_1 \bar{P} \bar{d}$")
            plt.plot(grd1, rhs2, color='r', label=r"$+(1-\Gamma_1) W_P$")
            plt.plot(grd1, rhs3, color='b', label=r"$+(\Gamma_3 -1)(\overline{\rho}\widetilde{\epsilon}_{nuc})$")
            plt.plot(grd1, rhs4, color='m', label=r"$+(\Gamma_3 -1)\varepsilon_k$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_P$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{P})$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\overline{u}_r \partial_r \overline{P}$")
            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_p $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-\Gamma_1 \bar{P} \bar{d}$")
            plt.plot(grd1, rhs2, color='r', label=r"$+(1-\Gamma_1) W_P$")
            plt.plot(grd1, rhs3, color='b', label=r"$+(\Gamma_3 -1)(\overline{\rho}\widetilde{\epsilon}_{nuc})$")
            plt.plot(grd1, rhs4, color='m', label=r"$+(\Gamma_3 -1)\varepsilon_k$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_P$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'pp_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'pp_eq.eps')

    def plot_dAdt(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot dAdt in the model. A =  p / rho**gamma"""

        # load x GRID
        grd1 = self.xzn0

        plt1 = self.dAdt

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title('dAdt')

        plt.plot(grd1, plt1, color='g', label=r'$+dA/dt$')

        setxlabel = r'r (cm)'
        setylabel = r'$...$'

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'dAdt.png')
