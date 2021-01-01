import numpy as np
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

class EnthalpyEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, nsdim, tke_diss, data_prefix):
        super(EnthalpyEquation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        hh = self.getRAdata(eht, 'hh')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddhh = self.getRAdata(eht, 'ddhh')[intc]
        ddhhux = self.getRAdata(eht, 'ddhhux')[intc]

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
        t_dd = self.getRAdata(eht, 'dd')
        t_ddhh = self.getRAdata(eht, 'ddhh')
        t_fht_hh = t_ddhh / t_dd

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_hh = ddhh / dd
        fhh = ddhhux - ddux * ddhh / dd

        ###################
        # ENTHALPY EQUATION 
        ###################

        # LHS -dq/dt 		
        self.minus_dt_dd_fht_hh = -self.dt(t_dd * t_fht_hh, xzn0, t_timec, intc)

        # LHS -div dd fht_ux fht_hh		
        self.minus_div_dd_fht_ux_fht_hh = -self.Div(dd * fht_ux * fht_hh, xzn0)

        # RHS -div fhh
        self.minus_div_fhh = -self.Div(fhh, xzn0)

        # RHS -gamma1 P d = - gamma1 pp Div ux
        self.minus_gamma1_pp_div_ux = -gamma1 * pp * self.Div(ux, xzn0)

        # RHS -gamma1 Wp = -gamma1 eht_ppf_df
        self.minus_gamma1_eht_ppf_df = -gamma1 * (ppdivu - pp * divu)

        # RHS source + gamma3 dd enuc
        self.plus_gamma3_dd_fht_enuc = gamma3 * (ddenuc1 + ddenuc2)

        # RHS gamma3 dissipated turbulent kinetic energy
        self.plus_gamma3_disstke = +gamma3 * tke_diss

        # RHS gamma3 div ft
        self.plus_gamma3_div_ft = +np.zeros(nx)

        # -res
        self.minus_resHHequation = -(self.minus_dt_dd_fht_hh + self.minus_div_dd_fht_ux_fht_hh + self.minus_div_fhh +
                                     self.minus_gamma1_pp_div_ux + self.minus_gamma1_eht_ppf_df +
                                     self.plus_gamma3_dd_fht_enuc + self.plus_gamma3_disstke + self.plus_gamma3_div_ft)

        #######################
        # END ENTHALPY EQUATION 
        #######################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fht_hh = fht_hh
        self.fext = fext
        self.nsdim = nsdim

    def plot_hh(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot mean Favrian enthalpy stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EnthalpyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fht_hh

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'enthalpy')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{h}$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{h}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\widetilde{h}$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\widetilde{h}$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_hh.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_hh.eps')

    def plot_hh_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot enthalpy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(EnthalpyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_fht_hh
        lhs1 = self.minus_div_dd_fht_ux_fht_hh

        rhs0 = self.minus_div_fhh
        rhs1 = self.minus_gamma1_pp_div_ux
        rhs2 = self.minus_gamma1_eht_ppf_df
        rhs3 = self.plus_gamma3_dd_fht_enuc
        rhs4 = self.plus_gamma3_disstke
        rhs5 = self.plus_gamma3_div_ft

        res = self.minus_resHHequation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('enthalpy equation ' + str(self.nsdim) + " D")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \widetilde{h})$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\overline{\rho}\widetilde{u}_x \widetilde{h}$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_h $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-\Gamma_1 \bar{P} \bar{d}$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\Gamma_1 W_P$")
            plt.plot(grd1, rhs3, color='b', label=r"$+\Gamma_3 \overline{\rho}\widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1, rhs4, color='m', label=r"$+\Gamma_3 \varepsilon_k$")
            plt.plot(grd1, rhs5, color='c', label=r"$+\Gamma_3 \nabla_x f_T$ (not incl.)")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_h$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \widetilde{h})$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{h}$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_h $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-\Gamma_1 \bar{P} \bar{d}$")
            plt.plot(grd1, rhs2, color='r', label=r"$-\Gamma_1 W_P$")
            plt.plot(grd1, rhs3, color='b', label=r"$+\Gamma_3 \overline{\rho}\widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1, rhs4, color='m', label=r"$+\Gamma_3 \varepsilon_k$")
            plt.plot(grd1, rhs5, color='c', label=r"$+\Gamma_3 \nabla_r f_T$ (not incl.)")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_h$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hh_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hh_eq.eps')