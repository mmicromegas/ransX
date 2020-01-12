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

class InternalEnergyEquation(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, tke_diss, data_prefix):
        super(InternalEnergyEquation, self).__init__(ig)

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

        ddux = self.getRAdata(eht,'ddux')[intc]
        ddei = self.getRAdata(eht,'ddei')[intc]
        ddeiux = self.getRAdata(eht,'ddeiux')[intc]

        divu = self.getRAdata(eht,'divu')[intc]
        ppdivu = self.getRAdata(eht,'ppdivu')[intc]

        ddenuc1 = self.getRAdata(eht,'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht,'ddenuc2')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht,'timec')
        t_dd = self.getRAdata(eht,'dd')
        t_ddei = self.getRAdata(eht,'ddei')
        t_fht_ei = t_ddei / t_dd

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ei = ddei / dd
        fei = ddeiux - ddux * ddei / dd

        ##########################
        # INTERNAL ENERGY EQUATION 
        ##########################

        # LHS -dq/dt 		
        self.minus_dt_dd_fht_ei = -self.dt(t_dd * t_fht_ei, xzn0, t_timec, intc)

        # LHS -div dd fht_ux fht_ei		
        self.minus_div_dd_fht_ux_fht_ei = -self.Div(dd * fht_ux * fht_ei, xzn0)

        # RHS -div fei
        self.minus_div_fei = -self.Div(fei, xzn0)

        # RHS -div ftt (not included) heat flux
        self.minus_div_ftt = -np.zeros(nx)

        # RHS -P d = - pp Div ux
        self.minus_pp_div_ux = -pp * self.Div(ux, xzn0)

        # RHS -Wp = -eht_ppf_df
        self.minus_eht_ppf_df = -(ppdivu - pp * divu)

        # RHS source + dd enuc
        self.plus_dd_fht_enuc = ddenuc1 + ddenuc2

        # RHS dissipated turbulent kinetic energy
        self.plus_disstke = +tke_diss

        # -res
        self.minus_resEiEquation = -(self.minus_dt_dd_fht_ei + self.minus_div_dd_fht_ux_fht_ei + \
                                     self.minus_div_fei + self.minus_div_ftt + self.minus_pp_div_ux + self.minus_eht_ppf_df + \
                                     self.plus_dd_fht_enuc + self.plus_disstke)

        ##############################
        # END INTERNAL ENERGY EQUATION 
        ##############################			

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fht_ei = fht_ei

    def plot_ei(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mean Favrian internal energy stratification in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fht_ei

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'internal energy')
        plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{\varepsilon}_I$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{\varepsilon}_I$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_ei.png')

    def plot_ei_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot internal energy equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_fht_ei
        lhs1 = self.minus_div_dd_fht_ux_fht_ei

        rhs0 = self.minus_div_fei
        rhs1 = self.minus_div_ftt
        rhs2 = self.minus_pp_div_ux
        rhs3 = self.minus_eht_ppf_df
        rhs4 = self.plus_dd_fht_enuc
        rhs5 = self.plus_disstke

        res = self.minus_resEiEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('internal energy equation')
        plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \widetilde{\epsilon}_I )$")
        plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{\epsilon}_I$)")

        plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_I $")
        plt.plot(grd1, rhs1, color='c', label=r"$-\nabla_r f_T$ (not incl.)")
        plt.plot(grd1, rhs2, color='#802A2A', label=r"$-\bar{P} \bar{d}$")
        plt.plot(grd1, rhs3, color='r', label=r"$-W_P$")
        plt.plot(grd1, rhs4, color='b', label=r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}$")
        plt.plot(grd1, rhs5, color='m', label=r"$+\varepsilon_k$")

        plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_\epsilon$")

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'ei_eq.png')
