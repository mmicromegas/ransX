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

class TemperatureEquation(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, tke_diss, data_prefix):
        super(TemperatureEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht,'xzn0')
        nx = self.getRAdata(eht,'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht,'dd')[intc]
        ux = self.getRAdata(eht,'ux')[intc]
        tt = self.getRAdata(eht,'tt')[intc]
        cv = self.getRAdata(eht,'cv')[intc]

        ddux = self.getRAdata(eht,'ddux')[intc]
        ttux = self.getRAdata(eht,'ttux')[intc]

        divu = self.getRAdata(eht,'divu')[intc]
        ttdivu = self.getRAdata(eht,'ttdivu')[intc]

        enuc1_o_cv = self.getRAdata(eht,'enuc1_o_cv')[intc]
        enuc2_o_cv = self.getRAdata(eht,'enuc2_o_cv')[intc]

        gamma1 = self.getRAdata(eht,'gamma1')[intc]
        gamma3 = self.getRAdata(eht,'gamma3')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if (ieos == 1):
            cp = self.getRAdata(eht,'cp')[intc]
            cv = self.getRAdata(eht,'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1

        # store time series for time derivatives
        t_timec = self.getRAdata(eht,'timec')
        t_tt = self.getRAdata(eht,'tt')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        ftt = ttux - tt * ux

        ######################
        # TEMPERATURE EQUATION 
        ######################

        # LHS -dq/dt 		
        self.minus_dt_tt = -self.dt(t_tt, xzn0, t_timec, intc)

        # LHS -ux grad T		
        self.minus_ux_grad_tt = -ux * self.Grad(tt, xzn0)

        # RHS -div ftt
        self.minus_div_ftt = -self.Div(ftt, xzn0)

        # RHS +(1-gamma3) T d = +(1-gamma3) tt Div eht_ux
        self.plus_one_minus_gamma3_tt_div_ux = +(1. - gamma3) * tt * self.Div(ux, xzn0)

        # RHS +(2-gamma3) Wt = +(2-gamma3) eht_ttf_df
        self.plus_two_minus_gamma3_eht_ttf_df = +(2. - gamma3) * (ttdivu - tt * divu)

        # RHS source +enuc/cv
        self.plus_enuc_o_cv = enuc1_o_cv + enuc2_o_cv

        # RHS +dissipated turbulent kinetic energy _o_ cv (this is a guess)
        self.plus_disstke_o_cv = +tke_diss / (dd * cv)

        # RHS +div ftt/dd cv (not included)	
        self.plus_div_ftt_o_dd_cv = np.zeros(nx)

        # RHS +viscous tensor grad u / dd cv
        # self.plus_tau_grad_u_o_dd_cv = np.zeros(nx)

        # -res
        self.minus_resTTequation = -(self.minus_dt_tt + self.minus_ux_grad_tt + \
                                     self.minus_div_ftt + self.plus_one_minus_gamma3_tt_div_ux + self.plus_two_minus_gamma3_eht_ttf_df + \
                                     self.plus_enuc_o_cv + self.plus_disstke_o_cv + self.plus_div_ftt_o_dd_cv)

        ##########################
        # END TEMPERATURE EQUATION 
        ##########################			

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.tt = tt
        self.ig = ig

    def plot_tt(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mean temperature stratification in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.tt

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'temperature')

        if (self.ig == 1):
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{T}$')
            # define x LABEL
            setxlabel = r"x (cm)"
        elif (self.ig == 2):
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{T}$')
            # define x LABEL
            setxlabel = r"r (cm)"
        else:
            print(
                "ERROR (TemperatureEquation.py): geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

            # define y LABEL
        setylabel = r"$\overline{T} (K)$"

        # show x/y LABELS
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_tt.png')

    def plot_tt_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot temperature equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_tt
        lhs1 = self.minus_ux_grad_tt

        rhs0 = self.minus_div_ftt
        rhs1 = self.plus_one_minus_gamma3_tt_div_ux
        rhs2 = self.plus_two_minus_gamma3_eht_ttf_df
        rhs3 = self.plus_enuc_o_cv
        rhs4 = self.plus_disstke_o_cv
        rhs5 = self.plus_div_ftt_o_dd_cv

        res = self.minus_resTTequation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('temperature equation')
        if (self.ig == 1):
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{T})$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\overline{u}_x \partial_x \overline{T}$")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_T $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$+(1-\Gamma_3) \bar{T} \bar{d}$")
            plt.plot(grd1, rhs2, color='r', label=r"$+(2-\Gamma_3) \overline{T'd'}$")
            plt.plot(grd1, rhs3, color='b', label=r"$+\overline{\epsilon_{nuc} / cv}$")
            plt.plot(grd1, rhs4, color='g', label=r"$+\overline{\varepsilon_{diss}} \ / \ \overline{cv}$")
            plt.plot(grd1, rhs5, color='m', label=r"+$\nabla \cdot F_T/ \rho c_v$ (not incl.)")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_T$")
            # define X label
            setxlabel = r"x (cm)"
        elif (self.ig == 2):
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{T})$")
            plt.plot(grd1, lhs1, color='k', label=r"$-\overline{u}_r \partial_r \overline{T}$")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_T $")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$+(1-\Gamma_3) \bar{T} \bar{d}$")
            plt.plot(grd1, rhs2, color='r', label=r"$+(2-\Gamma_3) \overline{T'd'}$")
            plt.plot(grd1, rhs3, color='b', label=r"$+\overline{\epsilon_{nuc} / cv}$")
            plt.plot(grd1, rhs4, color='g', label=r"$+\overline{\varepsilon / cv}$")
            plt.plot(grd1, rhs5, color='m', label=r"+$\nabla \cdot F_T/ \rho c_v$ (not incl.)")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_T$")
            # define X label
            setxlabel = r"r (cm)"
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

            # define y LABEL
        setylabel = r"K s$^{-1}$"

        # show x/y LABELS		
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tt_eq.png')
