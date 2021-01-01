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

class AbarTransportEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, intc, nsdim, data_prefix):
        super(AbarTransportEquation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        abar = self.getRAdata(eht, 'abar')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]

        ddabar = self.getRAdata(eht, 'ddabar')[intc]
        ddabarux = self.getRAdata(eht, 'ddabarux')[intc]

        self.ddabarsq_sum_xdn_o_an = self.getRAdata(eht, 'ddabarsq_sum_xdn_o_an')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_ddabar = self.getRAdata(eht, 'ddabar')

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_abar = ddabar / dd
        fabar = ddabarux - dd * fht_abar * fht_ux

        #########################
        # ABAR TRANSPORT EQUATION 
        #########################

        # LHS -dt dd abar 		
        self.minus_dt_eht_dd_abar = -self.dt(t_ddabar, xzn0, t_timec, intc)

        # LHS -div dd fht_ux abar
        self.minus_div_eht_dd_fht_ux_abar = -self.Div(ddux * abar, xzn0)

        # RHS -div fabar
        self.minus_div_fabar = -self.Div(fabar, xzn0)

        # RHS -ddabarsq_sum_xdn_o_an
        self.minus_ddabarsq_sum_xdn_o_an = -self.ddabarsq_sum_xdn_o_an

        # override NaNs (happens for ccp setup in PROMPI)
        self.minus_ddabarsq_sum_xdn_o_an = np.nan_to_num(self.minus_ddabarsq_sum_xdn_o_an)

        # -res
        self.minus_resAbarEquation = -(self.minus_dt_eht_dd_abar + self.minus_div_eht_dd_fht_ux_abar +
                                       self.minus_div_fabar + self.minus_ddabarsq_sum_xdn_o_an)

        #############################
        # END ABAR TRANSPORT EQUATION
        #############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.abar = abar
        self.nsdim = nsdim

    def plot_abar(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot abar stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(AbarTransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.abar

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # plot DATA 
        plt.title('abar')
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{A}$')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\overline{A}$"
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\overline{A}$"

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_abar.png')

    def plot_abar_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot abar equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(AbarTransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_abar
        lhs1 = self.minus_div_eht_dd_fht_ux_abar

        rhs0 = self.minus_div_fabar
        rhs1 = self.minus_ddabarsq_sum_xdn_o_an

        res = self.minus_resAbarEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('abar equation ' + str(self.nsdim) + "D")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='g', label=r'$-\partial_t (\overline{\rho} \widetilde{A})$')
            plt.plot(grd1, lhs1, color='r', label=r'$-\nabla_x (\rho \widetilde{u}_x \widetilde{A})$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f_A$')
            plt.plot(grd1, rhs1, color='m', label=r'$-\overline{\rho A^2 \sum_\alpha (\dot{X}_\alpha^{nuc}/A_\alpha)}$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='g', label=r'$-\partial_t (\overline{\rho} \widetilde{A})$')
            plt.plot(grd1, lhs1, color='r', label=r'$-\nabla_r (\rho \widetilde{u}_r \widetilde{A})$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_r f_A$')
            plt.plot(grd1, rhs1, color='m', label=r'$-\overline{\rho A^2 \sum_\alpha (\dot{X}_\alpha^{nuc}/A_\alpha)}$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'abar_eq.png')
