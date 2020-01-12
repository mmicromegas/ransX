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

class ZbarTransportEquation(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(ZbarTransportEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        zbar = self.getRAdata(eht, 'zbar')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddzbar = self.getRAdata(eht, 'ddzbar')[intc]
        ddzbarux = self.getRAdata(eht, 'ddzbarux')[intc]
        ddabazbar_sum_xdn_o_an = self.getRAdata(eht, 'ddabazbar_sum_xdn_o_an')[intc]
        ddabar_sum_znxdn_o_an = self.getRAdata(eht, 'ddabar_sum_znxdn_o_an')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddzbar = self.getRAdata(eht, 'ddzbar')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_zbar = ddzbar / dd
        fzbar = ddzbarux - dd * fht_zbar * fht_ux

        #########################
        # ZBAR TRANSPORT EQUATION 
        #########################

        # LHS -dt dd zbar 		
        self.minus_dt_eht_dd_zbar = -self.dt(t_ddzbar, xzn0, t_timec, intc)

        # LHS -div dd fht_ux zbar
        self.minus_div_eht_dd_fht_ux_zbar = -self.Div(ddux * zbar, xzn0)

        # RHS -div fzbar
        self.minus_div_fzbar = -self.Div(fzbar, xzn0)

        # RHS -ddabazbar_sum_xdn_o_an
        self.minus_ddabazbar_sum_xdn_o_an = -ddabazbar_sum_xdn_o_an

        # RHS +ddabar_sum_znxdn_o_an				
        self.plus_ddabar_sum_znxdn_o_an = +ddabar_sum_znxdn_o_an

        # override NaNs (happens for ccp setup in PROMPI)
        self.plus_ddabar_sum_znxdn_o_an = np.nan_to_num(self.plus_ddabar_sum_znxdn_o_an)
        self.minus_ddabazbar_sum_xdn_o_an = np.nan_to_num(self.minus_ddabazbar_sum_xdn_o_an)

        # -res
        self.minus_resZbarEquation = -(self.minus_dt_eht_dd_zbar + \
                                       self.minus_div_eht_dd_fht_ux_zbar + self.minus_div_fzbar + \
                                       self.minus_ddabazbar_sum_xdn_o_an + self.plus_ddabar_sum_znxdn_o_an)

        #############################
        # END ZBAR TRANSPORT EQUATION
        #############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.zbar = zbar

    def plot_zbar(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot zbar stratification in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.zbar

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('zbar')
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{Z}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{Z}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_zbar.png')

    def plot_zbar_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot zbar equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_zbar
        lhs1 = self.minus_div_eht_dd_fht_ux_zbar

        rhs0 = self.minus_div_fzbar
        rhs1 = self.minus_ddabazbar_sum_xdn_o_an
        rhs2 = self.plus_ddabar_sum_znxdn_o_an

        res = self.minus_resZbarEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('zbar equation')
        plt.plot(grd1, lhs0, color='g', label=r'$-\partial_t (\overline{\rho} \widetilde{Z})$')
        plt.plot(grd1, lhs1, color='r', label=r'$-\nabla_r (\rho \widetilde{u}_r \widetilde{Z})$')
        plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_r f_Z$')
        plt.plot(grd1, rhs1, color='m', label=r'$-\overline{\rho Z A \sum_\alpha (\dot{X}_\alpha^{nuc}/A_\alpha)}$')
        plt.plot(grd1, rhs2, color='c',
                 label=r'$-\overline{\rho A \sum_\alpha (Z_\alpha \dot{X}_\alpha^{nuc}/A_\alpha)}$')

        plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'zbar_eq.png')
