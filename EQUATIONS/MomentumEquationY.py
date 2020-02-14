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

class MomentumEquationY(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, data_prefix):
        super(MomentumEquationY, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]

        dduxuy = self.getRAdata(eht, 'dduxuy')[intc]
        dduzuzcoty = self.getRAdata(eht, 'dduzuzcoty')[intc]

        gradypp = self.getRAdata(eht, 'gradypp')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dduy = self.getRAdata(eht, 'dduy')

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_uy = dduy / dd
        ryx = dduxuy - ddux * dduy / dd

        #####################
        # Y MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_dduy = -self.dt(t_dduy, xzn0, t_timec, intc)

        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_uy = -self.Div(dd * fht_ux * fht_uy, xzn0)

        # RHS -div ryx
        self.minus_div_ryx = -self.Div(ryx, xzn0)

        # RHS -G
        self.minus_G = -(dduxuy - dduzuzcoty) / xzn0

        # RHS -1/r grady_pp		
        self.minus_1_o_grady_pp = -(1. / xzn0) * gradypp

        # -res
        self.minus_resResYmomentumEquation = \
            -(self.minus_dt_dduy + self.minus_div_eht_dd_fht_ux_fht_uy +
              self.minus_div_ryx + self.minus_G + self.minus_1_o_grady_pp)

        #########################
        # END Y MOMENTUM EQUATION 
        #########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dduy = dduy
        self.ig = ig
        self.fext = fext

    def plot_momentum_y(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot dduy stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationY.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.dduy

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('dduy')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{u}_y$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{u}_\theta$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\overline{\rho} \widetilde{u}_y$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\overline{\rho} \widetilde{u}_\theta$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_dduy.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_dduy.eps')

    def plot_momentum_equation_y(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot momentum y equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationY.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dduy
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_uy

        rhs0 = self.minus_div_ryx
        rhs1 = self.minus_G
        rhs2 = self.minus_1_o_grady_pp

        res = self.minus_resResYmomentumEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('y momentum equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t ( \overline{\rho} \widetilde{u}_y ) $")
            plt.plot(grd1, lhs1, color='m', label=r"$-\nabla_x (\overline{\rho} \widetilde{u}_y \widetilde{u}_y ) $")
            plt.plot(grd1, rhs0, color='b', label=r"$-\nabla_x (\widetilde{R}_{yx})$")
            # plt.plot(grd1,rhs1,color='g',label=r"$-\overline{G^{M}_\theta}$")
            # plt.plot(grd1,rhs2,color='r',label=r"$-(1/r) \overline{\partial_\theta P}$")
            plt.plot(grd1, lhs0 + lhs1 + rhs0, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t ( \overline{\rho} \widetilde{u}_\theta ) $")
            plt.plot(grd1, lhs1, color='m',
                     label=r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{u}_\theta ) $")
            plt.plot(grd1, rhs0, color='b', label=r"$-\nabla_r (\widetilde{R}_{\theta r})$")
            plt.plot(grd1, rhs1, color='g', label=r"$-\overline{G^{M}_\theta}$")
            plt.plot(grd1, rhs2, color='r', label=r"$-(1/r) \overline{\partial_\theta P}$")
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        setylabel = r"g cm$^{-2}$  s$^{-2}$"
        if self.ig == 1:
            setxlabel = r'x (cm)'
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'momentum_y_eq.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'momentum_y_eq.eps')
