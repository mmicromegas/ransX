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

class MomentumEquationZ(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, data_prefix):
        super(MomentumEquationZ, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        dduxuz = self.getRAdata(eht, 'dduxuz')[intc]
        dduzuycoty = self.getRAdata(eht, 'dduzuycoty')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dduz = self.getRAdata(eht, 'dduz')

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_uz = dduz / dd
        rzx = dduxuz - ddux * dduz / dd

        #####################
        # Z MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_dduz = -self.dt(t_dduz, xzn0, t_timec, intc)

        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_uz = -self.Div(dd * fht_ux * fht_uz, xzn0)

        # RHS -div rzx
        self.minus_div_rzx = -self.Div(rzx, xzn0)

        # RHS -G
        self.minus_G = -(dduxuz + dduzuycoty) / xzn0

        # -res
        self.minus_resResZmomentumEquation = \
            -(self.minus_dt_dduz + self.minus_div_eht_dd_fht_ux_fht_uz + self.minus_div_rzx
              + self.minus_G)

        #########################
        # END Z MOMENTUM EQUATION 
        #########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dduz = dduz
        self.fext = fext

    def plot_momentum_z(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot dduz stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationZ.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.dduz

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('dduz')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{u}_z$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{u}_\phi$')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\overline{\rho} \widetilde{u}_z$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\overline{\rho} \widetilde{u}_\phi$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_dduz.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_dduz.eps')

    def plot_momentum_equation_z(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot momentum z equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationZ.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dduz
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_uz

        rhs0 = self.minus_div_rzx
        rhs1 = self.minus_G

        res = self.minus_resResZmomentumEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('z momentum equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t ( \overline{\rho} \widetilde{u}_z ) $")
            plt.plot(grd1, lhs1, color='m', label=r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{u}_z ) $")
            plt.plot(grd1, rhs0, color='b', label=r"$-\nabla_x (\widetilde{R}_{zx})$")
            # plt.plot(grd1,rhs1,color='g',label=r"$-\overline{G^{M}_\phi}$")
            plt.plot(grd1, lhs0 + lhs1 + rhs0, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t ( \overline{\rho} \widetilde{u}_\phi ) $")
            plt.plot(grd1, lhs1, color='m', label=r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{u}_\phi ) $")
            plt.plot(grd1, rhs0, color='b', label=r"$-\nabla_r (\widetilde{R}_{\phi r})$")
            plt.plot(grd1, rhs1, color='g', label=r"$-\overline{G^{M}_\phi}$")
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

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
            plt.savefig('RESULTS/' + self.data_prefix + 'momentum_z_eq.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'momentum_z_eq.eps')
