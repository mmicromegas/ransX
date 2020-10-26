import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class MomentumEquationX(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, nsdim, data_prefix):
        super(MomentumEquationX, self).__init__(ig)

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
        gg = self.getRAdata(eht, 'gg')[intc]

        ddgg = self.getRAdata(eht, 'ddgg')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_ddux = self.getRAdata(eht, 'ddux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        rxx = dduxux - ddux * ddux / dd

        #####################
        # X MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_ddux = -self.dt(t_ddux, xzn0, t_timec, intc)

        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_ux = -self.Div(dd * fht_ux * fht_ux, xzn0)

        # RHS -div rxx
        self.minus_div_rxx = -self.Div(rxx, xzn0)

        # RHS -G
        if self.ig == 1:
            self.minus_G = np.zeros(nx)
        elif self.ig == 2:
            self.minus_G = -(-dduyuy - dduzuz) / xzn0

        # RHS -(grad P - rho g)
        #self.minus_gradx_pp_eht_dd_eht_gg = -self.Grad(pp,xzn0) +dd*gg
        self.minus_gradx_pp_eht_dd_eht_gg = -self.Grad(pp, xzn0) + ddgg

        # for i in range(nx):
        #    print(2.*ddgg[i],dd[i]*gg[i]		

        # -res
        self.minus_resResXmomentumEquation = \
            -(self.minus_dt_ddux + self.minus_div_eht_dd_fht_ux_fht_ux + self.minus_div_rxx
              + self.minus_G + self.minus_gradx_pp_eht_dd_eht_gg)

        #########################
        # END X MOMENTUM EQUATION 
        #########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ddux = ddux
        self.ux = ux
        self.ig = ig
        self.fext = fext
        self.nsdim = nsdim

    def plot_momentum_x(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot ddux stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationX.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ddux
        # plt2 = self.ux
        # plt3 = self.vexp

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA and set labels
        plt.title('ddux')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{u}_x$')
            # plt.plot(grd1,plt2,color='green',label = r'$\overline{u}_x$')
            # plt.plot(grd1,plt3,color='red',label = r'$v_{exp}$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{u}_r$')
            # plt.plot(grd1,plt2,color='green',label = r'$\overline{u}_x$')
            # plt.plot(grd1,plt3,color='red',label = r'$v_{exp}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\overline{\rho} \widetilde{u}_x$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\overline{\rho} \widetilde{u}_r$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ddux.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ddux.eps')

    def plot_momentum_equation_x(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot momentum x equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationX.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_ddux
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_ux

        rhs0 = self.minus_div_rxx
        rhs1 = self.minus_G
        rhs2 = self.minus_gradx_pp_eht_dd_eht_gg

        res = self.minus_resResXmomentumEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('x momentum equation ' + str(self.nsdim) + "D")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t ( \overline{\rho} \widetilde{u}_r ) $")
            plt.plot(grd1, lhs1, color='m', label=r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{u}_x ) $")
            plt.plot(grd1, rhs0, color='b', label=r"$-\nabla_x (\widetilde{R}_{xx})$")
            plt.plot(grd1, rhs2, color='r', label=r"$-(\partial_x \overline{P} - \bar{\rho}\tilde{g}_x)$")
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t ( \overline{\rho} \widetilde{u}_r ) $")
            plt.plot(grd1, lhs1, color='m', label=r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{u}_r ) $")
            plt.plot(grd1, rhs0, color='b', label=r"$-\nabla_r (\widetilde{R}_{rr})$")
            plt.plot(grd1, rhs1, color='g', label=r"$-\overline{G^{M}_r}$")
            plt.plot(grd1, rhs2, color='r', label=r"$-(\partial_r \overline{P} - \bar{\rho}\tilde{g}_r)$")
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
            plt.savefig('RESULTS/' + self.data_prefix + 'momentum_x_eq.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'momentum_x_eq.eps')
