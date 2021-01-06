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

class HsseMomentumEquationX(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, ieos, fext, intc, data_prefix, bconv, tconv):
        super(HsseMomentumEquationX, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        gg = self.getRAdata(eht, 'gg')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        gamma1 = self.getRAdata(eht, 'gamma1')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if (ieos == 1):
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddux = self.getRAdata(eht, 'ddux')
        t_fht_ux = t_ddux / t_dd

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        rxx = dduxux - ddux * ddux / dd

        gg = -gg

        fht_rxx = dduxux - ddux * ddux / dd
        fdil = (uxdivu - ux * divu)

        ##########################
        # HSSE X MOMENTUM EQUATION 
        ##########################

        # LHS -gradx p
        self.minus_gradx_pp = -self.Grad(pp, xzn0)

        # RHS - dd gg		
        self.minus_dd_gg = -dd * gg

        # RHS -dd dt fht_ux 		
        self.minus_dd_dt_fht_ux = -dd * self.dt(t_fht_ux, xzn0, t_timec, intc)

        # RHS -div rxx
        self.minus_div_rxx = -self.Div(rxx, xzn0)

        # RHS -G
        self.minus_G = -(-dduyuy - dduzuz) / xzn0

        # RHS -dd fht_ux gradx fht_ux
        self.minus_dd_fht_ux_gradx_fht_ux = -dd * fht_ux * self.Grad(fht_ux, xzn0)

        # -res (geometry dependent)
        if ig == 1:
            self.minus_resResXmomentumEquation = \
                -(self.minus_gradx_pp + self.minus_dd_gg + self.minus_dd_dt_fht_ux + self.minus_div_rxx + +self.minus_dd_fht_ux_gradx_fht_ux)
        elif ig == 2:
            self.minus_resResXmomentumEquation = \
                -(self.minus_gradx_pp + self.minus_dd_gg + self.minus_dd_dt_fht_ux + self.minus_div_rxx +
                  self.minus_G + self.minus_dd_fht_ux_gradx_fht_ux)
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        ##############################
        # END HSSE X MOMENTUM EQUATION 
        ##############################

        ###############################
        # ALTERNATIVE MOMENTUM EQUATION 
        ###############################

        self.minus_gamma1_pp_dd_fdil_o_fht_rxx = -gamma1 * pp * dd * fdil / fht_rxx

        self.minus_resResXmomentumEquation2 = -(self.minus_gradx_pp + self.minus_gamma1_pp_dd_fdil_o_fht_rxx)

        ###################################
        # END ALTERNATIVE MOMENTUM EQUATION 
        ###################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ddux = ddux
        self.ux = ux
        self.bconv = bconv
        self.tconv = tconv
        self.ig = ig
        self.fext = fext

    def plot_momentum_x(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot ddux stratification in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ddux
        plt2 = self.ux
        # plt3 = self.vexp

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('ddux')
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{u}_x$')
        # plt.plot(grd1,plt2,color='green',label = r'$\overline{u}_x$')
        # plt.plot(grd1,plt3,color='red',label = r'$v_{exp}$')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
        elif self.ig == 2:
            setxlabel = r"r (cm)"
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        setylabel = r"$\overline{\rho} \widetilde{u}_x$ (g cm$^{-2}$ s$^{-1}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ddux.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_ddux.eps')

    def plot_momentum_equation_x(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot momentum x equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_pp

        rhs0 = self.minus_dd_gg
        rhs1 = self.minus_dd_dt_fht_ux
        rhs2 = self.minus_div_rxx
        rhs3 = self.minus_G
        rhs4 = self.minus_dd_fht_ux_gradx_fht_ux

        res = self.minus_resResXmomentumEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs1, rhs2, rhs3, rhs4, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('hsse x momentum equation')
        if (self.ig == 1):
            # plt.plot(grd1,lhs0,color='c',label = r"$-\partial_r \overline{P} $")
            # plt.plot(grd1,rhs0,color='m',label = r"$-\overline{\rho} \ \overline{g}_r$")
            # plt.plot(grd1,rhs1,color='r',label = r"$-\overline{\rho} \partial_t \widetilde{u}_r$")
            # plt.plot(grd1,rhs2,color='b',label=r"$-\nabla_r (\widetilde{R}_{rr})$")
            # plt.plot(grd1,rhs3,color='g',label=r"$-\overline{G^{M}_r}$")
            # plt.plot(grd1,rhs4,color='y',label=r"$-\overline{\rho} \widetilde{u}_r \partial_r \widetilde{u}_r$")
            # plt.plot(grd1,res,color='k',linestyle='--',label='res')

            xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
            xlimitbottom = np.where(grd1 < self.bconv)
            xlimittop = np.where(grd1 > self.tconv)

            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='c', label=r"$-\partial_x \overline{P} $")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='m', label=r"$-\overline{\rho} \ \overline{g}_x$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='r',
                     label=r"$-\overline{\rho} \partial_t \widetilde{u}_r$")
            plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='b', label=r"$-\nabla_r (\widetilde{R}_{xx})$")
            plt.plot(grd1[xlimitrange], rhs4[xlimitrange], color='y',
                     label=r"$-\overline{\rho} \widetilde{u}_x \partial_x \widetilde{u}_x$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label='res')

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='c', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs2[xlimitbottom], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs4[xlimitbottom], '.', color='y', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='c', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs2[xlimittop], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs4[xlimittop], '.', color='y', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

            # define x LABELS
            setxlabel = r"x (cm)"
        elif (self.ig == 2):
            # plt.plot(grd1,lhs0,color='c',label = r"$-\partial_r \overline{P} $")
            # plt.plot(grd1,rhs0,color='m',label = r"$-\overline{\rho} \ \overline{g}_r$")
            # plt.plot(grd1,rhs1,color='r',label = r"$-\overline{\rho} \partial_t \widetilde{u}_r$")
            # plt.plot(grd1,rhs2,color='b',label=r"$-\nabla_r (\widetilde{R}_{rr})$")
            # plt.plot(grd1,rhs3,color='g',label=r"$-\overline{G^{M}_r}$")
            # plt.plot(grd1,rhs4,color='y',label=r"$-\overline{\rho} \widetilde{u}_r \partial_r \widetilde{u}_r$")
            # plt.plot(grd1,res,color='k',linestyle='--',label='res')

            xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
            xlimitbottom = np.where(grd1 < self.bconv)
            xlimittop = np.where(grd1 > self.tconv)

            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='c', label=r"$-\partial_r \overline{P} $")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='m', label=r"$-\overline{\rho} \ \overline{g}_r$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='r',
                     label=r"$-\overline{\rho} \partial_t \widetilde{u}_r$")
            plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='b', label=r"$-\nabla_r (\widetilde{R}_{rr})$")
            plt.plot(grd1[xlimitrange], rhs3[xlimitrange], color='g', label=r"$-\overline{G^{M}_r}$")
            plt.plot(grd1[xlimitrange], rhs4[xlimitrange], color='y',
                     label=r"$-\overline{\rho} \widetilde{u}_r \partial_r \widetilde{u}_r$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label='res')

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='c', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs2[xlimitbottom], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs3[xlimitbottom], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs4[xlimitbottom], '.', color='y', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='c', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs2[xlimittop], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs3[xlimittop], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs4[xlimittop], '.', color='y', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

            # define x LABELS
            setxlabel = r"r (cm)"
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')

        # define y LABEL		
        setylabel = r"erg cm$^{-3}$  cm$^{-1}$"

        # show x/y LABELS		
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_momentum_x_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_momentum_x_eq.eps')

    def plot_momentum_equation_x_2(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot momentum x equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_pp

        rhs0 = self.minus_gamma1_pp_dd_fdil_o_fht_rxx

        res = self.minus_resResXmomentumEquation2

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('alternative hsse x momentum equation')
        # plt.plot(grd1,lhs0,color='c',label = r"$-\partial_r \overline{P} $")
        # plt.plot(grd1,rhs0,color='m',label = r"$-\Gamma_1 \ \overline{\rho} \ \overline{P} \ \overline{u'_r d''} / \ \widetilde{R}_{rr}$")
        # plt.plot(grd1,res,color='k',linestyle='--',label='res')

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='c', label=r"$-\partial_r \overline{P} $")
        plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='m',
                 label=r"$-\Gamma_1 \ \overline{\rho} \ \overline{P} \ \overline{u'_r d''} / \ \widetilde{R}_{rr}$")
        plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label='res')

        plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='c', markersize=0.5)
        plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='m', markersize=0.5)
        plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

        plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='c', markersize=0.5)
        plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='m', markersize=0.5)
        plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$  cm$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_momentum_x_eq_alternative.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_momentum_x_eq_alternative.eps')

    def plot_momentum_equation_x_3(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot momentum x equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_pp

        rhs0 = self.minus_dd_gg

        res = -(self.minus_gradx_pp + self.minus_dd_gg)

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('alternative hsse x momentum eq simp')
        # plt.plot(grd1,lhs0,color='c',label = r"$-\partial_r \overline{P} $")
        # plt.plot(grd1,rhs0,color='m',label = r"$-\overline{\rho} \ \overline{g}_r$")
        # plt.plot(grd1,res,color='k',linestyle='--',label='res')

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='c', label=r"$-\partial_r \overline{P} $")
        plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='m', label=r"$-\overline{\rho} \ \overline{g}_r$")
        plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label='res')

        plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='c', markersize=0.5)
        plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='m', markersize=0.5)
        plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

        plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='c', markersize=0.5)
        plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='m', markersize=0.5)
        plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$  cm$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_momentum_x_eq_alternative_simplified.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_momentum_x_eq_alternative_simplified.eps')
