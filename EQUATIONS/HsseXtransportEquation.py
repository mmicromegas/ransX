import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseXtransportEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, inuc, element, bconv, tconv, intc, data_prefix):
        super(HsseXtransportEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = self.getRAdata(eht, 'dd')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]

        ############################
        # HSSE Xi TRANSPORT EQUATION 
        ############################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_fht_xi = t_ddxi / t_dd

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        fxi = ddxiux - ddxi * ddux / dd

        # LHS -dq/dt 		
        self.minus_dt_fht_xi = -self.dt(t_fht_xi, xzn0, t_timec, intc)

        # RHS +fht Xidot 
        self.plus_fht_xidot = +ddxidot / dd

        # RHS -(1/dd)div fxi 
        self.minus_one_o_dd_div_fxi = -(1. / dd) * self.Div(fxi, xzn0)

        # LHS -fht_ux gradx fht_xi
        self.minus_div_eht_dd_fht_ux_fht_xi = -fht_ux * self.Grad(fht_xi, xzn0)

        # -res
        self.minus_resXiTransport = -(self.minus_dt_fht_xi + self.plus_fht_xidot + self.minus_one_o_dd_div_fxi +
                                      self.minus_div_eht_dd_fht_ux_fht_xi)

        ################################
        # END HSSE Xi TRANSPORT EQUATION
        ################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.inuc = inuc
        self.element = element
        self.ddxi = ddxi

        self.bconv = bconv
        self.tconv = tconv
        self.fext = fext
        self.nx = nx

    def plot_Xrho(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xrho stratification in the model"""

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ddxi

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('rhoX for ' + element)
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{X}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + element + '.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + element + '.eps')

    def plot_Xtransport_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xrho transport equation in the model"""

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fht_xi

        rhs0 = self.plus_fht_xidot
        rhs1 = self.minus_one_o_dd_div_fxi
        rhs2 = self.minus_div_eht_dd_fht_ux_fht_xi

        res = self.minus_resXiTransport

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('hsse rhoX transport for ' + element)
        # plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t \widetilde{X}_i$')
        # plt.plot(grd1,rhs0,color='g',label=r'$+\widetilde{\dot{X}}^{\rm nuc}_i$')
        # plt.plot(grd1,rhs1,color='b',label=r'$-(1/\overline{\rho}) \nabla_r f_i$')
        # plt.plot(grd1,rhs2,color='y',label=r"$-\widetilde{u}_r \partial_r \widetilde{X}_i$")
        # plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitrange = np.where((grd1 > self.xzn0[0]) & (grd1 < self.xzn0[self.nx-1]))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='r', label=r'$-\partial_t \widetilde{X}_i$')
        plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='g', label=r'$+\widetilde{\dot{X}}^{\rm nuc}_i$')
        plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='b', label=r'$-(1/\overline{\rho}) \nabla_r f_i$')
        plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='y',
                 label=r"$-\widetilde{u}_r \partial_r \widetilde{X}_i$")
        plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label='res')

        # plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='r', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='g', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='b', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], rhs2[xlimitbottom], '.', color='y', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

        # plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='r', markersize=0.5)
        # plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='g', markersize=0.5)
        # plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='b', markersize=0.5)
        # plt.plot(grd1[xlimittop], rhs2[xlimittop], '.', color='y', markersize=0.5)
        # plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"s$^{-1}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_mean_Xtransport_' + element + '.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_mean_Xtransport_' + element + '.eps')