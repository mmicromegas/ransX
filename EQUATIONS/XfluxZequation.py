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

class XfluxZequation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, inuc, element, bconv, tconv, tke_diss, tauL, intc, data_prefix):
        super(XfluxZequation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        xi = self.getRAdata(eht, 'x' + inuc)[intc]

        uxy = self.getRAdata(eht, 'uxy')[intc]
        uxz = self.getRAdata(eht, 'uxz')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]
        ddgg = self.getRAdata(eht, 'ddgg')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]
        dduxuy = self.getRAdata(eht, 'dduxuy')[intc]
        dduxuz = self.getRAdata(eht, 'dduxuz')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uxuy = self.getRAdata(eht, 'uxuy')[intc]
        uxuz = self.getRAdata(eht, 'uxuz')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]

        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        xiux = self.getRAdata(eht, 'x' + inuc + 'ux')[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxiuy = self.getRAdata(eht, 'ddx' + inuc + 'uy')[intc]
        ddxiuz = self.getRAdata(eht, 'ddx' + inuc + 'uz')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]

        gradzpp_o_siny = self.getRAdata(eht, 'gradzpp_o_siny')[intc]
        print("ERROR(XfluxZequation.py): gradzpp missing ... ")
        gradzpp = np.zeros(nx)
        # sys.exit()
        # gradzpp = self.getRAdata(eht,'gradzpp')[intc]

        ddxiuzuzcoty = self.getRAdata(eht, 'ddx' + inuc + 'uzuzcoty')[intc]
        dduzuzcoty = self.getRAdata(eht, 'dduzuzcoty')[intc]

        ddxiuzuycoty = self.getRAdata(eht, 'ddx' + inuc + 'uzuycoty')[intc]
        dduzuycoty = self.getRAdata(eht, 'dduzuycoty')[intc]

        xigradxpp = self.getRAdata(eht, 'x' + inuc + 'gradxpp')[intc]
        xigradypp = self.getRAdata(eht, 'x' + inuc + 'gradypp')[intc]
        # xigradzpp = self.getRAdata(eht,'x' + inuc + 'gradzpp')[intc]
        xigradzpp = np.zeros(nx)
        xigradzpp_o_siny = self.getRAdata(eht, 'x' + inuc + 'gradzpp_o_siny')[intc]

        ddxidotux = self.getRAdata(eht, 'ddx' + inuc + 'dotux')[intc]
        ddxidotuy = self.getRAdata(eht, 'ddx' + inuc + 'dotuy')[intc]
        ddxidotuz = self.getRAdata(eht, 'ddx' + inuc + 'dotuz')[intc]

        ddxiuxux = self.getRAdata(eht, 'ddx' + inuc + 'uxux')[intc]
        ddxiuyuy = self.getRAdata(eht, 'ddx' + inuc + 'uyuy')[intc]
        ddxiuzuz = self.getRAdata(eht, 'ddx' + inuc + 'uzuz')[intc]
        ddxiuxuy = self.getRAdata(eht, 'ddx' + inuc + 'uxuy')[intc]
        ddxiuxuz = self.getRAdata(eht, 'ddx' + inuc + 'uxuz')[intc]

        xiddgg = self.getRAdata(eht, 'x' + inuc + 'ddgg')[intc]
        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        gamma1 = self.getRAdata(eht, 'gamma1')[intc]
        gamma3 = self.getRAdata(eht, 'gamma3')[intc]

        gamma1 = self.getRAdata(eht, 'ux')[intc]
        gamma3 = self.getRAdata(eht, 'ux')[intc]

        fht_rxx = dduxux - ddux * ddux / dd
        fdil = (uxdivu - ux * divu)

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_dduz = self.getRAdata(eht, 'dduz')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_ddxiuz = self.getRAdata(eht, 'ddx' + inuc + 'uz')

        ##################
        # Xi FLUX EQUATION 
        ##################

        # construct equation-specific mean fields
        t_fzi = t_ddxiuz - t_ddxi * t_dduz / t_dd

        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd
        fht_xi = ddxi / dd

        rxx = dduxux - ddux * ddux / dd
        ryx = dduxuy - dduy * ddux / dd
        rzx = dduxuz - dduz * ddux / dd

        fxi = ddxiux - ddxi * ddux / dd
        fyi = ddxiuy - ddxi * dduy / dd
        fzi = ddxiuz - ddxi * dduz / dd

        fxxi = ddxiuxux - (ddxi / dd) * dduxux - (ddux / dd) * ddxiux - (
                ddux / dd) * ddxiux + 2. * ddxi * ddux * ddux / (dd * dd)
        fyxi = ddxiuxuy - (ddxi / dd) * dduxuy - (dduy / dd) * ddxiux - (
                ddux / dd) * ddxiuy + 2. * ddxi * dduy * ddux / (dd * dd)
        fzxi = ddxiuxuz - (ddxi / dd) * dduxuz - (dduz / dd) * ddxiux - (
                ddux / dd) * ddxiuz + 2. * ddxi * dduz * ddux / (dd * dd)

        # LHS -dq/dt 
        self.minus_dt_fzi = -self.dt(t_fzi, xzn0, t_timec, intc)

        # LHS -div(dduxfzi)
        self.minus_div_fht_ux_fzi = -self.Div(fht_ux * fzi, xzn0)

        # RHS -div fzxi  
        self.minus_div_fzxi = -self.Div(fzxi, xzn0)

        # RHS -fxi gradx fht_uz
        self.minus_fxi_gradx_fht_uz = -fxi * self.Grad(fht_uz, xzn0)

        # RHS -rzx gradx fht_xi
        self.minus_rzx_gradx_fht_xi = -rzx * self.Grad(fht_xi, xzn0)

        if ig == 1:
            # RHS -xff_gradz_pp
            self.minus_eht_xff_gradz_pp_o_sinyrr = -(xigradzpp - fht_xi * gradzpp)
        elif ig == 2:
            # RHS -xff_gradz_pp_o_siny_rr
            self.minus_eht_xff_gradz_pp_o_sinyrr = -(xigradzpp_o_siny - fht_xi * gradzpp_o_siny) / xzn0

        # RHS +uzff_eht_dd_xidot
        self.plus_uzff_eht_dd_xidot = +(ddxidotuz - (dduz / dd) * ddxidot)

        # RHS +gi 
        self.plus_gi = \
            -((ddxiuxuz - (ddxi / dd) * dduxuz) / xzn0 + ((ddxiuzuycoty - (ddxi / dd) * dduzuycoty) / xzn0))

        # -res				   
        self.minus_resXiFlux = -(self.minus_dt_fzi + self.minus_div_fht_ux_fzi + self.minus_div_fzxi +
                                 self.minus_fxi_gradx_fht_uz + self.minus_rzx_gradx_fht_xi +
                                 self.minus_eht_xff_gradz_pp_o_sinyrr + self.plus_uzff_eht_dd_xidot + self.plus_gi)

        ######################
        # END Xi FLUX EQUATION 
        ######################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.inuc = inuc
        self.element = element
        self.fzi = fzi
        self.nx = nx
        self.bconv = bconv
        self.tconv = tconv

    def plot_XfluxZ(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xflux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XfluxZEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load and calculate DATA to plot
        plt1 = self.fzi

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xflux Z for ' + self.element)
        plt.plot(grd1, plt1, color='k', label=r'f')

        # convective boundary markers
        plt.axvline(self.bconv + 0.46e8, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers		
        # plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')
        # plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxZ_' + element + '.png')

    def plot_XfluxZ_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xi flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XfluxZEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fzi
        lhs1 = self.minus_div_fht_ux_fzi

        rhs0 = self.minus_div_fzxi
        rhs1 = self.minus_fxi_gradx_fht_uz
        rhs2 = self.minus_rzx_gradx_fht_xi
        rhs3 = self.minus_eht_xff_gradz_pp_o_sinyrr
        rhs4 = self.plus_uzff_eht_dd_xidot
        rhs5 = self.plus_gi

        if self.ig == 1:
            res = -(lhs0 + lhs1 + rhs0 + rhs1 + rhs2 + rhs3 + rhs4)
            rhs5 = np.zeros(self.nx)
        elif self.ig == 2:
            res = self.minus_resXiFlux

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xflux Z equation for ' + self.element)
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_z$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla_x (\widetilde{u}_x f_z)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f^z$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_{r} \partial_x \widetilde{u}_z$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{xz} \partial_x \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan', label=r"$-\overline{X''\partial_z P}$ (not calc.)")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_z \rho \dot{X}}$")
            # plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_{\phi}$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla (\widetilde{u}_x f_{\phi})$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla f^\phi$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_{r} \partial_r \widetilde{u}_\theta$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{r\phi} \partial_r \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan', label=r"$-\overline{X''\partial_\phi P/r sin \theta} = 0 \ bug $")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_\phi \rho \dot{X}}$")
            plt.plot(grd1, rhs5, color='yellow', label=r'$+G$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers		
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"g cm$^{-2}$ s$^{-2}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"g cm$^{-2}$ s$^{-2}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxZequation_' + element + '.png')
