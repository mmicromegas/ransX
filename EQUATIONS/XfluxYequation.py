import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al
import UTILS.Tools as uT
import UTILS.Errors as eR

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XfluxYequation(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, inuc, element, bconv, tconv, tke_diss, tauL, intc, data_prefix):
        super(XfluxYequation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht,'xzn0')
        nx = self.getRAdata(eht,'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht,'dd')[intc]
        ux = self.getRAdata(eht,'ux')[intc]
        uy = self.getRAdata(eht,'uy')[intc]
        uz = self.getRAdata(eht,'uz')[intc]
        pp = self.getRAdata(eht,'pp')[intc]
        xi = self.getRAdata(eht,'x' + inuc)[intc]

        uxy = self.getRAdata(eht,'uxy')[intc]
        uxz = self.getRAdata(eht,'uxz')[intc]

        ddux = self.getRAdata(eht,'ddux')[intc]
        dduy = self.getRAdata(eht,'dduy')[intc]
        dduz = self.getRAdata(eht,'dduz')[intc]
        ddgg = self.getRAdata(eht,'ddgg')[intc]

        dduxux = self.getRAdata(eht,'dduxux')[intc]
        dduyuy = self.getRAdata(eht,'dduyuy')[intc]
        dduzuz = self.getRAdata(eht,'dduzuz')[intc]
        dduxuy = self.getRAdata(eht,'dduxuy')[intc]
        dduxuz = self.getRAdata(eht,'dduxuz')[intc]

        uxux = self.getRAdata(eht,'uxux')[intc]
        uxuy = self.getRAdata(eht,'uxuy')[intc]
        uxuz = self.getRAdata(eht,'uxuz')[intc]
        uyuy = self.getRAdata(eht,'uyuy')[intc]
        uzuz = self.getRAdata(eht,'uzuz')[intc]

        ddxi = self.getRAdata(eht,'ddx' + inuc)[intc]
        xiux = self.getRAdata(eht,'x' + inuc + 'ux')[intc]
        ddxiux = self.getRAdata(eht,'ddx' + inuc + 'ux')[intc]
        ddxiuy = self.getRAdata(eht,'ddx' + inuc + 'uy')[intc]
        ddxiuz = self.getRAdata(eht,'ddx' + inuc + 'uz')[intc]
        ddxidot = self.getRAdata(eht,'ddx' + inuc + 'dot')[intc]

        gradypp = self.getRAdata(eht,'gradypp')[intc]

        ddxiuzuzcoty = self.getRAdata(eht,'ddx' + inuc + 'uzuzcoty')[intc]
        dduzuzcoty = self.getRAdata(eht,'dduzuzcoty')[intc]

        xigradxpp = self.getRAdata(eht,'x' + inuc + 'gradxpp')[intc]
        xigradypp = self.getRAdata(eht,'x' + inuc + 'gradypp')[intc]

        ddxidotux = self.getRAdata(eht,'ddx' + inuc + 'dotux')[intc]
        ddxidotuy = self.getRAdata(eht,'ddx' + inuc + 'dotuy')[intc]
        ddxidotuz = self.getRAdata(eht,'ddx' + inuc + 'dotuz')[intc]

        ddxiuxux = self.getRAdata(eht,'ddx' + inuc + 'uxux')[intc]
        ddxiuyuy = self.getRAdata(eht,'ddx' + inuc + 'uyuy')[intc]
        ddxiuzuz = self.getRAdata(eht,'ddx' + inuc + 'uzuz')[intc]
        ddxiuxuy = self.getRAdata(eht,'ddx' + inuc + 'uxuy')[intc]
        ddxiuxuz = self.getRAdata(eht,'ddx' + inuc + 'uxuz')[intc]

        xiddgg = self.getRAdata(eht,'x' + inuc + 'ddgg')[intc]
        uxdivu = self.getRAdata(eht,'uxdivu')[intc]

        divu = self.getRAdata(eht,'divu')[intc]
        gamma1 = self.getRAdata(eht,'gamma1')[intc]
        gamma3 = self.getRAdata(eht,'gamma3')[intc]

        gamma1 = self.getRAdata(eht,'ux')[intc]
        gamma3 = self.getRAdata(eht,'ux')[intc]

        fht_rxx = dduxux - ddux * ddux / dd
        fdil = (uxdivu - ux * divu)

        # store time series for time derivatives
        t_timec = self.getRAdata(eht,'timec')
        t_dd = self.getRAdata(eht,'dd')
        t_dduy = self.getRAdata(eht,'dduy')
        t_ddxi = self.getRAdata(eht,'ddx' + inuc)
        t_ddxiuy = self.getRAdata(eht,'ddx' + inuc + 'uy')

        ##################
        # Xi FLUX EQUATION 
        ##################

        # construct equation-specific mean fields
        t_fyi = t_ddxiuy - t_ddxi * t_dduy / t_dd

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
        self.minus_dt_fyi = -self.dt(t_fyi, xzn0, t_timec, intc)

        # LHS -div(dduxfyi)
        self.minus_div_fht_ux_fyi = -self.Div(fht_ux * fyi, xzn0)

        # RHS -div fyxi  
        self.minus_div_fyxi = -self.Div(fyxi, xzn0)

        # RHS -fxi gradx fht_uy
        self.minus_fxi_gradx_fht_uy = -fxi * self.Grad(fht_uy, xzn0)

        # RHS -ryx gradx fht_xi
        self.minus_ryx_gradx_fht_xi = -ryx * self.Grad(fht_xi, xzn0)

        if (ig == 1):
            # RHS -xff_grady_pp
            self.minus_eht_xff_grady_pp_o_rr = -(xigradypp - fht_xi*gradypp)
        elif (ig == 2):
            # RHS -xff_grady_pp_o_rr
            self.minus_eht_xff_grady_pp_o_rr = -(xigradypp - fht_xi*gradypp)/xzn0

        # RHS +uyff_eht_dd_xidot
        self.plus_uyff_eht_dd_xidot = +(ddxidotuy - (dduy / dd) * ddxidot)

        # RHS +gi 
        self.plus_gi = \
            -((ddxiuxuy - (ddxi / dd) * dduxuy) / xzn0 - \
              ((ddxiuzuzcoty + (ddxi / dd) * dduzuzcoty) / xzn0))

        # -res				   
        self.minus_resXiFlux = -(self.minus_dt_fyi + self.minus_div_fht_ux_fyi + self.minus_div_fyxi + \
                                 self.minus_fxi_gradx_fht_uy + self.minus_ryx_gradx_fht_xi + \
                                 self.minus_eht_xff_grady_pp_o_rr + self.plus_uyff_eht_dd_xidot + self.plus_gi)

        ######################
        # END Xi FLUX EQUATION 
        ######################	

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nx = nx
        self.inuc = inuc
        self.element = element
        self.fyi = fyi
        self.bconv = bconv
        self.tconv = tconv

    def plot_XfluxY(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xflux stratification in the model"""

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load and calculate DATA to plot
        plt1 = self.fyi

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xflux Y for ' + self.element)
        plt.plot(grd1, plt1, color='k', label=r'f')

        # convective boundary markers
        plt.axvline(self.bconv + 0.46e8, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers		
        # plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')
        # plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')

        # define and show x/y LABELS
        if (self.ig == 1):
            setxlabel = r'x (10$^{8}$ cm)'
        elif (self.ig == 2):
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        setylabel = r"$\overline{\rho} \widetilde{X''_i u''_\theta}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxY_' + element + '.png')

    def plot_XfluxY_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xi flux equation in the model"""

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fyi
        lhs1 = self.minus_div_fht_ux_fyi

        rhs0 = self.minus_div_fyxi
        rhs1 = self.minus_fxi_gradx_fht_uy
        rhs2 = self.minus_ryx_gradx_fht_xi
        rhs3 = self.minus_eht_xff_grady_pp_o_rr
        rhs4 = self.plus_uyff_eht_dd_xidot
        rhs5 = self.plus_gi

        if (self.ig == 1):
            res = -(lhs0 + lhs1 + rhs0 + rhs1 + rhs2 + rhs3 + rhs4)
            rhs5 = np.zeros(self.nx)
        elif (self.ig == 2):
            res = self.minus_resXiFlux

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xflux Y equation for ' + self.element)
        if (self.ig == 1):
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_y$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla_x (\widetilde{u}_x f_y)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f^y$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_{r} \partial_x \widetilde{u}_y$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{xy} \partial_x \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan', label=r"$-\overline{X''\partial_y P}$")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_y \rho \dot{X}}$")
            # plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif (self.ig == 2):
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_{\theta}$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla (\widetilde{u}_x f_{\theta})$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla f^\theta$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_{r} \partial_r \widetilde{u}_\theta$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{r\theta} \partial_x \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan', label=r"$-\overline{X''\partial_\theta P/r} = 0 \ bug$")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_\theta \rho \dot{X}}$")
            plt.plot(grd1, rhs5, color='yellow', label=r'$+G$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        # convective boundary markers		
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if (self.ig == 1):
            setxlabel = r'x (10$^{8}$ cm)'
        elif (self.ig == 2):
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        setylabel = r"g cm$^{-2}$ s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxYequation_' + element + '.png')
