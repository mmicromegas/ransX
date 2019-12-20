import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XfluxYequation(calc.CALCULUS, al.ALIMIT, object):

    def __init__(self, filename, ig, inuc, element, bconv, tconv, tke_diss, tauL, intc, data_prefix):
        super(XfluxYequation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0'))
        nx = np.asarray(eht.item().get('nx'))

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])
        uy = np.asarray(eht.item().get('uy')[intc])
        uz = np.asarray(eht.item().get('uz')[intc])
        pp = np.asarray(eht.item().get('pp')[intc])
        xi = np.asarray(eht.item().get('x' + inuc)[intc])

        uxy = np.asarray(eht.item().get('uxy')[intc])
        uxz = np.asarray(eht.item().get('uxz')[intc])

        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])
        ddgg = np.asarray(eht.item().get('ddgg')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])
        dduxuy = np.asarray(eht.item().get('dduxuy')[intc])
        dduxuz = np.asarray(eht.item().get('dduxuz')[intc])

        uxux = np.asarray(eht.item().get('uxux')[intc])
        uxuy = np.asarray(eht.item().get('uxuy')[intc])
        uxuz = np.asarray(eht.item().get('uxuz')[intc])
        uyuy = np.asarray(eht.item().get('uyuy')[intc])
        uzuz = np.asarray(eht.item().get('uzuz')[intc])

        ddxi = np.asarray(eht.item().get('ddx' + inuc)[intc])
        xiux = np.asarray(eht.item().get('x' + inuc + 'ux')[intc])
        ddxiux = np.asarray(eht.item().get('ddx' + inuc + 'ux')[intc])
        ddxiuy = np.asarray(eht.item().get('ddx' + inuc + 'uy')[intc])
        ddxiuz = np.asarray(eht.item().get('ddx' + inuc + 'uz')[intc])
        ddxidot = np.asarray(eht.item().get('ddx' + inuc + 'dot')[intc])

        gradypp = np.asarray(eht.item().get('gradypp')[intc])

        ddxiuzuzcoty = np.asarray(eht.item().get('ddx' + inuc + 'uzuzcoty')[intc])
        dduzuzcoty = np.asarray(eht.item().get('dduzuzcoty')[intc])

        xigradxpp = np.asarray(eht.item().get('x' + inuc + 'gradxpp')[intc])
        xigradypp = np.asarray(eht.item().get('x' + inuc + 'gradypp')[intc])

        ddxidotux = np.asarray(eht.item().get('ddx' + inuc + 'dotux')[intc])
        ddxidotuy = np.asarray(eht.item().get('ddx' + inuc + 'dotuy')[intc])
        ddxidotuz = np.asarray(eht.item().get('ddx' + inuc + 'dotuz')[intc])

        ddxiuxux = np.asarray(eht.item().get('ddx' + inuc + 'uxux')[intc])
        ddxiuyuy = np.asarray(eht.item().get('ddx' + inuc + 'uyuy')[intc])
        ddxiuzuz = np.asarray(eht.item().get('ddx' + inuc + 'uzuz')[intc])
        ddxiuxuy = np.asarray(eht.item().get('ddx' + inuc + 'uxuy')[intc])
        ddxiuxuz = np.asarray(eht.item().get('ddx' + inuc + 'uxuz')[intc])

        xiddgg = np.asarray(eht.item().get('x' + inuc + 'ddgg')[intc])
        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])

        divu = np.asarray(eht.item().get('divu')[intc])
        gamma1 = np.asarray(eht.item().get('gamma1')[intc])
        gamma3 = np.asarray(eht.item().get('gamma3')[intc])

        gamma1 = np.asarray(eht.item().get('ux')[intc])
        gamma3 = np.asarray(eht.item().get('ux')[intc])

        fht_rxx = dduxux - ddux * ddux / dd
        fdil = (uxdivu - ux * divu)

        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))
        t_dd = np.asarray(eht.item().get('dd'))
        t_dduy = np.asarray(eht.item().get('dduy'))
        t_ddxi = np.asarray(eht.item().get('ddx' + inuc))
        t_ddxiuy = np.asarray(eht.item().get('ddx' + inuc + 'uy'))

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

        # RHS -xff_grady_pp_o_rr
        # self.minus_eht_xff_grady_pp_o_rr = -(xigradypp - fht_xi*gradypp)/xzn0
        self.minus_eht_xff_grady_pp_o_rr = np.zeros(nx)

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
            plt.plot(grd1, rhs3, color='cyan', label=r"$-\overline{X''\partial_y P/r}$")
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
