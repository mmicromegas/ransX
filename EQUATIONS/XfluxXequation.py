import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import os


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XfluxXequation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, inuc, element, bconv, tconv, tke_diss, tauL, intc, data_prefix):
        super(XfluxXequation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

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

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uxuy = self.getRAdata(eht, 'uxuy')[intc]
        uxuz = self.getRAdata(eht, 'uxuz')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]

        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        xiux = self.getRAdata(eht, 'x' + inuc + 'ux')[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]

        # print(ddxidot)
        # print("-------------------")
        # print(ddgg)

        ddxidotux = self.getRAdata(eht, 'ddx' + inuc + 'dotux')[intc]
        ddxiuxux = self.getRAdata(eht, 'ddx' + inuc + 'uxux')[intc]
        ddxiuyuy = self.getRAdata(eht, 'ddx' + inuc + 'uyuy')[intc]
        ddxiuzuz = self.getRAdata(eht, 'ddx' + inuc + 'uzuz')[intc]

        xiddgg = self.getRAdata(eht, 'x' + inuc + 'ddgg')[intc]

        xigradxpp = self.getRAdata(eht, 'x' + inuc + 'gradxpp')[intc]

        # Reynolds-averaged mean fields for flux modelling:
        ddcp = self.getRAdata(eht, 'ddcp')[intc]
        ddtt = self.getRAdata(eht, 'ddtt')[intc]
        ddhh = self.getRAdata(eht, 'ddhh')[intc]
        ddhhux = self.getRAdata(eht, 'ddhhux')[intc]
        ddttsq = self.getRAdata(eht, 'ddttsq')[intc]

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
        t_ddux = self.getRAdata(eht, 'ddux')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')

        ##################
        # Xi FLUX EQUATION 
        ##################

        # construct equation-specific mean fields
        t_fxi = t_ddxiux - t_ddxi * t_ddux / t_dd

        fht_ux = ddux / dd
        fht_xi = ddxi / dd

        rxx = dduxux - ddux * ddux / dd
        fxi = ddxiux - ddxi * ddux / dd
        fxxi = ddxiuxux - (ddxi / dd) * dduxux - 2. * (ddux / dd) * ddxiux + 2. * ddxi * ddux * ddux / (dd * dd)

        # this is for Rogers et al.1989 model		
        fxi1 = fxi / dd
        fxi2 = xiux - xi * ux

        # LHS -dq/dt 
        self.minus_dt_fxi = -self.dt(t_fxi, xzn0, t_timec, intc)

        # LHS -div(dduxfxi)
        self.minus_div_fht_ux_fxi = -self.Div(fht_ux * fxi, xzn0)

        # RHS -div fxxi  
        self.minus_div_fxxi = -self.Div(fxxi, xzn0)

        # RHS -fxi gradx fht_ux
        self.minus_fxi_gradx_fht_ux = -fxi * self.Grad(fht_ux, xzn0)

        # RHS -rxx d_r xi
        self.minus_rxx_gradx_fht_xi = -rxx * self.Grad(fht_xi, xzn0)

        # RHS - X''i gradx P - X''_i gradx P'
        self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff = \
            -(xi * self.Grad(pp, xzn0) - fht_xi * self.Grad(pp, xzn0)) - (xigradxpp - xi * self.Grad(pp, xzn0))

        # RHS +uxff_eht_dd_xidot
        self.plus_uxff_eht_dd_xidot = +(ddxidotux - (ddux / dd) * ddxidot)

        # RHS +gi 
        self.plus_gi = \
            -(ddxiuyuy - (ddxi / dd) * dduyuy - 2. * (dduy / dd) + 2. * ddxi * dduy * dduy / (dd * dd)) / xzn0 - \
            (ddxiuzuz - (ddxi / dd) * dduzuz - 2. * (dduz / dd) + 2. * ddxi * dduz * dduz / (dd * dd)) / xzn0 + \
            (ddxiuyuy - (ddxi / dd) * dduyuy) / xzn0 + \
            (ddxiuzuz - (ddxi / dd) * dduzuz) / xzn0

        # -res				   
        self.minus_resXiFlux = -(self.minus_dt_fxi + self.minus_div_fht_ux_fxi + self.minus_div_fxxi +
                                 self.minus_fxi_gradx_fht_ux + self.minus_rxx_gradx_fht_xi +
                                 self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff +
                                 self.plus_uxff_eht_dd_xidot + self.plus_gi)

        ######################
        # END Xi FLUX EQUATION 
        ######################

        # -eht_xiddgg + fht_xx eht_ddgg		
        self.minus_xiddgg = -xiddgg
        self.plus_fht_xi_eht_ddgg = fht_xi * ddgg

        self.minus_xiddgg_plus_fht_xi_eht_ddgg = -xiddgg + fht_xi * ddgg

        # -res				   
        self.minus_resXiFlux2 = -(self.minus_dt_fxi + self.minus_div_fht_ux_fxi + self.minus_div_fxxi +
                                  self.minus_fxi_gradx_fht_ux + self.minus_rxx_gradx_fht_xi +
                                  self.minus_xiddgg_plus_fht_xi_eht_ddgg +
                                  self.plus_uxff_eht_dd_xidot + self.plus_gi)

        # variance of temperature fluctuations		
        sigmatt = (ddttsq - ddtt * ddtt / dd) / dd

        # enthalpy flux 
        fhh = ddhhux - ddhh * ddux / dd

        # heat capacity		
        fht_cp = ddcp / dd

        # mlt velocity		
        alphae = 1.
        u_mlt = fhh / (alphae * fht_cp * sigmatt)

        self.minus_ddxiumlt = -dd * xi * u_mlt

        # models
        self.model_1 = self.minus_rxx_gradx_fht_xi
        self.model_2 = self.minus_ddxiumlt

        # grad models		
        self.plus_gradx_fxi = +self.Grad(fxi, xzn0)
        cnst = gamma1
        self.minus_cnst_dd_fxi_fdil_o_fht_rxx = -cnst * dd * fxi * fdil / fht_rxx

        hp = 2.5e8

        # Dgauss gradient model	

        # this should be OS independent
        dir_model = os.path.join(os.path.realpath('.'), 'DATA', 'INIMODEL', 'imodel.tycho')

        data = np.loadtxt(dir_model, skiprows=26)
        nxmax = 500

        rr = data[1:nxmax, 2]
        vmlt = data[1:nxmax, 8]
        u_mlt = np.interp(xzn0, rr, vmlt)

        # Dumlt1     = (1./3.)*u_mlt*lc

        alpha = 1.5
        Dumlt2 = (1. / 3.) * u_mlt * alpha * hp

        alpha = 1.6
        Dumlt3 = (1. / 3.) * u_mlt * alpha * hp

        def gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * (sigma ** 2)))

        Dmlt = Dumlt3

        ampl = max(Dmlt)
        xx0 = (bconv + 0.46e8 + tconv) / 2.
        width = 4.e7

        Dgauss = gauss(xzn0, ampl, xx0, width)

        self.model_3 = -Dgauss * dd * self.Grad(fht_xi, xzn0)
        self.model_4 = -Dumlt3 * dd * self.Grad(fht_xi, xzn0)

        # model isotropic turbulence		

        uxffuxff = (dduxux / dd - ddux * ddux / (dd * dd))
        uyffuyff = (dduyuy / dd - dduy * dduy / (dd * dd))
        uzffuzff = (dduzuz / dd - dduz * dduz / (dd * dd))

        uxfuxf = (uxux - ux * ux)
        uyfuyf = (uyuy - uy * uy)
        uzfuzf = (uzuz - uz * uz)

        uxfuyf = (uxuy - ux * uy)
        uxfuzf = (uxuz - ux * uz)

        cd1 = 100.  # assumption
        cd2 = 10.
        # q = uxffuxff + uyffuyff + uzffuzff
        q = uxfuxf + uyfuyf + uzfuzf
        self.model_5 = -(dd / (3. * cd1)) * ((q ** 2) / tke_diss) * self.Grad(fht_xi, xzn0)

        Drr = +(tauL / cd2) * uxfuxf + uxy * tauL * (tauL / cd2 ** 2) * (-uxfuyf)
        Drt = +(tauL / cd2) * uxfuyf - uxy * tauL * (tauL / cd2 ** 2) * (uyfuyf)
        Drp = +(tauL / cd2) * uxfuzf - uxy * tauL * (tauL / cd2 ** 2) * (uzfuzf)

        Drr1 = +(tauL / cd1) * uxfuxf + uxy * tauL * (tauL / cd1 ** 2) * (-uxfuyf)
        Drr2 = +(tauL / cd1) * uxfuxf + uxz * tauL * (tauL / cd1 ** 2) * (-uxfuyf)

        self.model_6 = dd * (Drr + Drt + Drp) * self.Grad(fht_xi, xzn0)

        self.model_1_rogers1989 = -Drr1 * self.Grad(xi, xzn0)
        self.model_2_rogers1989 = -Drr2 * self.Grad(xi, xzn0)

        # turbulent thermal diffusion model (the extra term)
        self.model_3_turb_thermal_diff = self.model_1_rogers1989 + (Drr1*self.Grad(dd, xzn0)/dd)*xi
        self.model_4 = (Drr1*self.Grad(dd, xzn0)/dd)*xi

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.inuc = inuc
        self.element = element
        self.fxi = fxi
        self.fxi1 = fxi1
        self.fxi2 = fxi2
        self.bconv = bconv
        self.tconv = tconv
        self.dd = dd

    def plot_XfluxX(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xflux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XfluxXEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load and calculate DATA to plot
        plt1 = self.fxi
        plt2 = self.model_1
        # plt3 = self.model_2
        # plt4 = self.model_3
        # plt5 = self.model_4
        # plt6 = self.model_5
        # plt7 = self.model_6

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xflux X for ' + self.element)
        plt.plot(grd1, plt1, color='k', label=r'f')
        # plt.plot(grd1,plt2,color='c',label=r"$-\widetilde{R}_{rr} \partial_r \widetilde{X}$")
        # plt.plot(grd1,plt3,color='g',label=r"$-\overline{\rho} \ \overline{X} \ u_{mlt}$")
        # plt.plot(grd1,plt4,color='r',label=r"$-D_{gauss} \ \partial_r \widetilde{X}$")
        # plt.plot(grd1,plt5,color='b',label=r"$-D_{mlt} \ \partial_r \widetilde{X}$",linewidth=0.7)
        # plt.plot(grd1,plt6,color='g',label=r"$-1/3 C_D (\overline{u''_i u''_i}^2/\varepsilon_{tke}) \ \partial_r \widetilde{X}$")
        # plt.plot(grd1,plt7,color='c',label=r"$-(D_{rr}+D_{r\theta}+D_{r\phi}) \ \partial_r \widetilde{X}$")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers		
        # plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')
        # plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\overline{\rho} \widetilde{X''_i u''_x}$ (g cm$^{-2}$ s$^{-1}$)"
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
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxX_' + element + '.png')

    def plot_XfluxXRogers1989(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xflux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XfluxXEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load and calculate DATA to plot
        plt1 = self.dd*self.fxi1
        plt2 = self.dd*self.fxi2

        plt3 = self.dd*self.model_1_rogers1989
        plt4 = self.dd*self.model_2_rogers1989

        plt5 = self.dd*self.model_3_turb_thermal_diff
        plt6 = self.dd*self.model_4

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3, plt4, plt5, plt6]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xflux for ' + self.element)
        plt.plot(grd1, plt1, color='k', label=r"$+\overline{\rho}\widetilde{X''u''_r}$")
        plt.plot(grd1, plt2, color='r', linestyle='--', label=r"$+\overline{\rho}\overline{X'u'_r}$")

        plt.plot(grd1, plt3, color='g', label=r"$model (1)$")
        # plt.plot(grd1, plt4, color='b', linestyle='--', label=r"$model (2)$")
        plt.plot(grd1, plt5, color='m', label=r"$model (1) + D (\nabla \rho / \rho) X$")
        plt.plot(grd1, plt6, color='pink', label=r"$+D (\nabla \rho / \rho) X$")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers		
        # plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')
        # plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$f$ (g cm$^{-2}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$f$ (cm s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxXRogers1989models_' + element + '.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxXRogers1989models_' + element + '.eps')

    def plot_Xflux_gradient(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xflux stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XfluxXEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load and calculate DATA to plot
        plt1 = self.plus_gradx_fxi
        plt2 = self.minus_cnst_dd_fxi_fdil_o_fht_rxx

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('grad Xflux for ' + self.element)
        plt.plot(grd1, plt1, color='k', label=r"$+\partial_r f$")
        plt.plot(grd1, plt2, color='r', label=r"$.$")

        # convective boundary markers
        plt.axvline(self.bconv + 0.46e8, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers		
        # plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')
        # plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\partial_r \overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-3}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\partial_r \overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-3}$ s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_model_XfluxX_' + element + '.png')

    def plot_XfluxX_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xi flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XfluxXEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fxi
        lhs1 = self.minus_div_fht_ux_fxi

        rhs0 = self.minus_div_fxxi
        rhs1 = self.minus_fxi_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_xi
        rhs3 = self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff
        rhs4 = self.plus_uxff_eht_dd_xidot
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
        plt.title('Xflux X equation for ' + self.element)
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_i$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla_x (\widetilde{u}_x f)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f^x_i$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_i \partial_x \widetilde{u}_x$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{xx} \partial_x \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan',
                     label=r"$-\overline{X''} \partial_x \overline{P} - \overline{X'' \partial_x P'}$")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_x \rho \dot{X}}$")
            # plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_i$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla_r (\widetilde{u}_r f)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_r f^r_i$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_i \partial_r \widetilde{u}_r$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{rr} \partial_r \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan',
                     label=r"$-\overline{X''} \partial_r \overline{P} - \overline{X'' \partial_r P'}$")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_r \rho \dot{X}}$")
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
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxXequation_' + element + '.png')

    def plot_XfluxX_equation2(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xi flux equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XfluxXEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fxi
        lhs1 = self.minus_div_fht_ux_fxi

        rhs0 = self.minus_div_fxxi
        rhs1 = self.minus_fxi_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_xi
        # rhs3 = self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff
        # rhs3 = self.minus_xiddgg_plus_fht_xi_eht_ddgg
        rhs3a = self.minus_xiddgg
        rhs3b = self.plus_fht_xi_eht_ddgg
        rhs3 = rhs3a + rhs3b
        rhs4 = self.plus_uxff_eht_dd_xidot
        rhs5 = self.plus_gi

        res = self.minus_resXiFlux2

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xflux X equation for ' + self.element)
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_i$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla_x (\widetilde{u}_x f)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f^x_i$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_i \partial_x \widetilde{u}_x$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{xx} \partial_x \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan', label=r"$-\overline{X \rho g_x} + \widetilde{X} \overline{\rho g_x}$")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_x \rho \dot{X}}$")
            # plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#8B3626', label=r'$-\partial_t f_i$')
            plt.plot(grd1, lhs1, color='#FF7256', label=r'$-\nabla_r (\widetilde{u}_r f)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_r f^r_i$')
            plt.plot(grd1, rhs1, color='g', label=r'$-f_i \partial_r \widetilde{u}_r$')
            plt.plot(grd1, rhs2, color='r', label=r'$-R_{rr} \partial_r \widetilde{X}$')
            plt.plot(grd1, rhs3, color='cyan', label=r"$-\overline{X \rho g_r} - \widetilde{X} \overline{\rho g_r}$")
            # plt.plot(grd1,rhs3a,color='cyan',label=r"$-\overline{X \rho g_r}$")
            # plt.plot(grd1,rhs3b,color='brown',label=r"$+\widetilde{X} \overline{\rho g_r}$")
            plt.plot(grd1, rhs4, color='purple', label=r"$+\overline{u''_r \rho \dot{X}}$")
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
        plt.legend(loc=ilg, prop={'size': 10})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_XfluxXequation2_' + element + '.png')
