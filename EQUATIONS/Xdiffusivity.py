import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import os
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Xdiffusivity(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, ieos, inuc, element, lc, uconv, bconv, tconv, cnvz_in_hp,
                 tke_diss, tauL, super_ad_i, super_ad_o, intc, data_prefix):
        super(Xdiffusivity, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		
        # assign global data to be shared across whole class

        dd = self.getRAdata(eht,'dd')[intc]
        self.dd = self.getRAdata(eht,'dd')[intc]
        self.pp = self.getRAdata(eht,'pp')[intc]
        self.tt = self.getRAdata(eht,'tt')[intc]
        self.ddxi = self.getRAdata(eht,'ddx' + inuc)[intc]
        self.ddux = self.getRAdata(eht,'ddux')[intc]
        self.ddtt = self.getRAdata(eht,'ddtt')[intc]
        self.ddhh = self.getRAdata(eht,'ddhh')[intc]
        self.ddcp = self.getRAdata(eht,'ddcp')[intc]
        self.ddxiux = self.getRAdata(eht,'ddx' + inuc + 'ux')[intc]
        self.ddhhux = self.getRAdata(eht,'ddhhux')[intc]
        self.ddttsq = self.getRAdata(eht,'ddttsq')[intc]

        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]

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

        uxy = self.getRAdata(eht, 'uxy')[intc]
        uxz = self.getRAdata(eht, 'uxz')[intc]

        ux = self.getRAdata(eht, 'ux')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]

        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
        gamma1 = self.getRAdata(eht, 'gamma1')[intc]
        gamma3 = self.getRAdata(eht, 'gamma3')[intc]

        uxdivu = self.getRAdata(eht, 'ux')[intc]
        gamma1 = self.getRAdata(eht, 'ux')[intc]
        gamma3 = self.getRAdata(eht, 'ux')[intc]


        dd = self.dd
        pp = self.pp

        ddux = self.ddux
        ddxi = self.ddxi
        ddtt = self.ddtt
        ddhh = self.ddhh
        ddcp = self.ddcp

        ddxiux = self.ddxiux
        ddhhux = self.ddhhux
        ddttsq = self.ddttsq

        fht_xi = ddxi / dd
        fht_cp = ddcp / dd

        # composition flux
        fxi = ddxiux - ddxi * ddux / dd

        # effective diffusivity
        self.Deff = -fxi / (dd * self.Grad(fht_xi, xzn0))

        # urms diffusivity
        urms = uconv
        self.Durms = (1. / 3.) * urms * lc

        # enthalpy flux
        fhh = ddhhux - ddhh * ddux / dd

        # variance of temperature fluctuations
        sigmatt = (ddttsq - ddtt * ddtt / dd) / dd

        # T_rms fluctuations
        tt_rms = sigmatt ** 0.5

        # mlt velocity
        alphae = 0.1
        u_mlt = fhh / (alphae * dd * fht_cp * tt_rms)
        alpha_mlt = 1.5

        self.Dumlt = (1. / 3.) * u_mlt * lc

        # this should be OS independent
        dir_model = os.path.join(os.path.realpath('.'), 'DATA_D', 'INIMODEL', 'imodel.tycho')

        data = np.loadtxt(dir_model, skiprows=26)
        nxmax = 500

        rr = data[1:nxmax, 2]
        vmlt_3 = data[1:nxmax, 8]
        u_mltini = vmlt_3

        hp = 2.5e8

        self.Dumlt1 = (1. / 3.) * u_mltini * lc

        alpha = 1.5
        self.Dumlt2 = (1. / 3.) * u_mltini * alpha * hp

        alpha = 1.6
        self.Dumlt3 = (1. / 3.) * u_mltini * alpha * hp

        self.lagr = (4.*np.pi*(xzn0**2.)*self.dd)**2.

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
        # self.model_5 = -(dd / (3. * cd1)) * ((q ** 2) / tke_diss) * self.Grad(fht_xi, xzn0)

        Drr = +(tauL / cd2) * uxfuxf + uxy * tauL * (tauL / cd2 ** 2) * (-uxfuyf)
        Drt = +(tauL / cd2) * uxfuyf - uxy * tauL * (tauL / cd2 ** 2) * (uyfuyf)
        Drp = +(tauL / cd2) * uxfuzf - uxy * tauL * (tauL / cd2 ** 2) * (uzfuzf)

        # self.model_1_rogers1989 = -Drr1 * self.Grad(xi, xzn0)
        self.Drr1 = +(tauL / cd1) * uxfuxf + uxy * tauL * (tauL / cd1 ** 2) * (-uxfuyf)

        # self.model_2_rogers1989 = -Drr2 * self.Grad(xi, xzn0)
        self.Drr2 = +(tauL / cd1) * uxfuxf + uxz * tauL * (tauL / cd1 ** 2) * (-uxfuyf)

        # Gaussian diffusivity
        # https://stackoverflow.com/questions/19206332/gaussian-fit-for-python

        def gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * (sigma ** 2)))

        # p0 = [1.e15, 6.e8, 5.e7]
        # coeff, var_matrix = curve_fit(gauss, self.xzn0, Deff, p0=[1.e15, 6.e8, 5.e7]
        # Get the fitted curve
        # Deff_fit = gauss(self.xzn0, *coeff)

        # plt.plot(grd1,Deff_fit,label=r"$gauss fit$",linewidth=0.7)

        ampl = max(self.Dumlt)
        # xx0 = (self.bconv+0.46e8+self.tconv)/2.
        xx0 = (bconv + tconv) / 2.
        width = 6.e7

        self.Dgauss = gauss(xzn0,ampl,xx0,width)


        fht_rxx = dduxux - ddux * ddux / dd
        fdil = (uxdivu - ux * divu)

        #######################
        # Xi TRANSPORT EQUATION
        #######################

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
        self.minus_dt_dd_fht_xi = -self.dt(t_dd * t_fht_xi, xzn0, t_timec, intc)

        # LHS -div(ddXiux)
        self.minus_div_eht_dd_fht_ux_fht_xi = -self.Div(dd * fht_ux * fht_xi, xzn0)

        # RHS -div fxi
        self.minus_div_fxi = -self.Div(fxi, xzn0)

        # RHS +ddXidot
        self.plus_ddxidot = +ddxidot

        # -res
        self.minus_resXiTransport = -(self.minus_dt_dd_fht_xi + self.minus_div_eht_dd_fht_ux_fht_xi + self.minus_div_fxi + self.plus_ddxidot)

        ###########################
        # END Xi TRANSPORT EQUATION
        ###########################

        self.minus_dt_fht_xi = -self.dt(t_fht_xi, xzn0, t_timec, intc)

        pp = self.getRAdata(eht, 'pp')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        mu = self.getRAdata(eht, 'abar')[intc]
        chim = self.getRAdata(eht, 'chim')[intc]
        chit = self.getRAdata(eht, 'chit')[intc]
        gamma2 = self.getRAdata(eht, 'gamma2')[intc]
        # print(chim,chit,gamma2)

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma2 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients
        self.nabla = self.deriv(lntt, lnpp)
        self.nabla_ad = (gamma2 - 1.) / gamma2


        self.data_prefix = data_prefix
        self.xzn0 = self.getRAdata(eht,'xzn0')
        self.element = element
        self.inuc = inuc
        self.lc = lc
        self.uconv = uconv
        self.fht_xi = fht_xi

        self.bconv = bconv
        self.tconv = tconv
        self.fext = fext
        self.nx = nx

        self.super_ad_i = super_ad_i
        self.super_ad_o = super_ad_o

    def plot_X_Ediffusivity(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # Eulerian diffusivity

        if self.ig != 1 and self.ig != 2:
            print("ERROR(Xdiffusivity.py):" + self.errorGeometry(self.ig))
            sys.exit()

        grd1 = self.xzn0

        # convert nuc ID to string
        xnucid = str(self.inuc)
        lc = self.lc
        uconv = self.uconv
        element = self.element

        term0 = self.Deff
        term1 = self.Durms
        # term2 = self.Dumlt1
        # term3 = self.Dumlt2
        # term4 = self.Dumlt3
        term5 = self.Dumlt  # u_mlt = fhh / (alphae * dd * fht_cp * tt_rms)

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries   
        to_plot = [term0, term1, term5]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 		
        plt.title(r'Eulerian diffusivity for ' + self.element)
        plt.plot(grd1, term0, linestyle = '-', color='k', label=r"$D_{eff} = - f_i/(\overline{\rho} \ \partial_r \widetilde{X}_i)$")
        plt.plot(grd1,term1,label=r"$D_{urms} = (1/3) \ u_{rms} \ l_c $")
        # plt.plot(rr,term2,label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ l_c $")
        # plt.plot(rr,term3, label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.5)")
        # plt.plot(rr,term4,label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.6)")
        plt.plot(grd1,term5,label=r"$D_{mlt} = (1/3) \ u_{mlt} \ l_c $")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')


        #plt.plot(grd1,Dgauss,color='b',label=r"$D_{gauss}$")

        #plt.plot(grd1,self.Drr1,color='m',label=r"$D_{rogers1981}$")

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 15})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'Ediff_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'Ediff_' + element + '.eps')


    def plot_X_Ediffusivity2(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # Eulerian diffusivity

        if self.ig != 1 and self.ig != 2:
            print("ERROR(Xdiffusivity.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)
        lc = self.lc
        uconv = self.uconv
        element = self.element

        # load x GRID
        grd1 = self.xzn0
        xzn0 = self.xzn0

        dd = self.dd
        fht_xi = self.fht_xi

        idxl, idxr = self.idx_bndry(self.bconv, self.tconv)

        term0 = self.Deff[idxl:idxr]
        term1 = self.Durms[idxl:idxr]
        # term2 = self.Dumlt1[idxl:idxr]
        # term3 = self.Dumlt2[idxl:idxr]
        # term4 = self.Dumlt3[idxl:idxr]
        term5 = self.Dumlt[idxl:idxr]  # u_mlt = fhh / (alphae * dd * fht_cp * tt_rms)

        #idxl = 0
        #idxr = self.nx

        xx = self.xzn0[idxl:idxr]
        yy = (xx**2.)*(-self.minus_dt_dd_fht_xi[idxl:idxr] -
                       self.minus_div_eht_dd_fht_ux_fht_xi[idxl:idxr] -
                       self.plus_ddxidot[idxl:idxr])

        Deff2 = (1./(dd[idxl:idxr]*xx*xx*self.Grad(fht_xi[idxl:idxr],xx)))*integrate.cumtrapz(yy, xx, initial=0)

        xx = self.xzn0[idxl:idxr]
        yy = (xx**2.)*(-self.minus_dt_fht_xi[idxl:idxr])

        Deff3 = (1./(xx*xx*self.Grad(fht_xi[idxl:idxr],xx)))*integrate.cumtrapz(yy, xx, initial=0)

        term6 = Deff2
        term7 = Deff3
        term8 = self.Dgauss[idxl:idxr]

        # create FIGURE
        plt.figure(figsize=(7, 6))

        #plt.yscale('symlog')

        # set plot boundaries
        to_plot = [term0,term1,term5,term6,term7,term8]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title(r'radial diffusivity profile for ' + self.element)
        #plt.plot(grd1, term0, linestyle = '-', color='k', label=r"$D_{eff} = - f_i/(\overline{\rho} \ \partial_r \widetilde{X}_i)$")

        plt.semilogy(xx, term0, label=r"$D_{eff}$",color='g')
        plt.semilogy(xx, term1, label=r"$D_{rms}$",color='orange')
        plt.semilogy(xx, term5, label=r"$D_{mlt}$",color='m')
        plt.semilogy(xx, term6, label=r"$D_{fullx}$",color='r')
        plt.semilogy(xx, term7, label=r"$D_{diff}$",color='b')
        plt.semilogy(xx, term8, label=r"$D_{gauss}$",linestyle='--')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"cm$^{-2}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 15})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'Ediff2_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'Ediff2_' + element + '.eps')