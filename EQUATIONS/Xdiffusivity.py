import numpy as np
from scipy.optimize import curve_fit
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

    def __init__(self, filename, ig, fext, inuc, element, lc, uconv, bconv, tconv, tke_diss, tauL, intc, data_prefix):
        super(Xdiffusivity, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

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

        # model isotropic turbulence
        uxffuxff = (dduxux / dd - ddux * ddux / (dd * dd))
        uyffuyff = (dduyuy / dd - dduy * dduy / (dd * dd))
        uzffuzff = (dduzuz / dd - dduz * dduz / (dd * dd))

        uxfuxf = (uxux - ux * ux)
        uyfuyf = (uyuy - uy * uy)
        uzfuzf = (uzuz - uz * uz)

        uxfuyf = (uxuy - ux * uy)
        uxfuzf = (uxuz - ux * uz)

        uxy = self.getRAdata(eht, 'uxy')[intc]
        uxz = self.getRAdata(eht, 'uxz')[intc]


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

        self.data_prefix = data_prefix
        self.xzn0 = self.getRAdata(eht,'xzn0')
        self.element = element
        self.inuc = inuc
        self.lc = lc
        self.uconv = uconv

        self.bconv = bconv
        self.tconv = tconv
        self.fext = fext

    def plot_X_Ediffusivity(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
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

        # enthalpy flux 
        fhh = ddhhux - ddhh * ddux / dd

        # variance of temperature fluctuations		
        sigmatt = (ddttsq - ddtt * ddtt / dd) / dd

        # T_rms fluctuations
        tt_rms = sigmatt ** 0.5

        # effective diffusivity
        Deff = -fxi / (dd * self.Grad(fht_xi, xzn0))

        # urms diffusivity		
        Durms = (1. / 3.) * uconv * lc

        # pressure scale heigth
        hp = - pp / self.Grad(pp, xzn0)
        # print(hp)

        hp = 2.5e8

        # mlt velocity
        alphae = 0.1
        u_mlt = fhh / (alphae * dd * fht_cp * tt_rms)
        Dumlt = (1. / 3.) * u_mlt * lc

        # this should be OS independent
        dir_model = os.path.join(os.path.realpath('.'), 'DATA_D', 'INIMODEL', 'imodel.tycho')

        data = np.loadtxt(dir_model, skiprows=26)
        nxmax = 500

        rr = data[1:nxmax, 2]
        vmlt_3 = data[1:nxmax, 8]
        u_mltini = vmlt_3

        Dumlt1 = (1. / 3.) * u_mltini * lc

        alpha = 1.5
        Dumlt2 = (1. / 3.) * u_mltini * alpha * hp

        alpha = 1.6
        Dumlt3 = (1. / 3.) * u_mltini * alpha * hp

        self.lagr = (4.*np.pi*(self.xzn0**2.)*self.dd)**2.

        term0 = Deff
        term1 = Durms
        # term2 = Dumlt1
        # term3 = Dumlt2
        # term4 = Dumlt3
        term5 = Dumlt  # u_mlt = fhh / (alphae * dd * fht_cp * tt_rms)

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries   
        to_plot = [term0, term1, term5]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 		
        plt.title(r'Eulerian Diff for ' + self.element)
        plt.plot(grd1, term0, linestyle = '-', color='k', label=r"$D_{eff} = - f_i/(\overline{\rho} \ \partial_r \widetilde{X}_i)$")
        plt.plot(grd1,term1,label=r"$D_{urms} = (1/3) \ u_{rms} \ l_c $")
        # plt.plot(rr,term2,label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ l_c $")
        # plt.plot(rr,term3, label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.5)")
        # plt.plot(rr,term4,label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.6)")
        plt.plot(grd1,term5,label=r"$D_{mlt} = (1/3) \ u_{MLT} \ l_c $")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # https://stackoverflow.com/questions/19206332/gaussian-fit-for-python		

        def gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * (sigma ** 2)))

        # p0 = [1.e15, 6.e8, 5.e7]
        # coeff, var_matrix = curve_fit(gauss, self.xzn0, Deff, p0=[1.e15, 6.e8, 5.e7]
        # Get the fitted curve
        # Deff_fit = gauss(self.xzn0, *coeff)

        # plt.plot(grd1,Deff_fit,label=r"$gauss fit$",linewidth=0.7)

        ampl = max(term5)
        # xx0 = (self.bconv+0.46e8+self.tconv)/2.
        xx0 = (self.bconv + self.tconv) / 2.
        width = 5.e7

        Dgauss = gauss(self.xzn0,ampl,xx0,width)
        plt.plot(grd1,Dgauss,color='b',label=r"$D_{gauss}$")

        plt.plot(grd1,self.Drr1,color='m',label=r"$D_{rogers1981}$")

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


