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

class LuminosityEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, ieos, fext, intc, tke_diss, bconv, tconv, data_prefix):
        super(LuminosityEquation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        cp = self.getRAdata(eht, 'cp')[intc]
        gg = self.getRAdata(eht, 'gg')[intc]
        abar = self.getRAdata(eht, 'abar')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        ddttux = self.getRAdata(eht, 'ddttux')[intc]
        dduxttx = self.getRAdata(eht, 'dduxttx')[intc]
        dduytty = self.getRAdata(eht, 'dduytty')[intc]
        dduzttz = self.getRAdata(eht, 'dduzttz')[intc]

        eiuxddx = self.getRAdata(eht, 'eiuxddx')[intc]
        eiuyddy = self.getRAdata(eht, 'eiuyddy')[intc]
        eiuzddz = self.getRAdata(eht, 'eiuzddz')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]
        dduxuy = self.getRAdata(eht, 'dduxuy')[intc]
        dduxuz = self.getRAdata(eht, 'dduxuz')[intc]

        ddekux = self.getRAdata(eht, 'ddekux')[intc]
        ddek = self.getRAdata(eht, 'ddek')[intc]

        ddei = self.getRAdata(eht, 'ddei')[intc]
        ddeiux = self.getRAdata(eht, 'ddeiux')[intc]
        eiux = self.getRAdata(eht, 'eiux')[intc]

        ddetux = self.getRAdata(eht, 'ddetux')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        dddivu = self.getRAdata(eht, 'dddivu')[intc]
        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        ppux = self.getRAdata(eht, 'ppux')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        chim = self.getRAdata(eht, 'chim')[intc]
        chit = self.getRAdata(eht, 'chit')[intc]
        chid = self.getRAdata(eht, 'chid')[intc]

        gamma1 = self.getRAdata(eht, 'gamma1')[intc]

        gascon = 8.3144629e7 # gas constant in cgs

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        # print(gamma1)
        # print("-----------")
        # print((gamma1/(gamma1-1.))*gascon/abar)
        # print("-----------")
        # print(cp)


        ##########################
        # HSSE LUMINOSITY EQUATION 
        ##########################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_tt = self.getRAdata(eht, 'tt')
        t_pp = self.getRAdata(eht, 'pp')

        t_ddei = self.getRAdata(eht, 'ddei')
        t_ddss = self.getRAdata(eht, 'ddss')
        t_ddtt = self.getRAdata(eht, 'ddtt')

        t_ddux = self.getRAdata(eht, 'ddux')
        t_dduy = self.getRAdata(eht, 'dduy')
        t_dduz = self.getRAdata(eht, 'dduz')

        t_dduxux = self.getRAdata(eht, 'dduxux')
        t_dduyuy = self.getRAdata(eht, 'dduyuy')
        t_dduzuz = self.getRAdata(eht, 'dduzuz')

        t_uxux = self.getRAdata(eht, 'uxux')
        t_uyuy = self.getRAdata(eht, 'uyuy')
        t_uzuz = self.getRAdata(eht, 'uzuz')

        t_fht_ek = 0.5 * (t_dduxux + t_dduyuy + t_dduzuz) / t_dd
        t_fht_ei = t_ddei / t_dd
        t_fht_et = t_fht_ek + t_fht_ei
        t_fht_ss = t_ddss / t_dd

        t_fht_ux = t_ddux / t_dd
        t_fht_uy = t_dduy / t_dd
        t_fht_uz = t_dduz / t_dd

        t_fht_ui_fht_ui = t_fht_ux * t_fht_ux + t_fht_uy * t_fht_uy + t_fht_uz * t_fht_uz

        t_fht_tt = t_ddtt/t_dd

        # t_mm    = self.getRAdata(eht,'mm'))
        # minus_dt_mm = -self.dt(t_mm,xzn0,t_timec,intc)
        # fht_ux = minus_dt_mm/(4.*np.pi*(xzn0**2.)*dd)

        # construct equation-specific mean fields			
        # fht_ek = 0.5*(dduxux + dduyuy + dduzuz)/dd
        fht_ek = ddek / dd
        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd
        fht_ei = ddei / dd
        fht_et = fht_ek + fht_ei
        fht_enuc = (ddenuc1 + ddenuc2) / dd
        fht_eiux = ddeiux/dd

        fei = ddeiux - ddux * ddei / dd
        fekx = ddekux - fht_ux * fht_ek
        fpx = ppux - pp * ux
        fekx = ddekux - fht_ux * fht_ek

        fht_ui_fht_ui = fht_ux * fht_ux + fht_uy * fht_uy + fht_uz * fht_uz

        if self.ig == 1:  # Kippenhahn and Weigert, page 38
            alpha = 1.
            delta = 1.
            phi = 1.
        elif self.ig == 2:
            alpha = 1. / chid
            delta = -chit / chid
            phi = chid / chim

        fht_rxx = dduxux - ddux * ddux / dd
        fdil = (uxdivu - ux * divu)

        gg = -gg

        if self.ig == 1:
            surface = (yzn0[-1] - yzn0[0]) * (zzn0[-1] - zzn0[0])
        elif self.ig == 2:
            # sphere surface
            surface = +4. * np.pi * (xzn0 ** 2.)
        else:
            print("ERROR(Properties.py): " + self.errorGeometry(self.ig))
            sys.exit()

        ####################################
        # STANDARD LUMINOSITY EQUATION EXACT
        ####################################

        self.minus_cp_rho_dTdt = -cp*(self.dt(t_ddtt, xzn0, t_timec, intc) + self.Div(ddttux,xzn0) - (dduxttx + dduytty + dduzttz))

        self.plus_delta_dPdt = +delta * self.dt(t_pp, xzn0, t_timec, intc)

        self.minus_dd_div_eiui = -(self.Div(ddeiux, xzn0) - (eiuxddx + eiuyddy + eiuzddz))
        #self.minus_dd_div_eiui = -(self.Div(ddeiux, xzn0))

        self.plus_tke_diss = +tke_diss

        self.minus_resLumExactEquation = -(self.minus_cp_rho_dTdt+self.plus_delta_dPdt+self.minus_dd_div_eiui+self.plus_tke_diss)

        ########################################
        # END STANDARD LUMINOSITY EQUATION EXACT 
        ######################################## 

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fht_et = fht_ei + fht_ek
        self.nx = nx
        self.bconv = bconv
        self.tconv = tconv
        self.fext = fext

    def plot_et(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mean total energy stratification in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fht_et

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'total energy')
        plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{\varepsilon}_t$')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\widetilde{\varepsilon}_t$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\widetilde{\varepsilon}_t$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_et.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_et.eps')


    def plot_luminosity_equation_exact(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot luminosity equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        rhs0 = self.minus_cp_rho_dTdt
        rhs1 = self.plus_delta_dPdt
        rhs2 = self.minus_dd_div_eiui
        rhs3 = self.plus_tke_diss

        res = self.minus_resLumExactEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        self.bconv = 4.e8
        self.tconv = 1.2e9

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        # plot DATA 
        plt.title("standard luminosity equation exact")
        if self.ig == 1:
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00', label=r"$-c_P \overline{\rho \partial_t T}$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='y',label = r"$+\delta \overline{\partial_t P}$")
            plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='r',label = r"$-\overline{\rho \nabla \cdot \epsilon_I {\bf u}}$")
            plt.plot(grd1[xlimitrange], rhs3[xlimitrange], color='g',label = r"$+\varepsilon_K$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N$")

            zeros = np.zeros(self.nx)
            plt.plot(grd1, zeros, color='k', linewidth=0.6, label="zero")
        elif self.ig == 2:
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00', label=r"$-c_P \rho dT/dt$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='y',label = r"$+\delta dP/dt$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N$")

            zeros = np.zeros(self.nx)
            plt.plot(grd1, zeros, color='k', linewidth=0.6, label="zero")

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg g$^{-1}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg g$^{-1}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol = 2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'standard_luminosity_exact_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'standard_luminosity_exact_eq.eps')
