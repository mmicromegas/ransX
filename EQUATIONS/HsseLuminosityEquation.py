import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseLuminosityEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, fext, intc, tke_diss, bconv, tconv, data_prefix):
        super(HsseLuminosityEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

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

        fht_lum = surface * dd * fht_ux * fht_et
        # fht_lum_for_exact = surface * (ddeiux + ddekux)

        # LHS -grad fht_lum 			
        self.minus_gradx_fht_lum = -self.Grad(fht_lum, xzn0)

        # RHS +surface dd fht_enuc
        self.plus_surface_dd_fht_enuc = +surface * dd * fht_enuc

        # RHS -surface div fei
        self.minus_surface_div_fei = -surface * self.Div(fei, xzn0)

        # RHS -surface div ftt (not included) heat flux
        self.minus_surface_div_fth = -surface * np.zeros(nx)

        # RHS -surface div fekx
        self.minus_surface_div_fekx = -surface * self.Div(fekx, xzn0)

        # RHS -surface div fpx
        self.minus_surface_div_fpx = -surface * self.Div(fpx, xzn0)

        # RHS -surface P d = - surface eht_pp Div eht_ux
        self.minus_surface_pp_div_ux = -surface * pp * self.Div(ux, xzn0)

        # -R grad u

        rxx = dduxux - ddux * ddux / dd
        rxy = dduxuy - ddux * dduy / dd
        rxz = dduxuz - ddux * dduz / dd

        self.minus_surface_r_grad_u = -surface * (rxx * self.Grad(ddux / dd, xzn0) +
                                                  rxy * self.Grad(dduy / dd, xzn0) +
                                                  rxz * self.Grad(dduz / dd, xzn0))

        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = -ux + fht_ux

        # +buoyancy work
        self.plus_surface_wb = +surface * self.plus_ax * self.Grad(pp, xzn0)

        # +dd Dt fht_ui_fht_ui_o_two
        t_fht_ux = t_ddux / t_dd
        t_fht_uy = t_dduy / t_dd
        t_fht_uz = t_dduz / t_dd

        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd

        self.plus_surface_dd_Dt_fht_ui_fht_ui_o_two = \
            +surface * self.dt(t_dd * (t_fht_ux ** 2. + t_fht_uy ** 2. + t_fht_uz ** 2.), xzn0, t_timec, intc) - \
            self.Div(dd * fht_ux * (fht_ux ** 2. + fht_uy ** 2. + fht_uz ** 2.), xzn0) / 2.

        # RHS -surface dd dt et
        self.minus_surface_dd_dt_et = -surface * dd * self.dt(t_fht_et, xzn0, t_timec, intc)

        # RHS +fht_et gradx +surface dd fht_ux
        self.plus_fht_et_gradx_surface_dd_fht_ux = +fht_et * self.Grad(surface * dd * fht_ux, xzn0)

        # -res		
        self.minus_resLumEquation = -(self.minus_gradx_fht_lum + self.plus_surface_dd_fht_enuc +
                                      self.minus_surface_div_fei + self.minus_surface_div_fth +
                                      self.minus_surface_div_fekx + self.minus_surface_div_fpx + self.minus_surface_pp_div_ux +
                                      self.minus_surface_r_grad_u + self.plus_surface_wb +
                                      self.plus_surface_dd_Dt_fht_ui_fht_ui_o_two + self.minus_surface_dd_dt_et +
                                      self.plus_fht_et_gradx_surface_dd_fht_ux)

        #############################
        # END HSS LUMINOSITY EQUATION 
        #############################

        ####################################
        # STANDARD LUMINOSITY EQUATION EXACT
        ####################################

        # RHS -surface ddetux
        fht_lum_for_exact = surface * ddetux
        self.minus_gradx_fht_lum_for_exact = -self.Grad(fht_lum_for_exact, xzn0)

        # RHS +surface dd fht_enuc
        self.plus_surface_dd_fht_enuc_for_exact = +surface * dd * fht_enuc

        # RHS +surface dd tkediss
        self.plus_surface_tke_diss_for_exact = +surface * tke_diss

        # RHS -surface dd cp dt tt
        # self.minus_surface_dd_cp_dt_tt_for_exact = -surface * dd * cp * self.dt(t_tt, xzn0, t_timec, intc)
        self.minus_surface_dd_cp_dt_fht_tt_for_exact = -surface * dd * cp * self.dt(t_fht_tt, xzn0, t_timec, intc)

        # RHS -surface delta dt p
        self.minus_surface_delta_dt_pp_for_exact = -surface * delta * self.dt(t_pp, xzn0, t_timec, intc)

        # RHS turb.
        self.minus_surface_div_dd_fht_ei_fht_ux_for_exact = -surface * self.Div(dd * fht_ei * fht_ux, xzn0)
        self.minus_surface_div_fei_for_exact = -surface * self.Div(fei, xzn0)

        #self.plus_surface_div_dd_fht_ei_fht_ux_for_exact = +surface * self.Div(dd * fht_ei * fht_ux, xzn0)
        self.plus_surface_div_dd_fht_ei_fht_ux_for_exact = np.zeros(nx)

        #self.plus_surface_div_fei_for_exact = +surface * self.Div(fei, xzn0)
        self.plus_surface_div_fei_for_exact = np.zeros(nx)

        self.minus_resLumExactEquation = -(self.minus_gradx_fht_lum_for_exact + self.plus_surface_dd_fht_enuc_for_exact +
                                           self.plus_surface_tke_diss_for_exact + self.minus_surface_dd_cp_dt_fht_tt_for_exact +
                                           self.minus_surface_delta_dt_pp_for_exact + self.plus_surface_div_fei_for_exact +
                                           self.plus_surface_div_dd_fht_ei_fht_ux_for_exact)


        self.plus_eiux_gradx_dd = +surface * eiux * self.Grad(dd, xzn0)

        ########################################
        # END STANDARD LUMINOSITY EQUATION EXACT 
        ######################################## 

        ######################################
        # STANDARD LUMINOSITY EQUATION EXACT 2 
        ###################################### 

        self.minus_dd_tt_dt_fht_ss = -surface * dd * tt * self.dt(t_fht_ss, xzn0, t_timec, intc)
        self.minus_resLumExactEquation2 = -(
                self.minus_gradx_fht_lum_for_exact + self.plus_surface_dd_fht_enuc + self.minus_dd_tt_dt_fht_ss)

        ########################################
        # END STANDARD LUMINOSITY EQUATION EXACT 2 
        ######################################## 

        #################################
        # ALTERNATIVE LUMINOSITY EQUATION 
        #################################

        self.minus_surface_dd_fht_ux_pp_fdil_o_fht_rxx = -surface * dd * fht_ux * pp * fdil / fht_rxx

        self.plus_fht_et_gradx_surface_dd_fht_ux = +fht_et * self.Grad(surface * dd * fht_ux, xzn0)

        self.minus_resAlternativeLumEq = -(
                self.minus_gradx_fht_lum + self.minus_surface_dd_fht_ux_pp_fdil_o_fht_rxx + self.plus_fht_et_gradx_surface_dd_fht_ux)

        #####################################
        # END ALTERNATIVE LUMINOSITY EQUATION 
        #####################################

        ############################################
        # ALTERNATIVE LUMINOSITY EQUATION SIMPLIFIED
        ############################################

        self.minus_surface_dd_fht_ux_dd_o_gamma1 = -surface * fht_ux * dd * gg / gamma1

        self.plus_fht_et_gradx_surface_dd_fht_ux = +fht_et * self.Grad(surface * dd * fht_ux, xzn0)

        self.minus_resAlternativeLumEqSimplified = -(
                self.minus_gradx_fht_lum + self.minus_surface_dd_fht_ux_dd_o_gamma1 + self.plus_fht_et_gradx_surface_dd_fht_ux)

        ################################################
        # END ALTERNATIVE LUMINOSITY EQUATION SIMPLIFIED 
        ################################################

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

    def plot_luminosity_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot luminosity equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_fht_lum

        rhs0 = self.plus_surface_dd_fht_enuc
        rhs2 = self.minus_surface_div_fei
        rhs3 = self.minus_surface_div_fth
        rhs4 = self.minus_surface_div_fekx
        rhs5 = self.minus_surface_div_fpx
        rhs6 = self.minus_surface_pp_div_ux
        rhs7 = self.minus_surface_r_grad_u
        rhs8 = self.plus_surface_wb
        rhs9 = self.plus_surface_dd_Dt_fht_ui_fht_ui_o_two
        rhs10 = self.minus_surface_dd_dt_et
        rhs11 = self.plus_fht_et_gradx_surface_dd_fht_ux

        res = self.minus_resLumEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, rhs9, rhs10, rhs11, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('hsse luminosity equation')
        # plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_r \widetilde{L}$")

        # plt.plot(grd1,rhs0,color='#FF8C00',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{\epsilon}_{nuc}$")
        # plt.plot(grd1,rhs1,color='y',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{\varepsilon}_{k}$")
        # plt.plot(grd1,rhs2,color='g',label = r"$-4 \pi r^2 \nabla_r f_I$")
        # plt.plot(grd1,rhs3,color='gray',label = r"$-4 \pi r^2 \nabla_r f_{th}$ (not incl.)")
        # plt.plot(grd1,rhs4,color='#802A2A',label = r"$-4 \pi r^2  \nabla_r f_{K}$")
        # plt.plot(grd1,rhs5,color='darkmagenta',label = r"$-4 \pi r^2 \nabla_r f_{P}$")
        # plt.plot(grd1,rhs6,color='b',label=r"$-4 \pi r^2 \overline{P} \ \overline{d}$")
        # plt.plot(grd1,rhs7,color='pink',label=r"$-4 \pi r^2 \widetilde{R}_{ir} \partial_r \widetilde{u}_i$")
        # plt.plot(grd1,rhs8,color='r',label=r"$+4 \pi r^2 W_b$")
        # plt.plot(grd1,rhs9,color='m',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{D}_t \widetilde{u}_i \widetilde{u}_i /2$")
        # plt.plot(grd1,rhs10,color='chartreuse',label = r"$-4 \pi r^2 \overline{\rho} \partial_t#\widetilde{\epsilon}_t$")
        # plt.plot(grd1,rhs11,color='olive',label = r"$+\widetilde{\epsilon}_t \partial_r 4 \pi r^2 \overline{\rho} \widetilde{u}_r$")
        # plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N$")

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)


        if self.ig == 1:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_x \widetilde{L}$")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00',
                     label=r"$+surf \  \overline{\rho} \widetilde{\epsilon}_{nuc}$")
            # plt.plot(grd1[xlimitrange],rhs1[xlimitrange],color='y',label = r"$+surf \  \overline{\rho} \widetilde{\varepsilon}_{k}$")
            plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='g', label=r"$-surf \  \nabla_x f_I$")
            plt.plot(grd1[xlimitrange], rhs3[xlimitrange], color='gray', label=r"$-surf \  \nabla_x f_{th}$ (not incl.)")
            plt.plot(grd1[xlimitrange], rhs4[xlimitrange], color='#802A2A', label=r"$-surf \   \nabla_x f_{K}$")
            plt.plot(grd1[xlimitrange], rhs5[xlimitrange], color='darkmagenta', label=r"$-surf \  \nabla_x f_{P}$")
            plt.plot(grd1[xlimitrange], rhs6[xlimitrange], color='b', label=r"$-surf \  \overline{P} \ \overline{d}$")
            plt.plot(grd1[xlimitrange], rhs7[xlimitrange], color='pink',
                  label=r"$-surf \  \widetilde{R}_{ir} \partial_x \widetilde{u}_i$")
            plt.plot(grd1[xlimitrange], rhs8[xlimitrange], color='r', label=r"$+surf \  W_b$")
            plt.plot(grd1[xlimitrange], rhs9[xlimitrange], color='m',
                  label=r"$+surf \  \overline{\rho} \widetilde{D}_t \widetilde{u}_i \widetilde{u}_i /2$")
            plt.plot(grd1[xlimitrange], rhs10[xlimitrange], color='chartreuse',
                  label=r"$-surf \  \overline{\rho} \partial_t \widetilde{\epsilon}_t$")
            plt.plot(grd1[xlimitrange], rhs11[xlimitrange], color='olive',
                     label=r"$+\widetilde{\epsilon}_t \partial_x surf \  \overline{\rho} \widetilde{u}_x$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label=r"res $\sim N$")

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='#FF8C00', markersize=0.5)
            # plt.plot(grd1[xlimitbottom],rhs1[xlimitbottom],'.',color='y',markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs2[xlimitbottom], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs3[xlimitbottom], '.', color='gray', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs4[xlimitbottom], '.', color='#802A2A', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs5[xlimitbottom], '.', color='darkmagenta', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs6[xlimitbottom], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs7[xlimitbottom], '.', color='pink', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs8[xlimitbottom], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs9[xlimitbottom], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs10[xlimitbottom], '.', color='chartreuse', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs11[xlimitbottom], '.', color='olive', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='#FF8C00', markersize=0.5)
            # plt.plot(grd1[xlimittop],rhs1[xlimittop],'.',color='y',markersize=0.5)
            plt.plot(grd1[xlimittop], rhs2[xlimittop], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs3[xlimittop], '.', color='gray', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs4[xlimittop], '.', color='#802A2A', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs5[xlimittop], '.', color='darkmagenta', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs6[xlimittop], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs7[xlimittop], '.', color='pink', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs8[xlimittop], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs9[xlimittop], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs10[xlimittop], '.', color='chartreuse', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs11[xlimittop], '.', color='olive', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)
        elif self.ig == 2:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_r \widetilde{L}$")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00',
                     label=r"$+4 \pi r^2 \overline{\rho} \widetilde{\epsilon}_{nuc}$")
            # plt.plot(grd1[xlimitrange],rhs1[xlimitrange],color='y',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{\varepsilon}_{k}$")
            plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='g', label=r"$-4 \pi r^2 \nabla_r f_I$")
            plt.plot(grd1[xlimitrange], rhs3[xlimitrange], color='gray', label=r"$-4 \pi r^2 \nabla_r f_{th}$ (not incl.)")
            plt.plot(grd1[xlimitrange], rhs4[xlimitrange], color='#802A2A', label=r"$-4 \pi r^2  \nabla_r f_{K}$")
            plt.plot(grd1[xlimitrange], rhs5[xlimitrange], color='darkmagenta', label=r"$-4 \pi r^2 \nabla_r f_{P}$")
            plt.plot(grd1[xlimitrange], rhs6[xlimitrange], color='b', label=r"$-4 \pi r^2 \overline{P} \ \overline{d}$")
            plt.plot(grd1[xlimitrange], rhs7[xlimitrange], color='pink',
                  label=r"$-4 \pi r^2 \widetilde{R}_{ir} \partial_r \widetilde{u}_i$")
            plt.plot(grd1[xlimitrange], rhs8[xlimitrange], color='r', label=r"$+4 \pi r^2 W_b$")
            plt.plot(grd1[xlimitrange], rhs9[xlimitrange], color='m',
                  label=r"$+4 \pi r^2 \overline{\rho} \widetilde{D}_t \widetilde{u}_i \widetilde{u}_i /2$")
            plt.plot(grd1[xlimitrange], rhs10[xlimitrange], color='chartreuse',
                  label=r"$-4 \pi r^2 \overline{\rho} \partial_t \widetilde{\epsilon}_t$")
            plt.plot(grd1[xlimitrange], rhs11[xlimitrange], color='olive',
                     label=r"$+\widetilde{\epsilon}_t \partial_r 4 \pi r^2 \overline{\rho} \widetilde{u}_r$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label=r"res $\sim N$")

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='#FF8C00', markersize=0.5)
            # plt.plot(grd1[xlimitbottom],rhs1[xlimitbottom],'.',color='y',markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs2[xlimitbottom], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs3[xlimitbottom], '.', color='gray', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs4[xlimitbottom], '.', color='#802A2A', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs5[xlimitbottom], '.', color='darkmagenta', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs6[xlimitbottom], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs7[xlimitbottom], '.', color='pink', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs8[xlimitbottom], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs9[xlimitbottom], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs10[xlimitbottom], '.', color='chartreuse', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs11[xlimitbottom], '.', color='olive', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='#FF8C00', markersize=0.5)
            # plt.plot(grd1[xlimittop],rhs1[xlimittop],'.',color='y',markersize=0.5)
            plt.plot(grd1[xlimittop], rhs2[xlimittop], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs3[xlimittop], '.', color='gray', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs4[xlimittop], '.', color='#802A2A', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs5[xlimittop], '.', color='darkmagenta', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs6[xlimittop], '.', color='b', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs7[xlimittop], '.', color='pink', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs8[xlimittop], '.', color='r', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs9[xlimittop], '.', color='m', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs10[xlimittop], '.', color='chartreuse', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs11[xlimittop], '.', color='olive', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')


        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol = 2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_luminosity_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_luminosity_eq.eps')

    def plot_luminosity_equation_exact(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot luminosity equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_fht_lum_for_exact

        rhs0 = self.plus_surface_dd_fht_enuc_for_exact
        rhs1 = self.plus_surface_tke_diss_for_exact
        rhs2 = self.minus_surface_dd_cp_dt_fht_tt_for_exact
        rhs3 = self.minus_surface_delta_dt_pp_for_exact

        # rhs4 = self.plus_surface_div_fei
        # rhs5 = self.plus_surface_div_dd_fht_ei_fht_ux

        # rhs4 = self.plus_surface_div_fei_for_exact
        # rhs5 = self.plus_surface_div_dd_fht_ei_fht_ux_for_exact

        #rhs6 = self.plus_eiux_gradx_dd

        res = self.minus_resLumExactEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs2, rhs3, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        # plot DATA 
        plt.title("standard luminosity equation exact")
        if self.ig == 1:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_x \overline{L}$")

            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00', label=r"$+surf \  \overline{\rho} \widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='y',label = r"$+surf \  \overline{\rho} \widetilde{\varepsilon}_{k}$")
            plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='brown', label=r"$-surf \  c_P \ \overline{\rho} \partial_t \widetilde{T}$")
            plt.plot(grd1[xlimitrange], rhs3[xlimitrange], color='m', label=r"$-surf \  \delta \partial_t \overline{P}$")

            #plt.plot(grd1[xlimitrange], rhs4[xlimitrange], color='g', label=r"$+surf \  \nabla_x \overline{\rho }\widetilde{\epsilon''_I u''_x}$")
            #plt.plot(grd1[xlimitrange], rhs5[xlimitrange], color='b', label=r"$+surf \  \nabla_x \overline{\rho} \widetilde{\epsilon_I} \widetilde{u_x}$")

            # plt.plot(grd1[xlimitrange], rhs4[xlimitrange]-rhs5[xlimitrange] + rhs1[xlimitrange], color='pink', label=r"$rhs4-rhs5+rhs1$")
            # plt.plot(grd1, rhs4+rhs5, color='yellow', label=r"$rhs4+rhs5$")

            #plt.plot(grd1, rhs6, color='r', label=r"$-surf \  \overline{\epsilon_I u_r} \partial_r \overline{\rho}$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N$")

            zeros = np.zeros(self.nx)
            plt.plot(grd1, zeros, color='k', linewidth=0.6, label="zero")
        elif self.ig == 2:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_r \overline{L}$")

            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00', label=r"$+4 \pi r^2 \overline{\rho} \widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='y',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{\varepsilon}_{k}^{diss}}$")
            plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='brown', label=r"$-4 \pi r^2 c_P \overline{\rho} \partial_t \widetilde{T}$")
            plt.plot(grd1[xlimitrange], rhs3[xlimitrange], color='m', label=r"$-4 \pi r^2 \delta \partial_t \overline{P}$")

            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label=r"res $\sim N$")

            zeros = np.zeros(self.nx)
            # plt.plot(grd1[xlimitrange], zeros[xlimitrange], color='k', linewidth=0.6, label="zero")
            # ,label=r"$0 = -\partial_r l + \rho 4\pi r^2 \epsilon - \rho 4 \pi r^2 c_p \partial_t T + \rho 4 \pi r^2 \delta \partial_t P \\ Kippenhahn Weigert p.22, Eq.4.26$"
            # define and show x/y LABELS

            #plt.plot(grd1[xlimitrange], rhs4[xlimitrange], 'o', color='r', markersize = 2., label=r"$+4 \pi r^2 \nabla_r \overline{\rho }\widetilde{\epsilon''_I u''_r}$")
            #plt.plot(grd1[xlimitrange], rhs5[xlimitrange], color='b', label=r"$+4 \pi r^2 \nabla_r \overline{\rho} \widetilde{\epsilon_I} \widetilde{u_r}$")
            # plt.plot(grd1[xlimitrange], rhs4[xlimitrange]-rhs5[xlimitrange] + rhs1[xlimitrange], color='pink', label=r"$rhs4-rhs5+rhs1$")
            # plt.plot(grd1[xlimitrange], rhs4[xlimitrange]+rhs5[xlimitrange], color='yellow', label=r"$rhs4+rhs5$")

            # plt.plot(grd1, rhs6[xlimitrange], color='r', label=r"$-4 \pi r^2 \overline{\epsilon_I u_r} \partial_r \overline{\rho}$")

            #plt.plot(grd1, zeros, color='g',
            #         label = r"$-4 \pi r^2 (\nabla_r \overline{\rho} \widetilde{\epsilon_I} \widetilde{u_r} + \overline{\epsilon_I u_i \partial_i \rho})$ (not calc.)")

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='#FF6EB4', markersize=0.5)

            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='y', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs2[xlimittop], '.', color='brown', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs3[xlimittop], '.', color='m', markersize=0.5)

            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

            zeros = np.zeros(self.nx)
            #plt.plot(grd1, zeros, color='k', linewidth=0.6)

            #plt.plot(grd1[xlimittop], rhs4[xlimittop], 'o', color='r', markersize = 0.5)

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='#FF6EB4', markersize = 0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='#FF8C00', markersize = 0.5)
            plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='y',markersize = 0.5)
            plt.plot(grd1[xlimitbottom], rhs2[xlimitbottom], '.', color='brown', markersize = 0.5)
            plt.plot(grd1[xlimitbottom], rhs3[xlimitbottom], '.', color='m', markersize = 0.5)

            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            zeros = np.zeros(self.nx)
            #plt.plot(grd1[xlimitbottom], zeros[xlimitbottom], color='k', linewidth=0.6)

            #plt.plot(grd1[xlimitbottom], rhs4[xlimitbottom], 'o', color='r', markersize = 0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
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

    def plot_luminosity_equation_exact2(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot luminosity equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        # lhs0 = self.minus_gradx_fht_lum
        lhs0 = self.minus_gradx_fht_lum_for_exact

        rhs0 = self.plus_surface_dd_fht_enuc
        rhs1 = self.minus_dd_tt_dt_fht_ss

        # rhs4 = self.minus_surface_dd_div_fei_o_dd
        # rhs5 = self.minus_surface_dd_div_fht_ei_fht_ux

        res = self.minus_resLumExactEquation2

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title("standard luminosity equation exact 2")
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_x \overline{L}$")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$+surf \ \overline{\rho} \widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1, rhs1, color='brown', label=r"$-surf \ \overline{\rho} \overline{T} \partial_t \widetilde{s}$")

            # plt.plot(grd1, -(rhs4 + rhs5), color='r', label=r"$-(rhs4+rhs5)$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N$")

            zeros = np.zeros(self.nx)
            plt.plot(grd1, zeros, color='k', linewidth=0.6, label="zero")
            # ,label=r"$0 = -\partial_r l + \rho 4\pi r^2 \epsilon - \rho surf \ c_p \partial_t T + \rho surf \ \delta \partial_t P \\ Kippenhahn Weigert p.22, Eq.4.26$"
            # define and show x/y LABELS
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_r \overline{L}$")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$+4 \pi r^2 \overline{\rho} \widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1, rhs1, color='brown', label=r"$-4 \pi r^2 \overline{\rho} \overline{T} \partial_t \widetilde{s}$")

            # plt.plot(grd1, -(rhs4 + rhs5), color='r', label=r"$-(rhs4+rhs5)$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N$")

            zeros = np.zeros(self.nx)
            plt.plot(grd1, zeros, color='k', linewidth=0.6, label="zero")
            # ,label=r"$0 = -\partial_r l + \rho 4\pi r^2 \epsilon - \rho 4 \pi r^2 c_p \partial_t T + \rho 4 \pi r^2 \delta \partial_t P \\ Kippenhahn Weigert p.22, Eq.4.26$"
            # define and show x/y LABELS

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'standard_luminosity_exact2_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'standard_luminosity_exact2_eq.eps')

    def plot_luminosity_equation_2(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot luminosity equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_fht_lum

        rhs0 = self.minus_surface_dd_fht_ux_pp_fdil_o_fht_rxx
        rhs1 = self.plus_fht_et_gradx_surface_dd_fht_ux

        res = self.minus_resAlternativeLumEq

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title("alternative hsse luminosity equation")
        # plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_r \widetilde{L}$")
        # plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-4 \pi r^2 \ \widetilde{u}_{r} \ \overline{\rho} \ \overline{P} \ \overline{u'_r d''} / \ \widetilde{R}_{rr} $")
        # plt.plot(grd1,rhs1,color='g',label=r"$+\widetilde{\epsilon}_t \partial_r 4 \pi r^2 \overline{\rho} \ \widetilde{u}_{r}$")
        # plt.plot(grd1,res,color='k',linestyle='--',label=r"res")

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        if self.ig == 1:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_x \widetilde{L}$")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00',
                     label=r"$-surf \ \ \widetilde{u}_{x} \ \overline{\rho} \ \overline{P} \ \overline{u'_x d''} / \ \widetilde{R}_{xx} $")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='g',
                     label=r"$+\widetilde{\epsilon}_t \partial_x surf \ \overline{\rho} \ \widetilde{u}_{x}$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label=r"res")

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)
        elif self.ig == 2:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_r \widetilde{L}$")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00',
                     label=r"$-4 \pi r^2 \ \widetilde{u}_{r} \ \overline{\rho} \ \overline{P} \ \overline{u'_r d''} / \ \widetilde{R}_{rr} $")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='g',
                     label=r"$+\widetilde{\epsilon}_t \partial_r 4 \pi r^2 \overline{\rho} \ \widetilde{u}_{r}$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label=r"res")

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_luminosity_eq_alternative.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_luminosity_eq_alternative.eps')

    def plot_luminosity_equation_3(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot luminosity equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_fht_lum

        rhs0 = self.minus_surface_dd_fht_ux_dd_o_gamma1
        rhs1 = self.plus_fht_et_gradx_surface_dd_fht_ux

        res = self.minus_resAlternativeLumEqSimplified

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title("alternative hsse luminosity eq simp")
        # plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_r \widetilde{L}$")
        # plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-4 \pi r^2 \ \widetilde{u}_{r} \ \overline{\rho} \ \overline{g}_r / \Gamma_1$")
        # plt.plot(grd1,rhs1,color='g',label=r"$+\widetilde{\epsilon}_t \partial_r 4 \pi r^2 \overline{\rho} \ \widetilde{u}_{r}$")
        # plt.plot(grd1,res,color='k',linestyle='--',label=r"res")

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        if self.ig == 1:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_x \widetilde{L}$")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00',
                     label=r"$-surf \ \widetilde{u}_{x} \ \overline{\rho} \ \overline{g}_x / \Gamma_1$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='g',
                     label=r"$+\widetilde{\epsilon}_t \partial_x surf \overline{\rho} \ \widetilde{u}_{x}$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label=r"res")

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)
        elif self.ig == 2:
            plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='#FF6EB4', label=r"$-\partial_r \widetilde{L}$")
            plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='#FF8C00',
                     label=r"$-4 \pi r^2 \ \widetilde{u}_{r} \ \overline{\rho} \ \overline{g}_r / \Gamma_1$")
            plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='g',
                     label=r"$+\widetilde{\epsilon}_t \partial_r 4 \pi r^2 \overline{\rho} \ \widetilde{u}_{r}$")
            plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label=r"res")

            plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

            plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='#FF6EB4', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='#FF8C00', markersize=0.5)
            plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='g', markersize=0.5)
            plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg s$^{-1}$ cm$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_luminosity_eq_alternative_simplified.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_luminosity_eq_alternative_simplified.eps')
