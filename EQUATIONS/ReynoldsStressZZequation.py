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

class ReynoldsStressZZequation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, minus_kolmrate, data_prefix):
        super(ReynoldsStressZZequation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        nx = self.getRAdata(eht, 'nx')
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf 		

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        dduycoty = self.getRAdata(eht, 'dduycoty')[intc]
        dduzcoty = self.getRAdata(eht, 'dduzcoty')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        dduyuycoty = self.getRAdata(eht, 'dduyuycoty')[intc]
        dduzuzcoty = self.getRAdata(eht, 'dduzuzcoty')[intc]
        dduzuycoty = self.getRAdata(eht, 'dduzuycoty')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduxuy = self.getRAdata(eht, 'dduxuy')[intc]
        dduxuz = self.getRAdata(eht, 'dduxuz')[intc]

        dduxuxux = self.getRAdata(eht, 'dduxuxux')[intc]
        dduxuyuy = self.getRAdata(eht, 'dduxuyuy')[intc]
        dduxuzuz = self.getRAdata(eht, 'dduxuzuz')[intc]

        dduzuzuycoty = self.getRAdata(eht, 'dduzuzuycoty')[intc]

        ddekux = self.getRAdata(eht, 'ddekux')[intc]
        ddek = self.getRAdata(eht, 'ddek')[intc]

        ppdivux = self.getRAdata(eht, 'ppdivux')[intc]
        ppdivuy = self.getRAdata(eht, 'ppdivuy')[intc]
        ppdivuz = self.getRAdata(eht, 'ppdivuz')[intc]

        divux = self.getRAdata(eht, 'divux')[intc]
        divuy = self.getRAdata(eht, 'divuy')[intc]
        divuz = self.getRAdata(eht, 'divuz')[intc]

        ppux = self.getRAdata(eht, 'ppux')[intc]

        #############################
        # REYNOLDS STRESS ZZ EQUATION 
        #############################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        t_ddux = self.getRAdata(eht, 'ddux')
        t_dduy = self.getRAdata(eht, 'dduy')
        t_dduz = self.getRAdata(eht, 'dduz')

        t_dduxux = self.getRAdata(eht, 'dduxux')
        t_dduyuy = self.getRAdata(eht, 'dduyuy')
        t_dduzuz = self.getRAdata(eht, 'dduzuz')

        t_uxffuxff = t_dduxux / t_dd - t_ddux * t_ddux / (t_dd * t_dd)
        t_uyffuyff = t_dduyuy / t_dd - t_dduy * t_dduy / (t_dd * t_dd)
        t_uzffuzff = t_dduzuz / t_dd - t_dduz * t_dduz / (t_dd * t_dd)

        t_rzz = t_dd * t_uzffuzff

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd

        uzffuzff = (dduzuz / dd - dduz * dduz / (dd * dd))

        rzz = dd * uzffuzff

        fkp = dduxuzuz - 2. * fht_uz * dduxuz - fht_ux * dduzuz + 2. * dd * fht_uz * fht_uz * fht_ux

        # LHS -dq/dt 			
        self.minus_dt_rzz = -self.dt(t_rzz, xzn0, t_timec, intc)

        # LHS -div ux rzz
        self.minus_div_fht_ux_rzz = -self.Div(fht_ux * rzz, xzn0)

        # -div 2 fkp 
        self.minus_div_two_fkp = -self.Div(2. * fkp, xzn0)

        # +2 pressure pp dilatation
        self.plus_two_ppf_divuzff = 2. * (ppdivuz - pp * divuz)

        # -2 R grad u
        rzx = dduxuz - ddux * dduz / dd
        self.minus_two_rzx_grad_fht_uz = -2. * rzx * self.Grad(fht_uz, xzn0)

        # +GrrP
        GrrP = \
            2. * (dduxuzuz - 2. * dduz * dduxuz / dd - fht_ux * dduzuz + 2. * fht_uz * fht_uz * fht_ux * dd) / xzn0 - \
            2. * (dduzuzuycoty - 2. * dduycoty * dduyuycoty / dd - 2. * dduycoty * dduzuzcoty / dd +
                  2. * dduzcoty * dduzcoty * dduycoty / (dd * dd)) / xzn0

        uzff_GpM = (dduxuzuz - fht_uz * dduxuz) / xzn0 + (dduzuzuycoty - fht_uy * dduzuycoty) / xzn0

        # +2 Gkp
        if ig == 1:
            self.plus_two_Gkp = np.zeros(nx)
        elif ig == 2:
            self.plus_two_Gkp = 2. * ((1. / 2.) * GrrP - uzff_GpM)
        else:
            print("ERROR(ReynoldsStressZZequation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # -res		
        self.minus_resRzzEquation = -(self.minus_dt_rzz + self.minus_div_fht_ux_rzz + self.minus_div_two_fkp +
                                      self.plus_two_ppf_divuzff +
                                      self.minus_two_rzx_grad_fht_uz + self.plus_two_Gkp)

        # - kolm_rate 1/3 u'3/lc
        self.minus_onethrd_kolmrate = (1. / 3.) * minus_kolmrate

        #################################
        # END REYNOLDS STRESS ZZ EQUATION 
        #################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dd = dd
        self.rzz = rzz

    def plot_rzz(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot Reynolds stress zz in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(ReynoldsStressZZequation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot 		
        plt1 = self.rzz

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('rzz')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r"$\overline{\rho} \widetilde{u''_z u''_z}$")
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r"$\overline{\rho} \widetilde{u''_\phi u''_\phi}$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$R_{zz}$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$R_{\phi \phi}$ (erg g$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_rzz.png')

    def plot_rzz_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot Reynolds stress rzz equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ReynoldsStressZZequation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_rzz
        lhs1 = self.minus_div_fht_ux_rzz

        rhs0 = self.plus_two_ppf_divuzff
        rhs1 = self.minus_div_two_fkp
        rhs2 = self.minus_two_rzx_grad_fht_uz
        rhs3 = self.plus_two_Gkp

        res = self.minus_resRzzEquation

        rhs4 = self.minus_onethrd_kolmrate*self.dd

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # plot DATA
        plt.title('reynolds stress zz equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t R_{zz}$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_x (\widetilde{u}_x R_{zz})$")

            plt.plot(grd1, rhs0, color='c', label=r"$+2 \overline{P' \nabla u''_z }$")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-\nabla_x 2 f_k^z$")
            plt.plot(grd1, rhs2, color='b', label=r"$-\widetilde{R}_{zx}\partial_x \widetilde{u}_z$")
            # plt.plot(grd1, rhs3, color='y', label=r"$2 \mathcal{G}_k^p$")
            plt.plot(grd1, rhs4, color='k',linewidth=0.7,label = r"$-\overline{\rho} 1/3 u^{'3}_{rms}/l_c$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_{Rzz}$")
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t R_{\phi \phi}$')
            plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\widetilde{u}_r R_{\phi \phi})$")

            plt.plot(grd1, rhs0, color='c', label=r"$+2 \overline{P' \nabla u''_\phi }$")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-\nabla_r 2 f_k^p$")
            plt.plot(grd1, rhs2, color='b', label=r"$-\widetilde{R}_{\phi r}\partial_r \widetilde{u}_\phi$")
            plt.plot(grd1, rhs3, color='y', label=r"$2 \mathcal{G}_k^p$")
            plt.plot(grd1,rhs4,color='k',linewidth=0.7,label = r"$-\overline{\rho} 1/3 u^{'3}_{rms}/l_c$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_{Rpp}$")

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.ylabel(setylabel)
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.ylabel(setylabel)
            plt.xlabel(setxlabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'rzz_eq.png')

    # def tke_dissipation(self):
    #    return self.minus_resTkeEquation

    # def tke(self):
    #    return self.tke
