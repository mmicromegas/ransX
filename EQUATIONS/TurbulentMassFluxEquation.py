import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentMassFluxEquation(calc.CALCULUS, al.ALIMIT, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(TurbulentMassFluxEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0'))
        nx = np.asarray(eht.item().get('nx'))

        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])
        pp = np.asarray(eht.item().get('pp')[intc])
        gg = np.asarray(eht.item().get('gg')[intc])
        sv = np.asarray(eht.item().get('sv')[intc])
        mm = np.asarray(eht.item().get('mm')[intc])

        uxux = np.asarray(eht.item().get('uxux')[intc])
        ddux = np.asarray(eht.item().get('ddux')[intc])
        divu = np.asarray(eht.item().get('divu')[intc])
        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])

        uyuy = np.asarray(eht.item().get('uyuy')[intc])
        uzuz = np.asarray(eht.item().get('uzuz')[intc])

        svdduyuy = np.asarray(eht.item().get('svdduyuy')[intc])
        svdduzuz = np.asarray(eht.item().get('svdduzuz')[intc])

        svdddduyuy = np.asarray(eht.item().get('svdddduyuy')[intc])
        svdddduzuz = np.asarray(eht.item().get('svdddduzuz')[intc])

        svgradxpp = np.asarray(eht.item().get('svgradxpp')[intc])

        gamma1 = np.asarray(eht.item().get('gamma1')[intc])

        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))
        t_dd = np.asarray(eht.item().get('dd'))
        t_ux = np.asarray(eht.item().get('ux'))
        t_ddux = np.asarray(eht.item().get('ddux'))
        t_mm = np.asarray(eht.item().get('mm'))

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        rxx = dduxux - ddux * ddux / dd

        eht_ddf_uxf_uxf = dduxux - 2. * ux * ddux - dd * uxux + 2. * dd * ux * ux
        eht_b = 1. - sv * dd

        # a is turbulent mass flux
        eht_a = ux - ddux / dd

        ############################## 		
        # TURBULENT MASS FLUX EQUATION
        ##############################

        # time-series of turbulent mass flux 
        t_a = t_ux - t_ddux / t_dd

        # LHS -dq/dt 		
        self.minus_dt_eht_dd_a = -self.dt(t_dd * t_a, xzn0, t_timec, intc)

        # LHS -div eht_dd fht_ux a
        self.minus_div_eht_dd_fht_ux_eht_a = -self.Div(dd * fht_ux * eht_a, xzn0)

        # RHS minus_ddf_uxf_uxf_o_dd_gradx_eht_dd
        self.minus_ddf_uxf_uxf_o_dd_gradx_eht_dd = -(eht_ddf_uxf_uxf / dd) * self.Grad(dd, xzn0)

        # RHS +rxx_o_dd_gradx_eht_dd
        self.plus_rxx_o_dd_gradx_eht_dd = +(rxx / dd) * self.Grad(dd, xzn0)

        # RHS -eht_dd_div_a_a
        self.minus_eht_dd_div_a_a = -dd * self.Div(eht_a * eht_a, xzn0)

        # RHS +div_eht_ddf_uxf_uxf
        self.plus_div_eht_ddf_uxf_uxf = self.Div(eht_ddf_uxf_uxf, xzn0)

        ##########################

        # RHS +div rxx
        # self.plus_div_rxx = +self.Div(rxx,xzn0)

        # RHS -eht_dd div uxf uxf
        # self.minus_eht_dd_div_uxf_uxf = -dd*self.Div(uxux-ux*ux,xzn0)

        ########################## 

        # RHS -eht_dd_eht_a_div_eht_ux
        self.minus_eht_dd_eht_a_div_eht_ux = -dd * eht_a * self.Div(ux, xzn0)

        # RHS +eht_dd_eht_uxf_dff
        self.plus_eht_dd_eht_uxf_dff = +dd * (uxdivu - ux * divu)

        # RHS -eht_b_gradx_pp
        self.minus_eht_b_gradx_pp = -eht_b * self.Grad(pp, xzn0)

        # RHS +eht_ddf_sv_gradx_ppf = -eht_dd*svgradxpp
        # self.plus_eht_ddf_sv_gradx_ppf = +dd*(svgradxpp-sv*self.Grad(pp,xzn0))
        self.plus_eht_ddf_sv_gradx_ppf = np.zeros(nx)

        # RHS +Ga
        self.plus_Ga = -dduyuy / xzn0 - dduzuz / xzn0 + dd * (uyuy / xzn0 + uzuz / xzn0)

        # RHS minus_resAequation
        self.minus_resAequation = -(self.minus_dt_eht_dd_a + self.minus_div_eht_dd_fht_ux_eht_a + \
                                    self.minus_ddf_uxf_uxf_o_dd_gradx_eht_dd + self.plus_rxx_o_dd_gradx_eht_dd + \
                                    self.minus_eht_dd_div_a_a + self.plus_div_eht_ddf_uxf_uxf + self.minus_eht_dd_eht_a_div_eht_ux + \
                                    self.plus_eht_dd_eht_uxf_dff + self.minus_eht_b_gradx_pp + self.plus_eht_ddf_sv_gradx_ppf + \
                                    self.plus_Ga)

        # self.minus_resAequation = -(self.plus_div_rxx + self.minus_eht_dd_div_uxf_uxf + self.minus_eht_dd_eht_a_div_eht_ux + \
        #  self.plus_eht_dd_eht_uxf_dff + self.minus_eht_b_gradx_pp + self.plus_eht_ddf_sv_gradx_ppf + \
        #  self.plus_Ga)

        ################################## 		
        # END TURBULENT MASS FLUX EQUATION
        ##################################

        self.minus_dt_mm = -self.dt(t_mm, xzn0, t_timec, intc)
        self.plus_dt_mm = -4. * np.pi * (xzn0 ** 2.) * dd * fht_ux
        self.plus_grad_mm = +self.Grad(mm, xzn0)

        self.eht_a_model1 = mm * gg / (gamma1 * pp * fht_ux * 4. * np.pi * (xzn0 ** 2.))
        self.eht_a_model2 = ux * fht_ux / (ux + fht_ux)
        self.eht_a_model3 = (gamma1 * pp * 4. * np.pi * (xzn0 ** 2.)) / (2. * mm * gg) - ux
        self.eht_a_model4 = (mm * gg * ux / (gamma1 * pp * 4. * np.pi * (xzn0 ** 2.))) - 2. * fht_ux
        self.fht_ux_model = gamma1 * pp * fht_ux * 8. * np.pi * (xzn0 ** 2.) / (
                    ((4. / 3) * np.pi * (xzn0 ** 3.) * dd) * gg)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.eht_a = eht_a
        self.ux = ux
        self.fht_ux = fht_ux

    def plot_a(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mean turbulent mass flux in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.eht_a
        plt2 = self.eht_a_model1
        plt3 = self.eht_a_model2
        plt4 = self.ux
        plt5 = self.fht_ux
        plt6 = self.eht_a_model3
        plt7 = self.eht_a_model4
        plt8 = self.fht_ux_model

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'turbulent mass flux')
        plt.plot(grd1, plt1, color='brown', label=r"$a$")
        # plt.plot(grd1,plt2,color='r',label='model1')
        # plt.plot(grd1,plt3,color='g',label='model2')
        plt.plot(grd1, plt4, color='pink', label=r'$\overline{u}_r$')
        plt.plot(grd1, plt5, color='m', label=r'$\widetilde{u}_r$')
        # plt.plot(grd1,plt6,color='b',label=r'model3')
        plt.plot(grd1, plt7, color='b', label=r'model4')
        plt.plot(grd1, plt8, color='r', linestyle='--', label=r'model for fht ux')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho}$ $\overline{u''_x}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        # to_plot = [plt1,plt2,plt3,plt4,plt5,plt6]
        # self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)

        # self.minus_dt_mm = -self.dt(t_mm,xzn0,t_timec,intc)
        # self.plus_dt_mm = -4.*np.pi*(self.xzn0**2.)*dd*fht_ux
        # self.plus_grad_mm = +self.Grad(mm,xzn0)

        # plot DATA 
        plt.plot(grd1, 1. / self.ux, color='brown', label=r"$1/\overline{u}_r$")
        plt.plot(grd1, 1. / self.fht_ux, color='r', label=r"$1/\widetilde{u}_r$")
        # plt.plot(grd1,+self.eht_a_model1,color='g',label='model1')
        # plt.plot(grd1,(1./(self.ux)+(1./(self.fht_ux))),linestyle='--',color='b',label='xx')
        # plt.plot(grd1,1./(self.eht_a+self.fht_ux),color='pink',label='1/a')
        plt.plot(grd1, -self.plus_grad_mm / self.plus_dt_mm, color='k', linestyle='--', label='drMM/dtMM')

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_a.png')

    def plot_a_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """ turbulent mass flux equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_a
        lhs1 = self.minus_div_eht_dd_fht_ux_eht_a

        rhs0 = self.minus_ddf_uxf_uxf_o_dd_gradx_eht_dd
        rhs1 = self.plus_rxx_o_dd_gradx_eht_dd
        rhs2 = self.minus_eht_dd_div_a_a
        rhs3 = self.plus_div_eht_ddf_uxf_uxf
        # rhs0 = self.plus_div_rxx
        # rhs1 = self.minus_eht_dd_div_uxf_uxf
        rhs4 = self.minus_eht_dd_eht_a_div_eht_ux
        rhs5 = self.plus_eht_dd_eht_uxf_dff
        rhs6 = self.minus_eht_b_gradx_pp
        rhs7 = self.plus_eht_ddf_sv_gradx_ppf
        rhs8 = self.plus_Ga

        res = self.minus_resAequation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        # to_plot = [lhs0,lhs1,rhs0,rhs1,rhs4,rhs5,rhs6,rhs7,rhs8,res]
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, rhs8, res]

        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('turbulent mass flux equation')
        plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-\partial_t (\overline{\rho} \overline{u''_r})$')
        plt.plot(grd1, lhs1, color='k', label=r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \overline{u''_r})$")

        plt.plot(grd1, rhs0, color='r',
                 label=r"$-(\overline{\rho' u'_r u'_r} / \overline{\rho} \partial_r \overline{\rho})$")
        plt.plot(grd1, rhs1, color='#802A2A', label=r"$+\widetilde{R}_{rr}/\overline{\rho}\partial_r \overline{\rho} $")
        plt.plot(grd1, rhs2, color='c', label=r"$-\overline{\rho} \nabla_r (\overline{u''_r} \ \overline{u''_r}) $")
        plt.plot(grd1, rhs3, color='m', label=r"$+\nabla_r \overline{\rho' u'_r u'_r}$")

        # plt.plot(grd1,rhs0,color='r',label = r"$+\nabla \widetilde{R}_{xx}$")
        # plt.plot(grd1,rhs1,color='c',label = r"$-\overline{\rho} \nabla_r \overline{u'_r u'_r}$")

        plt.plot(grd1, rhs4, color='g', label=r"$-\overline{\rho} \overline{u''_r} \nabla_r \overline{u_r}$")
        plt.plot(grd1, rhs5, color='y', label=r"$+\overline{\rho} \overline{u'_r d''} $")
        plt.plot(grd1, rhs6, color='b', label=r"$-b \partial_r \overline{P}$")
        plt.plot(grd1, rhs7, color='orange', label=r"$+\overline{\rho' v \partial_r P'} \sim 0$")
        plt.plot(grd1, rhs8, color='skyblue', label=r"$+Ga$")

        plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_a$")

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-2}$ s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'a_eq.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'a_eq.eps')
