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

class TurbulentKineticEnergyEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, minus_kolmrate, data_prefix):
        super(TurbulentKineticEnergyEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		

        dd = self.getRAdata(eht, 'dd')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        gg = self.getRAdata(eht, 'gg')[intc]

        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]
        uyux = self.getRAdata(eht, 'uxuy')[intc]
        uzux = self.getRAdata(eht, 'uxuz')[intc]

        uxuxux = self.getRAdata(eht, 'uxuxux')[intc]
        uyuyux = self.getRAdata(eht, 'uyuyux')[intc]
        uzuzux = self.getRAdata(eht, 'uzuzux')[intc]

        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
        ppux = self.getRAdata(eht, 'ppux')[intc]
        ttux = self.getRAdata(eht, 'ttux')[intc]

        #########################
        # KINETIC ENERGY EQUATION 
        #########################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        t_ddux = self.getRAdata(eht, 'ddux')
        t_dduy = self.getRAdata(eht, 'dduy')
        t_dduz = self.getRAdata(eht, 'dduz')

        t_uxux = self.getRAdata(eht, 'uxux')
        t_uyuy = self.getRAdata(eht, 'uyuy')
        t_uzuz = self.getRAdata(eht, 'uzuz')

        t_q = t_uxux + t_uyuy + t_uzuz

        # construct equation-specific mean fields
        q = uxux + uyuy + uzuz
        fqx = uxuxux - ux*uxux - ux*uxux - ux*uxux + ux*ux*ux + \
              uyuyux - uy*uyux - uy*uyux - ux*uyuy + uy*uy*ux + \
              uzuzux - uz*uzux - uz*uzux - ux*uzuz + uz*uz*ux

        fpx = ppux - pp * ux
        ftx = ttux - tt*ux

        # LHS -dq/dt - U \partial_r q
        self.minus_Dt_onehlf_q = -self.dt(0.5*t_q, xzn0, t_timec, intc) - ux*self.Grad(q,xzn0)

        # -R grad u
        rxx = uxux - ux * ux
        ryx = uyux - uy * ux
        rzx = uzux - uz * ux

        self.minus_r_grad_u = -(rxx * self.Grad(ux, xzn0) +
                                ryx * self.Grad(uy, xzn0) +
                                rzx * self.Grad(uz, xzn0))

        # RHS -div fqx
        self.minus_div_fqx = -self.Div(fqx, xzn0)

        # -div acoustic flux		
        #self.minus_div_fpx = -self.Div(fpx, xzn0)
        self.minus_div_fpx = np.zeros(nx)

        # RHS lambda ftx
        alpha = 1./tt
        lmbda = gg*alpha
        self.plus_lambda_ftx = +lmbda*ftx

        # -res		
        self.minus_resTKEequation = - (self.minus_Dt_onehlf_q + self.minus_r_grad_u + self.minus_div_fqx +
                                       self.minus_div_fpx + self.plus_lambda_ftx)

        # - kolm_rate u'3/lc
        self.minus_kolmrate = minus_kolmrate

        #############################
        # END KINETIC ENERGY EQUATION 
        #############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dd = dd
        self.fext = fext

    def plot_tke_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot kinetic energy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_Dt_onehlf_q

        rhs0 = self.minus_r_grad_u
        rhs1 = self.minus_div_fqx
        rhs2 = self.minus_div_fpx
        rhs3 = self.plus_lambda_ftx

        res = self.minus_resTKEequation

        rhs4 = self.minus_kolmrate * self.dd

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs1, rhs2, rhs3, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # model constant for variance dissipation
        Cm = 0.5

        # plot DATA 
        plt.title(r"canuto1992 tke equation C$_m$ = " + str(Cm))
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r'$-D_t 1/2 q$')

            plt.plot(grd1, rhs0, color='b', label=r"$-\overline{u_j u_i} \partial_x U_i$")
            plt.plot(grd1, rhs1, color='#802A2A', label=r"$-\nabla_x 1/2 \overline{q^2 u_x}$")
            plt.plot(grd1, rhs2, color='m', label=r"$-\nabla_x \overline{p u_x}$")
            plt.plot(grd1, rhs0, color='r', label=r'$+\lambda \overline{u_x \theta}$')

            # plt.plot(grd1, Cm * rhs4, color='k', linewidth=0.7, label=r"$-C_m \overline{\rho} u^{'3}_{rms}/l_c$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_{\epsilon_K}$")
        elif self.ig == 2:
            pass

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"erg cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'ek_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'ek_eq.eps')

    def tke_dissipation(self):
        return self.minus_resTkeEquation

    def tke(self):
        return self.tke
