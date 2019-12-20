import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TotalEnergyEquation(calc.CALCULUS, al.ALIMIT, object):

    def __init__(self, filename, ig, intc, tke_diss, data_prefix):
        super(TotalEnergyEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0'))
        nx = np.asarray(eht.item().get('nx'))

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])
        pp = np.asarray(eht.item().get('pp')[intc])

        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])
        dduxuy = np.asarray(eht.item().get('dduxuy')[intc])
        dduxuz = np.asarray(eht.item().get('dduxuz')[intc])

        ddekux = np.asarray(eht.item().get('ddekux')[intc])
        ddek = np.asarray(eht.item().get('ddek')[intc])

        ddei = np.asarray(eht.item().get('ddei')[intc])
        ddeiux = np.asarray(eht.item().get('ddeiux')[intc])

        divu = np.asarray(eht.item().get('divu')[intc])
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])
        ppux = np.asarray(eht.item().get('ppux')[intc])

        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])

        #######################
        # TOTAL ENERGY EQUATION 
        #######################  		

        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))
        t_dd = np.asarray(eht.item().get('dd'))

        t_ddei = np.asarray(eht.item().get('ddei'))

        t_ddux = np.asarray(eht.item().get('ddux'))
        t_dduy = np.asarray(eht.item().get('dduy'))
        t_dduz = np.asarray(eht.item().get('dduz'))

        t_dduxux = np.asarray(eht.item().get('dduxux'))
        t_dduyuy = np.asarray(eht.item().get('dduyuy'))
        t_dduzuz = np.asarray(eht.item().get('dduzuz'))

        t_uxux = np.asarray(eht.item().get('uxux'))
        t_uyuy = np.asarray(eht.item().get('uyuy'))
        t_uzuz = np.asarray(eht.item().get('uzuz'))

        t_fht_ek = 0.5 * (t_dduxux + t_dduyuy + t_dduzuz) / t_dd
        t_fht_ei = t_ddei / t_dd

        # construct equation-specific mean fields			
        # fht_ek = 0.5*(dduxux + dduyuy + dduzuz)/dd
        fht_ek = ddek / dd
        fht_ux = ddux / dd
        fht_ei = ddei / dd

        fei = ddeiux - ddux * ddei / dd
        fekx = ddekux - fht_ux * fht_ek
        fpx = ppux - pp * ux

        # LHS -dq/dt 			
        self.minus_dt_eht_dd_fht_ek = -self.dt(t_dd * t_fht_ek, xzn0, t_timec, intc)
        self.minus_dt_eht_dd_fht_ei = -self.dt(t_dd * t_fht_ei, xzn0, t_timec, intc)
        self.minus_dt_eht_dd_fht_et = self.minus_dt_eht_dd_fht_ek + \
                                      self.minus_dt_eht_dd_fht_ei

        # LHS -div dd ux te
        self.minus_div_eht_dd_fht_ux_fht_ek = -self.Div(dd * fht_ux * fht_ek, xzn0)
        self.minus_div_eht_dd_fht_ux_fht_ei = -self.Div(dd * fht_ux * fht_ei, xzn0)
        self.minus_div_eht_dd_fht_ux_fht_et = self.minus_div_eht_dd_fht_ux_fht_ek + \
                                              self.minus_div_eht_dd_fht_ux_fht_ei

        # RHS -div fei
        self.minus_div_fei = -self.Div(fei, xzn0)

        # RHS -div ftt (not included) heat flux
        self.minus_div_ftt = -np.zeros(nx)

        # -div kinetic energy flux
        self.minus_div_fekx = -self.Div(fekx, xzn0)

        # -div acoustic flux		
        self.minus_div_fpx = -self.Div(fpx, xzn0)

        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = -ux + fht_ux

        # +buoyancy work
        self.plus_wb = self.plus_ax * self.Grad(pp, xzn0)

        # RHS -P d = - eht_pp Div eht_ux
        self.minus_pp_div_ux = -pp * self.Div(ux, xzn0)

        # -R grad u

        rxx = dduxux - ddux * ddux / dd
        rxy = dduxuy - ddux * dduy / dd
        rxz = dduxuz - ddux * dduz / dd

        self.minus_r_grad_u = -(rxx * self.Grad(ddux / dd, xzn0) + \
                                rxy * self.Grad(dduy / dd, xzn0) + \
                                rxz * self.Grad(dduz / dd, xzn0))

        # +dd Dt fht_ui_fht_ui_o_two
        t_fht_ux = t_ddux / t_dd
        t_fht_uy = t_dduy / t_dd
        t_fht_uz = t_dduz / t_dd

        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd

        self.plus_dd_Dt_fht_ui_fht_ui_o_two = \
            +self.dt(t_dd * (t_fht_ux ** 2. + t_fht_uy ** 2. + t_fht_uz ** 2.), xzn0, t_timec, intc) - \
            self.Div(dd * fht_ux * (fht_ux ** 2. + fht_uy ** 2. + fht_uz ** 2.), xzn0) / 2.

        # RHS source + dd enuc
        self.plus_dd_fht_enuc = ddenuc1 + ddenuc2

        # -res		
        self.minus_resTeEquation = - (self.minus_dt_eht_dd_fht_et + self.minus_div_eht_dd_fht_ux_fht_et + \
                                      self.minus_div_fei + self.minus_div_ftt + self.minus_div_fekx + \
                                      self.minus_div_fpx + self.minus_r_grad_u + self.minus_pp_div_ux + \
                                      self.plus_wb + self.plus_dd_fht_enuc + self.plus_dd_Dt_fht_ui_fht_ui_o_two)

        ###########################
        # END TOTAL ENERGY EQUATION 
        ###########################  

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.fht_et = fht_ei + fht_ek
        self.ig = ig

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

        if (self.ig == 1):
            plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{\varepsilon}_t$')
            # define x LABEL
            setxlabel = r"x (cm)"
        elif (self.ig == 2):
            plt.plot(grd1, plt1, color='brown', label=r'$\widetilde{\varepsilon}_t$')
            # define x LABEL
            setxlabel = r"r (cm)"
        else:
            print(
                "ERROR (TotalEnergyEquation.py): geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        # define y LABEL
        setylabel = r"$\widetilde{\varepsilon}_t$ (erg g$^{-1}$)"

        # show x/y LABELS
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_et.png')

    def plot_et_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot total energy equation in the model"""

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_fht_et
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_et

        rhs0 = self.minus_div_fei
        rhs1 = self.minus_div_ftt
        rhs2 = self.minus_div_fekx
        rhs3 = self.minus_div_fpx
        rhs4 = self.minus_r_grad_u
        rhs5 = self.minus_pp_div_ux
        rhs6 = self.plus_wb
        rhs7 = self.plus_dd_fht_enuc
        rhs8 = self.plus_dd_Dt_fht_ui_fht_ui_o_two

        res = self.minus_resTeEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, rhs6, rhs7, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('total energy equation')
        if (self.ig == 1):
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \widetilde{\epsilon}_t )$")
            plt.plot(grd1, lhs1, color='k',
                     label=r"$-\nabla_x (\overline{\rho}\widetilde{u}_x \widetilde{\epsilon}_t$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_x f_I $")
            plt.plot(grd1, rhs1, color='y', label=r"$-\nabla_x f_T$ (not incl.)")
            plt.plot(grd1, rhs2, color='silver', label=r"$-\nabla_x f_k$")
            plt.plot(grd1, rhs3, color='c', label=r"$-\nabla_x f_p$")
            plt.plot(grd1, rhs4, color='m', label=r"$-\widetilde{R}_{xi}\partial_x \widetilde{u_i}$")
            plt.plot(grd1, rhs5, color='#802A2A', label=r"$-\bar{P} \bar{d}$")
            plt.plot(grd1, rhs6, color='r', label=r'$+W_b$')
            plt.plot(grd1, rhs7, color='b', label=r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1, rhs8, color='g',
                     label=r"$+\overline{\rho} \widetilde{D}_t \widetilde{u}_i \widetilde{u}_i/2$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_{\epsilon_t}$")
            # define X label
            setxlabel = r'x (10$^{8}$ cm)'
        elif (self.ig == 2):
            plt.plot(grd1, lhs0, color='#FF6EB4', label=r"$-\partial_t (\overline{\rho} \widetilde{\epsilon}_t )$")
            plt.plot(grd1, lhs1, color='k',
                     label=r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{\epsilon}_t$)")

            plt.plot(grd1, rhs0, color='#FF8C00', label=r"$-\nabla_r f_I $")
            plt.plot(grd1, rhs1, color='y', label=r"$-\nabla_r f_T$ (not incl.)")
            plt.plot(grd1, rhs2, color='silver', label=r"$-\nabla_r f_k$")
            plt.plot(grd1, rhs3, color='c', label=r"$-\nabla_r f_p$")
            plt.plot(grd1, rhs4, color='m', label=r"$-\widetilde{R}_{ri}\partial_r \widetilde{u_i}$")
            plt.plot(grd1, rhs5, color='#802A2A', label=r"$-\bar{P} \bar{d}$")
            plt.plot(grd1, rhs6, color='r', label=r'$+W_b$')
            plt.plot(grd1, rhs7, color='b', label=r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}$")
            plt.plot(grd1, rhs8, color='g',
                     label=r"$+\overline{\rho} \widetilde{D}_t \widetilde{u}_i \widetilde{u}_i/2$")

            plt.plot(grd1, res, color='k', linestyle='--', label=r"res $\sim N_{\epsilon_t}$")
            # define X label
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        # define and show x/y LABELS				
        setylabel = r"erg cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'et_eq.png')
