import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ContinuityEquationWithMassFlux(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, data_prefix):
        super(ContinuityEquationWithMassFlux, self).__init__(ig)

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
        ddux = self.getRAdata(eht, 'ddux')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        # t_mm    = self.getRAdata(eht,'mm'))
        # minus_dt_mm = -self.dt(t_mm,xzn0,t_timec,intc)
        # fht_ux = minus_dt_mm/(4.*np.pi*(xzn0**2.)*dd)

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fdd = ddux - dd * ux

        ####################################
        # CONTINUITY EQUATION WITH MASS FLUX
        ####################################

        # LHS -dq/dt 		
        self.minus_dt_dd = -self.dt(t_dd, xzn0, t_timec, intc)

        # LHS -fht_ux Grad dd
        self.minus_fht_ux_grad_dd = -fht_ux * self.Grad(dd, xzn0)

        # RHS -Div fdd
        self.minus_div_fdd = -self.Div(fdd, xzn0)

        # RHS +fdd_o_dd gradx dd				
        self.plus_fdd_o_dd_gradx_dd = +(fdd / dd) * self.Grad(dd, xzn0)

        # RHS -dd Div ux 
        self.minus_dd_div_ux = -dd * self.Div(ux, xzn0)

        # -res
        self.minus_resContEquation = -(self.minus_dt_dd + self.minus_fht_ux_grad_dd + self.minus_div_fdd +
                                       self.plus_fdd_o_dd_gradx_dd + self.minus_dd_div_ux)

        ########################################
        # END CONTINUITY EQUATION WITH MASS FLUX
        ########################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0
        self.dd = dd
        self.nx = nx
        self.ig = ig
        self.fext = fext

    def plot_rho(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot rho stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithMassFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.dd

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('density')
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            plt.xlabel(setxlabel)

        setylabel = r"$\overline{\rho}$ (g cm$^{-3}$)"
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rho.png')
        elif self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rho.eps')

    def plot_continuity_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot continuity equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithMassFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd
        lhs1 = self.minus_fht_ux_grad_dd

        rhs0 = self.minus_div_fdd
        rhs1 = self.plus_fdd_o_dd_gradx_dd
        rhs2 = self.minus_dd_div_ux

        res = self.minus_resContEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('continuity equation with mass flux')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='g', label=r'$-\partial_t (\overline{\rho})$')
            plt.plot(grd1, lhs1, color='r', label=r'$-\widetilde{u}_x \partial_x (\overline{\rho})$')
            plt.plot(grd1, rhs0, color='c', label=r"$-\nabla_x f_\rho$")
            plt.plot(grd1, rhs1, color='m', label=r"$+f_\rho / \overline{\rho} \partial_x \overline{\rho}$")
            plt.plot(grd1, rhs2, color='b', label=r'$-\overline{\rho} \nabla_x (\overline{u}_x)$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='g', label=r'$-\partial_t (\overline{\rho})$')
            plt.plot(grd1, lhs1, color='r', label=r'$-\widetilde{u}_r \partial_r (\overline{\rho})$')
            plt.plot(grd1, rhs0, color='c', label=r"$-\nabla_r f_\rho$")
            plt.plot(grd1, rhs1, color='m', label=r"$+f_\rho / \overline{\rho} \partial_r \overline{\rho}$")
            plt.plot(grd1, rhs2, color='b', label=r'$-\overline{\rho} \nabla_r (\overline{u}_r)$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            plt.xlabel(setxlabel)

        setylabel = r"g cm$^{-3}$ s$^{-1}$"
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 13}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq.eps')

    def plot_continuity_equation_integral_budget(self, laxis, xbl, xbr, ybu, ybd):
        """Plot integral budgets of continuity equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithMassFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

        term1 = self.minus_dt_dd
        term2 = self.minus_fht_ux_grad_dd
        term3 = self.minus_div_fdd
        term4 = self.plus_fdd_o_dd_gradx_dd
        term5 = self.minus_dd_div_ux
        term6 = self.minus_resContEquation

        # calculate INDICES for grid boundaries 
        if laxis == 1 or laxis == 2:
            idxl, idxr = self.idx_bndry(xbl, xbr)
        else:
            idxl = 0
            idxr = self.nx - 1

        term1_sel = term1[idxl:idxr]
        term2_sel = term2[idxl:idxr]
        term3_sel = term3[idxl:idxr]
        term4_sel = term4[idxl:idxr]
        term5_sel = term5[idxl:idxr]
        term6_sel = term6[idxl:idxr]

        rc = self.xzn0[idxl:idxr]

        # handle geometry
        Sr = 0.
        if self.ig == 1:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.zzn0[-1] - self.zzn0[0])
        elif self.ig == 2:
            Sr = 4. * np.pi * rc ** 2

        int_term1 = integrate.simps(term1_sel * Sr, rc)
        int_term2 = integrate.simps(term2_sel * Sr, rc)
        int_term3 = integrate.simps(term3_sel * Sr, rc)
        int_term4 = integrate.simps(term4_sel * Sr, rc)
        int_term5 = integrate.simps(term5_sel * Sr, rc)
        int_term6 = integrate.simps(term6_sel * Sr, rc)

        fig = plt.figure(figsize=(7, 6))

        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd, ybu])

        fc = 1.

        # note the change: I'm only supplying y data.
        y = [int_term1 / fc, int_term2 / fc, int_term3 / fc, int_term4 / fc, int_term5 / fc, int_term6 / fc]

        # Calculate how many bars there will be
        N = len(y)

        # Generate a list of numbers, from 0 to N
        # This will serve as the (arbitrary) x-axis, which
        # we will then re-label manually.
        ind = range(N)

        # See note below on the breakdown of this command
        ax.bar(ind, y, facecolor='#0000FF',
               align='center', ecolor='black')

        # Create a y label
        ax.set_ylabel(r'g s$^{-1}$')

        # Create a title, in italics
        ax.set_title('continuity with mass flux integral budget')

        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        if self.ig == 1:
            group_labels = [r'$-\partial_t (\overline{\rho})$', r'$-\widetilde{u}_x \partial_x (\overline{\rho})$',
                            r"$-\nabla_x f_\rho$", r"$+f_\rho / \overline{\rho} \partial_x \overline{\rho}$",
                            r'$-\overline{\rho} \nabla_x (\overline{u}_x)$', 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=16)
        elif self.ig == 2:
            group_labels = [r'$-\partial_t (\overline{\rho})$', r'$-\widetilde{u}_r \partial_r (\overline{\rho})$',
                            r"$-\nabla_r f_\rho$", r"$+f_\rho / \overline{\rho} \partial_r \overline{\rho}$",
                            r'$-\overline{\rho} \nabla_r (\overline{u}_r)$', 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=16)

        # auto-rotate the x axis labels
        fig.autofmt_xdate()

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq_bar.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq_bar.eps')
