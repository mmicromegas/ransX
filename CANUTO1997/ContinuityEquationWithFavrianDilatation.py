# class for RANS ContinuityEquationWithFavrianDilatation #

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

class ContinuityEquationWithFavrianDilatation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, data_prefix):
        super(ContinuityEquationWithFavrianDilatation, self).__init__(ig)

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
        ddux = self.getRAdata(eht, 'ddux')[intc]
        mm = self.getRAdata(eht, 'mm')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd

        #############################################
        # CONTINUITY EQUATION WITH FAVRIAN DILATATION
        #############################################

        # LHS -Dq/Dt
        self.minus_Dt_dd = -(self.dt(t_dd, xzn0, t_timec, intc) + fht_ux * self.Grad(dd, xzn0))

        # RHS -dd Div fht_ux 
        self.minus_dd_div_fht_ux = -dd * self.Div(fht_ux, xzn0)

        # -res
        self.minus_resContEquation = -(self.minus_Dt_dd + self.minus_dd_div_fht_ux)

        #################################################
        # END CONTINUITY EQUATION WITH FAVRIAN DILATATION
        #################################################

        # ad hoc variables
        vol = (4. / 3.) * np.pi * xzn0 ** 3
        mm_ver2 = dd * vol

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0
        self.dd = dd
        self.nx = nx
        self.ig = ig
        self.fext = fext
        self.vol = vol
        self.mm = mm
        self.mm_ver2 = mm_ver2

    def plot_rho(self, laxis, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot rho stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorGeometry(self.ig))
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
        self.set_plt_axis(laxis, xbl, xbr, ybu, ybd, to_plot)

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

        # check supported file output extension
        if self.fext != "png" and self.fext != "eps":
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorOutputFileExtension(self.fext))
            sys.exit()

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rho_canuto1997.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rho_canuto1997.eps')

    def plot_continuity_equation(self, laxis, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot continuity equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_Dt_dd

        rhs0 = self.minus_dd_div_fht_ux

        res = self.minus_resContEquation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, res]
        self.set_plt_axis(laxis, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('continuity equation with Favrian dilatation')

        if self.ig == 1:
            plt.plot(grd1, lhs0, color='g', label=r'$-\widetilde{D}_t (\overline{\rho})$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\overline{\rho} \nabla_x (\widetilde{u}_x)$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='g', label=r'$-\widetilde{D}_t (\overline{\rho})$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\overline{\rho} \nabla_r (\widetilde{u}_r)$')
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
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # check supported file output extension
        if self.fext != "png" and self.fext != "eps":
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorOutputFileExtension(self.fext))
            sys.exit()

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityFavreDil_eq_canuto1997.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityFavreDil_eq_canuto1997.eps')

    def plot_continuity_equation_integral_budget(self, laxis, xbl, xbr, ybu, ybd):
        """Plot integral budgets of continuity equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        term1 = self.minus_Dt_dd
        term3 = self.minus_dd_div_fht_ux
        term4 = self.minus_resContEquation

        # hack for the ccp setup getting rid of bndry noise
        fct1 = 0.5e-1
        fct2 = 1.e-1
        xbl = xbl + fct1*xbl
        xbr = xbr - fct2*xbl
        print(xbl,xbr)

        # calculate INDICES for grid boundaries
        if laxis == 1 or laxis == 2:
            idxl, idxr = self.idx_bndry(xbl, xbr)
        else:
            idxl = 0
            idxr = self.nx - 1

        term1_sel = term1[idxl:idxr]
        term3_sel = term3[idxl:idxr]
        term4_sel = term4[idxl:idxr]

        rc = self.xzn0[idxl:idxr]

        # handle geometry
        Sr = 0.
        if self.ig == 1:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.zzn0[-1] - self.zzn0[0])
        elif self.ig == 2:
            Sr = 4. * np.pi * rc ** 2

        int_term1 = integrate.simps(term1_sel * Sr, rc)
        int_term3 = integrate.simps(term3_sel * Sr, rc)
        int_term4 = integrate.simps(term4_sel * Sr, rc)

        fig = plt.figure(figsize=(7, 6))

        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd, ybu])

        fc = 1.

        # note the change: I'm only supplying y data.
        y = [int_term1 / fc, int_term3 / fc, int_term4 / fc]

        # calculate how many bars there will be
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
        ax.set_title(r"continuity with $\widetilde{d}$ integral budget")

        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        if self.ig == 1:
            group_labels = [r"$-\widetilde{D}_t \overline{\rho}$",
                            r"$-\widetilde{u}_x \partial_x \overline{\rho}$", 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=16)
        elif self.ig == 2:
            group_labels = [r"$-\widetilde{D}_t \overline{\rho}$",
                            r"$-\widetilde{u}_r \partial_r \overline{\rho}$", 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=16)

        # auto-rotate the x axis labels
        fig.autofmt_xdate()

        # display PLOT
        plt.show(block=False)

        # check supported file output extension
        if self.fext != "png" and self.fext != "eps":
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorOutputFileExtension(self.fext))
            sys.exit()

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityFavreDil_eq_bar_canuto1997.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'continuityFavreDil_eq_bar_canuto1997.eps')
