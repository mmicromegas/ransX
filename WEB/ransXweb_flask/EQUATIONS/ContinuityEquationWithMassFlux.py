import numpy as np
import sys
import mpld3
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ContinuityEquationWithMassFlux(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, outputFile, filename, ig, fext, intc, nsdim, data_prefix):
        super(ContinuityEquationWithMassFlux, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')
        nx = self.getRAdata(eht, 'nx')
        ny = self.getRAdata(eht, 'ny')
        nz = self.getRAdata(eht, 'nz')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_frho = self.getRAdata(eht, 'ddux') - self.getRAdata(eht, 'dd') * self.getRAdata(eht, 'ux')

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
        self.ny = ny
        self.nz = nz
        self.ig = ig
        self.fext = fext
        self.t_timec = t_timec
        self.pp = pp
        self.nsdim = nsdim
        self.outputFile = outputFile

    def plot_rho(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
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
        # fig = plt.figure(figsize=(5, 4))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1 / ysc]
        self.set_plt_axis(LAXIS, xbl / xsc, xbr / xsc, ybu / ysc, ybd / ysc, [yval / ysc for yval in to_plot])

        # plot DATA 
        plt.title('density')
        plt.plot(grd1 / xsc, plt1 / ysc, color='brown', label=r'rho')

        # convective boundary markers
        # plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        # plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        ycol = np.linspace(ybu / ysc, ybd / ysc, num=100)
        for i in ycol:
            plt.text(bconv/xsc,i, '.',dict(size=15))
            plt.text(tconv/xsc,i, '.',dict(size=15))

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)

        setylabel = r'rho (' + "{:.0e}".format(ysc) + ' g/cm3)'
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        # plt.show(block=False)

        # save PLOT
        # if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_rho.png')
        # elif self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_rho.eps')

        # html_str = mpld3.fig_to_html(fig)
        # Html_file = open(self.outputFile, "w")
        # Html_file.write(html_str)
        # Html_file.close()

    def plot_continuity_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
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
        # fig = plt.figure(figsize=(5, 4))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0 , lhs1 , rhs0 , rhs1 , rhs2 , res]
        self.set_plt_axis(LAXIS, xbl / xsc, xbr / xsc, ybu / ysc, ybd / ysc, [yval / ysc for yval in to_plot])

        # plot DATA
        plt.title(str(self.nsdim) + "D")

        if self.ig == 1:
            plt.plot(grd1 / xsc, lhs0 / ysc, color='g', linewidth=4, label=r'-dt rho')
            plt.plot(grd1 / xsc, lhs1 / ysc, color='r', label=r'-ux gradx rho')
            plt.plot(grd1 / xsc, rhs0 / ysc, color='c', label=r"-div frho")
            plt.plot(grd1 / xsc, rhs1 / ysc, color='m', label=r"+(frho / rho) gradx dd")
            # plt.plot(grd1/xsc, rhs2/ysc, color='b', label=r'$-\overline{\rho} \nabla_x (\overline{u}_x)$')
            plt.plot(grd1 / xsc, rhs2 / ysc, color='b', label=r'-rho d')
            plt.plot(grd1 / xsc, res / ysc, color='k', linestyle='--', label='+res')
            plt.plot(grd1/xsc, np.zeros(self.nx), color='k', linestyle='dotted')
        elif self.ig == 2:
            plt.plot(grd1 / xsc, lhs0 / ysc, color='g', linewidth=4, label=r'-dt rho')
            plt.plot(grd1 / xsc, lhs1 / ysc, color='r', label=r'-ux gradx rho')
            plt.plot(grd1 / xsc, rhs0 / ysc, color='c', label=r"-div frho")
            plt.plot(grd1 / xsc, rhs1 / ysc, color='m', label=r"+(frho / rho) gradx dd")
            # plt.plot(grd1/xsc, rhs2/ysc, color='b', label=r'$-\overline{\rho} \nabla_x (\overline{u}_x)$')
            plt.plot(grd1 / xsc, rhs2 / ysc, color='b', label=r'-rho d')
            plt.plot(grd1 / xsc, res / ysc, color='k', linestyle='--', label='+res')
            plt.plot(grd1/xsc, np.zeros(self.nx), color='k', linestyle='dotted')

        ycol = np.linspace(ybu / ysc, ybd / ysc, num=100)
        for i in ycol:
            plt.text(bconv/xsc,i, '.',dict(size=15))
            plt.text(tconv/xsc,i, '.',dict(size=15))

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + ' cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + ' cm)'
            plt.xlabel(setxlabel)

        setylabel = r'g/cm3/s (' + "{:.0e}".format(ysc) + ')'
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12}, ncol=2)

        # display PLOT
        # plt.show(block=False)

        # save PLOT
        # if self.fext == 'png':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq.png')
        # elif self.fext == 'eps':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq.eps')

        # html_str = mpld3.fig_to_html(fig)
        # Html_file = open(self.outputFile, "w")
        # Html_file.write(html_str)
        # Html_file.close()

        return "success"

    def plot_continuity_equation_integral_budget(self, plabel, ax, laxis, xbl, xbr, ybu, ybd, fc):
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

        # hack for the ccp setup getting rid of bndry noise
        #fct1 = 0.5e-1
        #fct2 = 1.e-1
        #xbl = xbl + fct1*xbl
        #xbr = xbr - fct2*xbl
        if plabel == 'ccptwo':
            xbl = 4.5e8
            xbr = 11.5e8

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
        if self.ig == 1 and self.nsdim == 3:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.zzn0[-1] - self.zzn0[0])
        if self.ig == 1 and self.nsdim == 2:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.yzn0[-1] - self.yzn0[0])
        elif self.ig == 2:
            Sr = 4. * np.pi * rc ** 2

        int_term1 = integrate.simps(term1_sel * Sr, rc)
        int_term2 = integrate.simps(term2_sel * Sr, rc)
        int_term3 = integrate.simps(term3_sel * Sr, rc)
        int_term4 = integrate.simps(term4_sel * Sr, rc)
        int_term5 = integrate.simps(term5_sel * Sr, rc)
        int_term6 = integrate.simps(term6_sel * Sr, rc)

        # fig = plt.figure(figsize=(5, 4))

        # ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd / fc, ybu / fc])

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
        ax.set_ylabel(r'g/s (' + "{:.0e}".format(fc) + ')')

        # Create a title, in italics
        ax.set_title('integral budget')

        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        if self.ig == 1:
            # group_labels = [r'$-\partial_t (\overline{\rho})$', r'$-\widetilde{u}_x \partial_x (\overline{\rho})$',
            #                r"$-\nabla_x f_\rho$", r"$+f_\rho / \overline{\rho} \partial_x \overline{\rho}$",
            #                r'$-\overline{\rho} \nabla_x (\overline{u}_x)$', 'res']

            group_labels = [r'-dt rho', r'-ux gradx rho',
                            r'-div frho', r'+frgdd',
                            r'-rho div ux', 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)
        elif self.ig == 2:
            #group_labels = [r'$-\partial_t (\overline{\rho})$', r'$-\widetilde{u}_r \partial_r (\overline{\rho})$',
            #                r"$-\nabla_r f_\rho$", r"$+f_\rho / \overline{\rho} \partial_r \overline{\rho}$",
            #                r'$-\overline{\rho} \nabla_r (\overline{u}_r)$', 'res']
            group_labels = [r'-dt rho', r'-ux gradx rho',
                            r'-div frho', r'+frgdd',
                            r'-rho div ux', 'res']


            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)

        # auto-rotate the x axis labels
        # fig.autofmt_xdate()

        # display PLOT
        # plt.show(block=False)

        # save PLOT
        # if self.fext == 'png':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq_bar.png')
        # elif self.fext == 'eps':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'continuityWithMassFlux_eq_bar.eps')

        # html_str = mpld3.fig_to_html(fig)
        # Html_file = open(self.outputFile, "w")
        # Html_file.write(html_str)
        # Html_file.close()

        return "success"

    def plot_Frho_space_time(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot Frho space time diagram"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithMassFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

        t_timec = self.t_timec

        # load x GRID
        nx = self.nx
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.t_frho.T
        # plt1 = self.t_frho.T

        indRES = np.where((grd1 < 9.e8) & (grd1 > 4.e8))[0]

        # pltMax = np.max(plt1[indRES])
        # pltMin = np.min(plt1[indRES])

        pltMax = 0.2e10
        pltMin = -4.e10

        # create FIGURE
        # plt.figure(figsize=(5, 4))

        # print(t_timec[0], t_timec[-1], grd1[0], grd1[-1])

        # matplotlib.rc('xtick', labelsize=30.)
        # matplotlib.rc('ytick', labelsize=30.)
        # matplotlib.rc('font', size=30.)

        fig, ax = plt.subplots(figsize=(14, 7))
        # fig.suptitle("log(X) (" + self.setNucNoUp(str(element))+ ")")
        fig.suptitle(r"$f_\rho$ " + str(self.nx) + ' x ' + str(self.ny) + ' x ' + str(self.nz))

        im = ax.imshow(plt1, interpolation='bilinear', cmap=cm.jet,
                       origin='lower', extent=[t_timec[0], t_timec[-1], grd1[0], grd1[-1]], aspect='auto',
                       vmax=pltMax, vmin=pltMin)

        # extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'time (s)'
            setylabel = r"r ($10^8$ cm)"
            ax.set_xlabel(setxlabel)
            ax.set_ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'time (s)'
            setylabel = r"r ($10^8$ cm)"
            ax.set_xlabel(setxlabel)
            ax.set_ylabel(setylabel)

        # display PLOT
        # plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Frho_space_time' + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Frho_space_time' + '.eps')
