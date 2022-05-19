import numpy as np
import sys
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors
from scipy import integrate

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class MomentumEquationX(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, fext, intc, nsdim, data_prefix, plabel):
        super(MomentumEquationX, self).__init__(ig)

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
        gg = self.getRAdata(eht, 'gg')[intc]

        ddgg = self.getRAdata(eht, 'ddgg')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_ddux = self.getRAdata(eht, 'ddux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        rxx = dduxux - ddux * ddux / dd

        #####################
        # X MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_ddux = -self.dt(t_ddux, xzn0, t_timec, intc)

        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_ux = -self.Div(dd * fht_ux * fht_ux, xzn0)

        # RHS -div rxx
        self.minus_div_rxx = -self.Div(rxx, xzn0)

        # RHS -G
        if self.ig == 1:
            self.minus_G = np.zeros(nx)
        elif self.ig == 2:
            self.minus_G = -(-dduyuy - dduzuz) / xzn0

        # RHS -(grad P - rho g)
        if plabel == '3d-neshellBoost10x-25ele':
            self.minus_gradx_pp_eht_dd_eht_gg = -self.Grad(pp,xzn0) +dd*gg
        else:
            self.minus_gradx_pp_eht_dd_eht_gg = -self.Grad(pp, xzn0) + ddgg

        # for i in range(nx):
        #    print(2.*ddgg[i],dd[i]*gg[i]		

        # -res
        self.minus_resResXmomentumEquation = \
            -(self.minus_dt_ddux + self.minus_div_eht_dd_fht_ux_fht_ux + self.minus_div_rxx
              + self.minus_G + self.minus_gradx_pp_eht_dd_eht_gg)

        #########################
        # END X MOMENTUM EQUATION 
        #########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0

        self.ddux = ddux
        self.ux = ux
        self.ig = ig
        self.fext = fext
        self.nsdim = nsdim

    def plot_momentum_x(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
        """Plot ddux stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationX.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ddux
        # plt2 = self.ux
        # plt3 = self.vexp

        # create FIGURE
        #plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl / xsc, xbr / xsc, ybu / ysc, ybd / ysc, [yval / ysc for yval in to_plot])

        # plot DATA and set labels
        plt.title('ddux')
        if self.ig == 1:
            plt.plot(grd1/xsc, plt1/ysc, color='brown', label=r'rho ux')
            # plt.plot(grd1,plt2,color='green',label = r'$\overline{u}_x$')
            # plt.plot(grd1,plt3,color='red',label = r'$v_{exp}$')
        elif self.ig == 2:
            plt.plot(grd1/xsc, plt1/ysc, color='brown', label=r'rho ur')
            # plt.plot(grd1,plt2,color='green',label = r'$\overline{u}_x$')
            # plt.plot(grd1,plt3,color='red',label = r'$v_{exp}$')

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        ycol = np.linspace(ybu / ysc, ybd / ysc, num=100)
        for i in ycol:
            plt.text(bconv/xsc,i, '.',dict(size=15))
            plt.text(tconv/xsc,i, '.',dict(size=15))

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)
            setylabel = r'rho ux (' + "{:.0e}".format(ysc) + ' g/cm2/s)'
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)
            setylabel = r'rho ur (' + "{:.0e}".format(ysc) + ' g/cm2/s)'
            plt.ylabel(setylabel)


        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        #plt.show(block=False)

        # save PLOT
        #if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_ddux.png')
        #if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_ddux.eps')

    def plot_momentum_equation_x(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
        """Plot momentum x equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(MomentumEquationX.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_ddux
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_ux

        rhs0 = self.minus_div_rxx
        rhs1 = self.minus_G
        rhs2 = self.minus_gradx_pp_eht_dd_eht_gg

        res = self.minus_resResXmomentumEquation

        # create FIGURE
        #plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl / xsc, xbr / xsc, ybu / ysc, ybd / ysc, [yval / ysc for yval in to_plot])

        group_labels = [r'-dt rho ur', r'-divr ur ur',
                        r'-divr Rrr', r'-Geom', r'-(grad P - rho g)',
                        'res']

        # plot DATA 
        plt.title('x momentum equation ' + str(self.nsdim) + "D")
        if self.ig == 1:
            plt.plot(grd1/xsc, lhs0/ysc, color='c', label=r"-dt rho ux")
            plt.plot(grd1/xsc, lhs1/ysc, color='m', label=r"-divx ux ux")
            plt.plot(grd1/xsc, rhs0/ysc, color='b', label=r"-divx Rxx")
            plt.plot(grd1/xsc, rhs2/ysc, color='r', label=r"-(grad P - rho g)")
            plt.plot(grd1/xsc, res/ysc, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1/xsc, lhs0/ysc, color='c', label=r"-dt rho ur")
            plt.plot(grd1/xsc, lhs1/ysc, color='m', label=r"-divr ur ur")
            plt.plot(grd1/xsc, rhs0/ysc, color='b', label=r"-divr Rrr")
            plt.plot(grd1/xsc, rhs1/ysc, color='g', label=r"-Geom")
            plt.plot(grd1/xsc, rhs2/ysc, color='r', label=r"-(grad P - rho g)")
            plt.plot(grd1/xsc, res/ysc, color='k', linestyle='--', label='res')

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

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

        setylabel = r'g/cm2/s2 (' + "{:.0e}".format(ysc) + ')'
        plt.ylabel(setylabel)


        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        #if wxStudio:
        #    plt.show()
        #else:
        #    plt.show(block=False)

        # save PLOT
        #if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'momentum_x_eq.png')
        #if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'momentum_x_eq.eps')

    def plot_momentum_x_integral_budget(self, plabel, ax, laxis, xbl, xbr, ybu, ybd, fc):
        """Plot integral budgets of continuity equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithMassFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

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

        if self.ig == 1:
            term1 = self.minus_dt_ddux
            term2 = self.minus_div_eht_dd_fht_ux_fht_ux
            term3 = self.minus_div_rxx
            term4 = self.minus_gradx_pp_eht_dd_eht_gg
            term5 = self.minus_resResXmomentumEquation

            term1_sel = term1[idxl:idxr]
            term2_sel = term2[idxl:idxr]
            term3_sel = term3[idxl:idxr]
            term4_sel = term4[idxl:idxr]
            term5_sel = term5[idxl:idxr]

            rc = self.xzn0[idxl:idxr]

            # handle geometry
            if self.nsdim == 3:
                Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.zzn0[-1] - self.zzn0[0])
            elif self.nsdim == 2:
                Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.yzn0[-1] - self.yzn0[0])
            else:
                print('ERROR (MomentumEquationX.py): dimensionality not defined.')

            int_term1 = integrate.simps(term1_sel * Sr, rc)
            int_term2 = integrate.simps(term2_sel * Sr, rc)
            int_term3 = integrate.simps(term3_sel * Sr, rc)
            int_term4 = integrate.simps(term4_sel * Sr, rc)
            int_term5 = integrate.simps(term5_sel * Sr, rc)

            # fig = plt.figure(figsize=(5, 4))

            # ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.grid(color='gray', linestyle='dashed')
            ax.xaxis.grid(color='gray', linestyle='dashed')

            if laxis == 2:
                plt.ylim([ybd / fc, ybu / fc])

            # note the change: I'm only supplying y data.
            y = [int_term1 / fc, int_term2 / fc, int_term3 / fc, int_term4 / fc, int_term5 / fc]

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
            ax.set_ylabel(r'gcm/s2 (' + "{:.0e}".format(fc) + ')')

            # Create a title, in italics
            ax.set_title('integral budget')

            # This sets the ticks on the x axis to be exactly where we put
            # the center of the bars.
            ax.set_xticks(ind)

            # Labels for the ticks on the x axis.  It needs to be the same length
            # as y (one label for each bar)
            group_labels = [r'-dt rho ux', r'-divx ux ux',
                            r'-divx Rxx', r'-(grad P - rho g)',
                            'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)

        elif self.ig == 2:
            term1 = self.minus_dt_ddux
            term2 = self.minus_div_eht_dd_fht_ux_fht_ux
            term3 = self.minus_div_rxx
            term4 = self.minus_G
            term5 = self.minus_gradx_pp_eht_dd_eht_gg
            term6 = self.minus_resResXmomentumEquation


            term1_sel = term1[idxl:idxr]
            term2_sel = term2[idxl:idxr]
            term3_sel = term3[idxl:idxr]
            term4_sel = term4[idxl:idxr]
            term5_sel = term5[idxl:idxr]
            term6_sel = term6[idxl:idxr]

            rc = self.xzn0[idxl:idxr]

            # handle geometry
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
            ax.set_ylabel(r'gcm/s2 (' + "{:.0e}".format(fc) + ')')

            # Create a title, in italics
            ax.set_title('integral budget')

            # This sets the ticks on the x axis to be exactly where we put
            # the center of the bars.
            ax.set_xticks(ind)

            # Labels for the ticks on the x axis.  It needs to be the same length
            # as y (one label for each bar)
            group_labels = [r'-dt rho ur', r'-divr ur ur',
                            r'-divr Rrr', r'-Geom', r'-(grad P - rho g)',
                            'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)



        return "success"
