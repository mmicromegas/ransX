import matplotlib.pyplot as plt
from scipy import integrate
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Errors import Errors
from EQUATIONS.TurbulentKineticEnergyCalculation import TurbulentKineticEnergyCalculation
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.cm as cm
import mpld3

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentKineticEnergyEquation(SetAxisLimit, Errors, object):

    def __init__(self, outputFile, filename, ig, intc, nsdim, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o, data_prefix):
        super(TurbulentKineticEnergyEquation, self).__init__()

        # instantiate turbulent kinetic energy object
        tkeF = TurbulentKineticEnergyCalculation(filename, ig, intc)

        # load all fields
        tkefields = tkeF.getTKEfield()

        self.xzn0 = tkefields['xzn0']
        self.yzn0 = tkefields['yzn0']
        self.zzn0 = tkefields['zzn0']

        # LHS -dq/dt
        self.minus_dt_dd_tke = tkefields['minus_dt_dd_tke']

        # LHS -dq/dt
        self.minus_dt_dd_tke = tkefields['minus_dt_dd_tke']

        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = tkefields['minus_div_eht_dd_fht_ux_tke']

        # -div kinetic energy flux
        self.minus_div_fekx = tkefields['minus_div_fekx']

        # -div acoustic flux		
        self.minus_div_fpx = tkefields['minus_div_fpx']

        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = tkefields['plus_ax']

        # +buoyancy work
        self.plus_wb = tkefields['plus_wb']

        # +pressure dilatation
        self.plus_wp = tkefields['plus_wp']

        # -R grad u
        self.minus_r_grad_u = tkefields['minus_r_grad_u']
        # -res		
        self.minus_resTkeEquation = tkefields['minus_resTkeEquation']

        #######################################
        # END TURBULENT KINETIC ENERGY EQUATION 
        #######################################

        # - kolm_rate u'3/lc
        self.minus_kolmrate = -kolmdissrate

        # convection boundaries
        self.tconv = tconv
        self.bconv = bconv

        # assign more global data to be shared across whole class
        self.data_prefix = data_prefix
        self.ig = ig
        self.dd = tkefields['dd']
        self.tke = tkefields['tke']

        self.nx = tkefields['nx']
        self.t_timec = tkefields['t_timec']
        self.t_tke = tkefields['t_tke']

        self.nsdim = nsdim

        self.super_ad_i = super_ad_i
        self.super_ad_o = super_ad_o
        self.outputFile = outputFile

    def plot_tke(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, yscTke):
        """Plot turbulent kinetic energy stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot 		
        plt1 = self.tke
        # plt2 = self.eht_tke

        # create FIGURE
        #plt.figure(figsize=(5, 4))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        #print(plt1, grd1)
        #print(xbl,xbr,ybu,ybd,xsc, yscTke)

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl/xsc, xbr/xsc, ybu/yscTke, ybd/yscTke, [yval / yscTke for yval in to_plot])

        # plot DATA 
        plt.title('turbulent kinetic energy')
        plt.plot(grd1/xsc, plt1/yscTke, color='brown', label=r"tke")
        # plt.plot(grd1, plt2, color='r', linestyle='--', label=r"$\frac{1}{2} \overline{u'_i u'_i}$")

        # convective boundary markers
        #plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        ycol = np.linspace(ybu / yscTke, ybd / yscTke, num=100)
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

        setylabel = r'ergs/g (' + "{:.0e}".format(yscTke) + ')'
        plt.ylabel(setylabel)


        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        #plt.show(block=False)

        # save PLOT
        #plt.savefig('RESULTS/' + self.data_prefix + 'mean_tke.png')

    def plot_tke_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg, xsc, yscTkeEq):
        """Plot turbulent kinetic energy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_tke
        lhs1 = self.minus_div_eht_dd_fht_ux_tke

        rhs0 = self.plus_wb
        rhs1 = self.plus_wp
        rhs2 = self.minus_div_fekx
        rhs3 = self.minus_div_fpx
        rhs4 = self.minus_r_grad_u

        res = self.minus_resTkeEquation

        rhs5 = self.minus_kolmrate * self.dd

        # create FIGURE
        #fig = plt.figure(figsize=(5, 4))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(LAXIS, xbl/xsc, xbr/xsc, ybu/yscTkeEq, ybd/yscTkeEq, [yval / yscTkeEq for yval in to_plot])

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # model constant for tke dissipation
        Cm = 1.

        # plot DATA
        plt.title(str(self.nsdim) + "D")

        if self.ig == 1:
            plt.plot(grd1/xsc, lhs0/yscTkeEq, color='#FF6EB4', label=r'-dt (rho*k)')
            plt.plot(grd1/xsc, lhs1/yscTkeEq, color='k', label=r"-div (rho*ux*k)")
            plt.plot(grd1/xsc, rhs0/yscTkeEq, color='r', label=r'+Wb')
            plt.plot(grd1/xsc, rhs1/yscTkeEq, color='c', label=r'+Wp')
            plt.plot(grd1/xsc, rhs2/yscTkeEq, color='#802A2A', label=r"-div fk")
            plt.plot(grd1/xsc, rhs3/yscTkeEq, color='m', label=r"-div fP")
            plt.plot(grd1/xsc, rhs4/yscTkeEq, color='b', label=r"-R*grad u")
            if self.nsdim !=2:
                # plt.plot(grd1/xsc, Cm * rhs5, color='k', linewidth=0.7, label=r"-C_m \overline{\rho} u^{'3}_{rms}/l_c")
                # plt.plot(grd1/xsc, Cm * rhs5/yscTkeEq, color='k', linewidth=0.7, label=r"-C_m \overline{\rho} u^{'3}_{rms}/l_d")
                pass
            plt.plot(grd1/xsc, res/yscTkeEq, color='k', linestyle='--', label=r"res")
        elif self.ig == 2:
            plt.plot(grd1/xsc, lhs0/yscTkeEq, color='#FF6EB4', label=r'-dt (rho*k)')
            plt.plot(grd1/xsc, lhs1/yscTkeEq, color='k', label=r"-div (rho*ux*k)")
            plt.plot(grd1/xsc, rhs0/yscTkeEq, color='r', label=r'+Wb')
            plt.plot(grd1/xsc, rhs1/yscTkeEq, color='c', label=r'+Wp')
            plt.plot(grd1/xsc, rhs2/yscTkeEq, color='#802A2A', label=r"-div fk")
            plt.plot(grd1/xsc, rhs3/yscTkeEq, color='m', label=r"-div fP")
            plt.plot(grd1/xsc, rhs4/yscTkeEq, color='b', label=r"-R*grad u")
            plt.plot(grd1/xsc, res/yscTkeEq, color='k', linestyle='--', label=r"res")

        # convective boundary markers
        # plt.axvline(self.bconv/xsc, linestyle='--', linewidth=0.7, color='k')
        # plt.axvline(self.tconv/xsc, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        #plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        #plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

        ycol = np.linspace(ybu / yscTkeEq, ybd / yscTkeEq, num=100)
        for i in ycol:
            plt.text(self.bconv/xsc,i, '.',dict(size=15))
            plt.text(self.tconv/xsc,i, '.',dict(size=15))

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)

        setylabel = r'erg/cm3/s (' + "{:.0e}".format(yscTkeEq) + ')'
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12}, ncol=2)

        # display PLOT
        # plt.show(block=False)

        # save PLOT
        # plt.savefig('RESULTS/' + self.data_prefix + 'tke_eq.png')
        # plt.savefig('RESULTS/' + self.data_prefix + 'tke_eq.eps')

        #return "success"

    def plot_tke_equation_integral_budget(self, ax, plabel, laxis, xbl, xbr, ybu, ybd, fc):
        """Plot integral budgets of tke equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        term1 = self.minus_dt_dd_tke
        term2 = self.minus_div_eht_dd_fht_ux_tke
        term3 = self.plus_wb
        term4 = self.plus_wp
        term5 = self.minus_div_fekx
        term6 = self.minus_div_fpx
        term7 = self.minus_r_grad_u
        term8 = self.minus_resTkeEquation

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
        term7_sel = term7[idxl:idxr]
        term8_sel = term8[idxl:idxr]

        rc = self.xzn0[idxl:idxr]

        # handle geometry
        Sr = 0.
        if self.ig == 1 and self.nsdim == 3:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.zzn0[-1] - self.zzn0[0])
        elif self.ig == 1 and self.nsdim == 2:
            Sr = (self.yzn0[-1] - self.yzn0[0]) * (self.yzn0[-1] - self.yzn0[0])
        elif self.ig == 2:
            Sr = 4. * np.pi * rc ** 2

        int_term1 = integrate.simps(term1_sel * Sr, rc)
        int_term2 = integrate.simps(term2_sel * Sr, rc)
        int_term3 = integrate.simps(term3_sel * Sr, rc)
        int_term4 = integrate.simps(term4_sel * Sr, rc)
        int_term5 = integrate.simps(term5_sel * Sr, rc)
        int_term6 = integrate.simps(term6_sel * Sr, rc)
        int_term7 = integrate.simps(term7_sel * Sr, rc)
        int_term8 = integrate.simps(term8_sel * Sr, rc)

        #fig = plt.figure(figsize=(7, 6))

        #ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd/fc, ybu/fc])

        # note the change: I'm only supplying y data.
        y = [int_term1 / fc, int_term2 / fc, int_term3 / fc, int_term4 / fc,
             int_term5 / fc, int_term6 / fc, int_term7 / fc, int_term8 / fc]

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
        ax.set_ylabel(r'ergs/s (' + "{:.0e}".format(fc) + ')')

        if self.nsdim != 2:
            ax.set_title(r"integral budget " + str(self.nsdim) + "D")
        else:
            ax.set_title(r"integral budget " + str(self.nsdim) + "D")

        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        if self.ig == 1:
            group_labels = [r'-dt (rho*k)',
                            r"-div (rho*ux*k)",
                            r'+Wb',r'+Wp',r"-div fk",r"-div fP",
                            r"-R*grad u",'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)
        elif self.ig == 2:
            group_labels = [r'-dt (rho*k)',
                            r"-div (rho*ux*k)",
                            r'+Wb',r'+Wp',r"-div fk",r"-div fP",
                            r"-R*grad u",'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)

        # auto-rotate the x axis labels
        #fig.autofmt_xdate()

        # display PLOT
        #plt.show(block=False)
        #plt.show()

        # save PLOT
        #plt.savefig('RESULTS/' + self.data_prefix + 'tke_eq_bar.png')
        #plt.savefig('RESULTS/' + self.data_prefix + 'tke_eq_bar.eps')