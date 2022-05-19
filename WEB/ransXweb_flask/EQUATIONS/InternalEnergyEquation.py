import numpy as np
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors
import sys
from scipy import integrate


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class InternalEnergyEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, fext, intc, nsdim, tke_diss, data_prefix):
        super(InternalEnergyEquation, self).__init__(ig)

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

        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddei = self.getRAdata(eht, 'ddei')[intc]
        ddeiux = self.getRAdata(eht, 'ddeiux')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddei = self.getRAdata(eht, 'ddei')
        t_fht_ei = t_ddei / t_dd

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ei = ddei / dd
        fei = ddeiux - ddux * ddei / dd

        ##########################
        # INTERNAL ENERGY EQUATION 
        ##########################

        # LHS -dq/dt 		
        self.minus_dt_dd_fht_ei = -self.dt(t_dd * t_fht_ei, xzn0, t_timec, intc)

        # LHS -div dd fht_ux fht_ei		
        self.minus_div_dd_fht_ux_fht_ei = -self.Div(dd * fht_ux * fht_ei, xzn0)

        # RHS -div fei
        self.minus_div_fei = -self.Div(fei, xzn0)

        # RHS -div ftt (not included) heat flux
        self.minus_div_ftt = -np.zeros(nx)

        # RHS -P d = - pp Div ux
        self.minus_pp_div_ux = -pp * self.Div(ux, xzn0)

        # RHS -Wp = -eht_ppf_df
        self.minus_eht_ppf_df = -(ppdivu - pp * divu)

        # RHS source + dd enuc
        self.plus_dd_fht_enuc = ddenuc1 + ddenuc2

        # RHS dissipated turbulent kinetic energy
        self.plus_disstke = +tke_diss

        # -res
        self.minus_resEiEquation = -(self.minus_dt_dd_fht_ei + self.minus_div_dd_fht_ux_fht_ei +
                                     self.minus_div_fei + self.minus_div_ftt + self.minus_pp_div_ux + self.minus_eht_ppf_df +
                                     self.plus_dd_fht_enuc + self.plus_disstke)

        ##############################
        # END INTERNAL ENERGY EQUATION 
        ##############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0

        self.fht_ei = fht_ei
        self.fext = fext
        self.nsdim = nsdim

    def plot_ei(self, laxis, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
        """Plot mean Favrian internal energy stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(InternalEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fht_ei

        # create FIGURE
        # plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(laxis, xbl/xsc, xbr/xsc, ybu/ysc, ybd/ysc, [yval / ysc for yval in to_plot])

        # plot DATA 
        plt.title(r'internal energy')
        plt.plot(grd1/xsc, plt1/ysc, color='brown', label=r'ei')

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

        setylabel = r'ei (' + "{:.0e}".format(ysc) + ' ergs/g)'
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        # plt.show(block=False)

        # save PLOT
        # if self.fext == 'png':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_ei.png')
        # elif self.fext == 'eps':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_ei.eps')

    def plot_ei_equation(self, laxis, bconv, tconv, xbl, xbr, ybu, ybd, ilg, xsc, ysc):
        """Plot internal energy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(InternalEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_fht_ei
        lhs1 = self.minus_div_dd_fht_ux_fht_ei

        rhs0 = self.minus_div_fei
        rhs1 = self.minus_div_ftt
        rhs2 = self.minus_pp_div_ux
        rhs3 = self.minus_eht_ppf_df
        rhs4 = self.plus_dd_fht_enuc
        rhs5 = self.plus_disstke

        res = self.minus_resEiEquation

        # create FIGURE
        # plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, rhs3, rhs4, rhs5, res]
        self.set_plt_axis(laxis, xbl/xsc, xbr/xsc, ybu/ysc, ybd/ysc, [yval / ysc for yval in to_plot])

        # plot DATA 
        plt.title('internal energy equation')
        if self.ig == 1:
            plt.plot(grd1/xsc, lhs0/ysc, color='#FF6EB4', label=r"-dt rho ei")
            plt.plot(grd1/xsc, lhs1/ysc, color='k',label=r"-div rho u ei")

            plt.plot(grd1/xsc, rhs0/ysc, color='#FF8C00', label=r"-div fI")
            plt.plot(grd1/xsc, rhs1/ysc, color='c', label=r"-div fT (not incl.)")
            plt.plot(grd1/xsc, rhs2/ysc, color='#802A2A', label=r"-P d")
            plt.plot(grd1/xsc, rhs3/ysc, color='r', label=r"-W_P")
            plt.plot(grd1/xsc, rhs4/ysc, color='b', label=r"+rho enuc")
            plt.plot(grd1/xsc, rhs5/ysc, color='m', label=r"+epsilonK")

            plt.plot(grd1/xsc, res/ysc, color='k', linestyle='--', label=r"res")
        elif self.ig == 2:
            plt.plot(grd1/xsc, lhs0/ysc, color='#FF6EB4', label=r"-dt rho ei")
            plt.plot(grd1/xsc, lhs1/ysc, color='k',label=r"-div rho u ei")

            plt.plot(grd1/xsc, rhs0/ysc, color='#FF8C00', label=r"-div fI")
            plt.plot(grd1/xsc, rhs1/ysc, color='c', label=r"-div fT (not incl.)")
            plt.plot(grd1/xsc, rhs2/ysc, color='#802A2A', label=r"-P d")
            plt.plot(grd1/xsc, rhs3/ysc, color='r', label=r"-W_P")
            plt.plot(grd1/xsc, rhs4/ysc, color='b', label=r"+rho enuc")
            plt.plot(grd1/xsc, rhs5/ysc, color='m', label=r"+epsilonK")

            plt.plot(grd1/xsc, res/ysc, color='k', linestyle='--', label=r"res")

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

        setylabel = r'erg/cm3/s (' + "{:.0e}".format(ysc) + ')'
        plt.ylabel(setylabel)


        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10}, ncol=2)

        # display PLOT
        # plt.show(block=False)

        # save PLOT
        # if self.fext == 'png':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'ei_eq.png')
        # elif self.fext == 'eps':
        #    plt.savefig('RESULTS/' + self.data_prefix + 'ei_eq.eps')

    def plot_ei_equation_integral_budget(self, ax, plabel, laxis, xbl, xbr, ybu, ybd, fc):
        """Plot integral budgets of tke equation in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        term1 = self.minus_dt_dd_fht_ei
        term2 = self.minus_div_dd_fht_ux_fht_ei
        term3 = self.minus_div_fei
        term4 = self.minus_div_ftt
        term5 = self.minus_pp_div_ux
        term6 = self.minus_eht_ppf_df
        term7 = self.plus_dd_fht_enuc
        term8 = self.plus_disstke
        term9 = self.minus_resEiEquation

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
        term9_sel = term9[idxl:idxr]

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
        int_term9 = integrate.simps(term9_sel * Sr, rc)

        # fig = plt.figure(figsize=(7, 6))

        # ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd / fc, ybu / fc])

        # note the change: I'm only supplying y data.
        y = [int_term1 / fc, int_term2 / fc, int_term3 / fc, int_term4 / fc,
             int_term5 / fc, int_term6 / fc, int_term7 / fc, int_term8 / fc, int_term9 / fc]

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
            group_labels = [r'-dt rho ei',
                            r"-div rho u ei",
                            r'-div fI', r'-div fT', r"-P d", r"-W_P",
                            r"+rho enuc", '+epsilonK', 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)
        elif self.ig == 2:
            group_labels = [r'-dt rho ei',
                            r"-div rho u ei",
                            r'-div fI', r'-div fT', r"-P d", r"-W_P",
                            r"+rho enuc", '+epsilonK', 'res']

            # Set the x tick labels to the group_labels defined above.
            ax.set_xticklabels(group_labels, fontsize=10)

        # auto-rotate the x axis labels
        # fig.autofmt_xdate()

        # display PLOT
        # plt.show(block=False)
        # plt.show()

        # save PLOT
        # plt.savefig('RESULTS/' + self.data_prefix + 'tke_eq_bar.png')
        # plt.savefig('RESULTS/' + self.data_prefix + 'tke_eq_bar.eps')
