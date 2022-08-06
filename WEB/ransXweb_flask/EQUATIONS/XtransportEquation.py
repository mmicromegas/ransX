import numpy as np
import matplotlib.cm as cm
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XtransportEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, plabel, ig, fext, inuc, element, bconv, tconv, super_ad_i, super_ad_o, intc, nsdim, data_prefix):
        super(XtransportEquation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')
        nx = self.getRAdata(eht, 'nx')
        nnuc = self.getRAdata(eht, 'nnuc')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = self.getRAdata(eht, 'dd')[intc]
        mm = self.getRAdata(eht, 'mm')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]

        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
        gamma1 = self.getRAdata(eht, 'gamma1')[intc]
        gamma3 = self.getRAdata(eht, 'gamma3')[intc]

        uxdivu = self.getRAdata(eht, 'ux')[intc]
        gamma1 = self.getRAdata(eht, 'ux')[intc]
        gamma3 = self.getRAdata(eht, 'ux')[intc]

        fht_rxx = dduxux - ddux * ddux / dd
        fdil = (uxdivu - ux * divu)

        #######################
        # Xi TRANSPORT EQUATION
        #######################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_fht_xi = t_ddxi / t_dd

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        fxi = ddxiux - ddxi * ddux / dd

        # LHS -dq/dt
        self.minus_dt_dd_fht_xi = -self.dt(t_dd * t_fht_xi, xzn0, t_timec, intc)

        # LHS -div(ddXiux)
        self.minus_div_eht_dd_fht_ux_fht_xi = -self.Div(dd * fht_ux * fht_xi, xzn0)

        # RHS -div fxi
        self.minus_div_fxi = -self.Div(fxi, xzn0)

        # RHS +ddXidot
        self.plus_ddxidot = +ddxidot

        # -res
        self.minus_resXiTransport = -(self.minus_dt_dd_fht_xi + self.minus_div_eht_dd_fht_ux_fht_xi + self.minus_div_fxi + self.plus_ddxidot)

        ###########################
        # END Xi TRANSPORT EQUATION
        ###########################

        # grad models
        self.plus_gradx_fht_xi = +self.Grad(fht_xi, xzn0)
        cnst = gamma1
        self.minus_cnst_dd_fht_xi_fdil_o_fht_rxx = -cnst * dd * fht_xi * fdil / fht_rxx

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0

        self.nx = nx
        self.inuc = inuc
        self.element = element
        self.ddxi = ddxi
        #self.ddxi2 = self.getRAdata(eht, 'ddx0002')[intc]
        #self.fht_xi2 = self.getRAdata(eht, 'ddx0002')[intc]/self.getRAdata(eht, 'dd')[intc]

        self.fht_xi = fht_xi

        self.bconv = bconv
        self.tconv = tconv
        self.super_ad_i = super_ad_i
        self.super_ad_o = super_ad_o

        self.ig = ig
        self.fext = fext
        self.mm = mm
        self.t_timec = t_timec
        self.t_fht_xi = t_fht_xi
        self.t_ddxi = t_ddxi
        self.ddxidot = ddxidot
        self.nnuc = nnuc
        self.nsdim = nsdim

    def plot_X(self, LAXIS, xbl, xbr, ybu, ybd, ilg, xsc, yscX):
        """Plot X stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.fht_xi
        #plt2 = self.fht_xi2

        # create FIGURE
        #plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl/xsc, xbr/xsc, ybu/yscX, ybd/yscX, [yval / yscX for yval in to_plot])

        # plot DATA
        plt.title("mass fraction" + " (" + element + ")")
        #plt.title('mass fraction')
        plt.semilogy(grd1/xsc, plt1/yscX, color='brown', label=r'X')

        ycol = np.linspace(ybu / yscX, ybd / yscX, num=100)
        for i in ycol:
            plt.text(self.bconv/xsc,i, '.',dict(size=15))
            plt.text(self.tconv/xsc,i, '.',dict(size=15))

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + 'cm)'
            setylabel = r"X"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + 'cm)'
            setylabel = r"X"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        #plt.rc('font', size=16.)

        # save PLOT
        #if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_' + element + '.png')
        #if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_' + element + '.eps')

    def plot_Xtransport_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg, xsc, yscXeq):
        """Plot Xrho transport equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_fht_xi
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_xi

        rhs0 = self.minus_div_fxi
        rhs1 = self.plus_ddxidot

        res = self.minus_resXiTransport

        # create FIGURE
        #plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        to_plot = [lhs0, lhs1, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl/xsc, xbr/xsc, ybu/yscXeq, ybd/yscXeq, [yval / yscXeq for yval in to_plot])

        # plot DATA
        plt.title(str(self.nsdim) + "D" + " (" + element + ")")

        if self.ig == 1:
            plt.plot(grd1/xsc, lhs0/yscXeq, color='r', label=r'-dt rhoX')
            plt.plot(grd1/xsc, lhs1/yscXeq, color='cyan', label=r'-div rhoXux')
            plt.plot(grd1/xsc, rhs0/yscXeq, color='b', label=r'-div f')
            plt.plot(grd1/xsc, rhs1/yscXeq, color='g', label=r'+rhoXnuc')
            plt.plot(grd1/xsc, res/yscXeq, color='k', linestyle='--', label='res')
            plt.plot(grd1/xsc, np.zeros(self.nx), color='k', linestyle='dotted')
        elif self.ig == 2:
            plt.plot(grd1/xsc, lhs0/yscXeq, color='r', label=r'-dt rhoX')
            plt.plot(grd1/xsc, lhs1/yscXeq, color='cyan', label=r'-div rhoXux')
            plt.plot(grd1/xsc, rhs0/yscXeq, color='b', label=r'-div f')
            plt.plot(grd1/xsc, rhs1/yscXeq, color='g', label=r'+rhoXnuc')
            plt.plot(grd1/xsc, res/yscXeq, color='k', linestyle='--', label='res')
            plt.plot(grd1/xsc, np.zeros(self.nx), color='k', linestyle='dotted')

        # convective boundary markers
        #plt.axvline(self.bconv/xsc, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(self.tconv/xsc, linestyle='--', linewidth=0.7, color='k')

        ycol = np.linspace(ybu / yscXeq, ybd / yscXeq, num=100)
        for i in ycol:
            plt.text(self.bconv/xsc,i, '.',dict(size=15))
            plt.text(self.tconv/xsc,i, '.',dict(size=15))

        # convective boundary markers - only super-adiatic regions
        # plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        # plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (' + "{:.0e}".format(xsc) + 'cm)'
            plt.xlabel(setxlabel)

        setylabel = r'g/cm3/s (' + "{:.0e}".format(yscXeq) + ')'
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 13},ncol=1)

        # display PLOT
        #plt.show(block=False)

        # save PLOT
        #if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.png')
        #if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.eps')

    def plot_Xtransport_equation_integral_budget(self, ax, laxis, xbl, xbr, ybu, ybd, fc):
        """Plot integral budgets of composition transport equation in the model"""

        #print(xbl,xbr,ybu,ybd,fc)

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        element = self.element

        # load x GRID
        grd1 = self.xzn0
        nx = self.nx

        term1 = self.minus_dt_dd_fht_xi
        term2 = self.minus_div_eht_dd_fht_ux_fht_xi
        term3 = self.minus_div_fxi
        term4 = self.plus_ddxidot
        term5 = self.minus_resXiTransport


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

        #fig = plt.figure(figsize=(7, 6))

        #ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd /fc, ybu/ fc])

        #fc = 1.

        # note the change: I'm only supplying y data.
        y = [int_term1 / fc, int_term2 / fc, int_term3 / fc, int_term4 / fc, int_term5 / fc]

        #print(y)

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

        # ax.set_title(r"rhoX transport budget for " + str(element) + str(self.nsdim) + "D")
        ax.set_title(r"transport budget")
        #ax.set_title('rhoX transport budget for ' + element)

        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        # group_labels = [r'$-\partial_t (\overline{\rho} \widetilde{X})$',
        #                r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$',
        #                r'$-\nabla_r f$', r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$', 'res']

        group_labels = [r'-dt rhoX',r'-div rhoXux',r'-div f', r'+rhoXnuc', 'res']

        # Set the x tick labels to the group_labels defined above.
        ax.set_xticklabels(group_labels, fontsize=16)

        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        #fig.autofmt_xdate()

        # display PLOT
        #plt.show(block=False)

        # save PLOT
        #if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'xtransport_' + element + '_eq_bar.png')
        #if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'xtransport_' + element + '_eq_bar.eps')


    def mlabels(self, grid):
        # calculate MM labels
        xzn0 = np.asarray(self.xzn0)
        msun = 1.989e33  # in grams
        M_label = []
        for grd in grid:
            xlm = np.abs(xzn0 - grd)
            idx = int(np.where(xlm == xlm.min())[0][0])
            M_label.append(str(np.round(self.mm[idx]/msun,1)))

        return M_label

    def setNucNoUp(self, inpt):
        elmnt = ""
        if inpt == "neut":
            elmnt = r"neut"
        if inpt == "prot":
            elmnt = r"prot"
        if inpt == "he4":
            elmnt = r"He$^{4}$"
        if inpt == "c12":
            elmnt = r"C$^{12}$"
        if inpt == "o16":
            elmnt = r"O$^{16}$"
        if inpt == "ne20":
            elmnt = r"Ne$^{20}$"
        if inpt == "na23":
            elmnt = r"Na$^{23}$"
        if inpt == "mg24":
            elmnt = r"Mg$^{24}$"
        if inpt == "si28":
            elmnt = r"Si$^{28}$"
        if inpt == "p31":
            elmnt = r"P$^{31}$"
        if inpt == "s32":
            elmnt = r"S$^{32}$"
        if inpt == "s34":
            elmnt = r"S$^{34}$"
        if inpt == "cl35":
            elmnt = r"Cl$^{35}$"
        if inpt == "ar36":
            elmnt = r"Ar$^{36}$"
        if inpt == "ar38":
            elmnt = r"Ar$^{38}$"
        if inpt == "k39":
            elmnt = r"K$^{39}$"
        if inpt == "ca40":
            elmnt = r"Ca$^{40}$"
        if inpt == "ca42":
            elmnt = r"Ca$^{42}$"
        if inpt == "ti44":
            elmnt = r"Ti$^{44}$"
        if inpt == "ti46":
            elmnt = r"Ti$^{46}$"
        if inpt == "cr48":
            elmnt = r"Cr$^{48}$"
        if inpt == "cr50":
            elmnt = r"Cr$^{50}$"
        if inpt == "fe52":
            elmnt = r"Fe$^{52}$"
        if inpt == "fe54":
            elmnt = r"Fe$^{54}$"
        if inpt == "ni56":
            elmnt = r"Ni$^{56}$"

        return elmnt