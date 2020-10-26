import numpy as np
import matplotlib.cm as cm
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XtransportEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, plabel, ig, fext, inuc, element, bconv, tconv, super_ad_i, super_ad_o, intc, nsdim, data_prefix):
        super(XtransportEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
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

        # calculate mass coordinate M
        #if plabel == 'oburn':
        #    coremass = 2.1061849325045916e33 # this is from PROMPI, subroutine starinit.f90
        #    msun = 1.989e33  # in grams
        #    MM = (coremass + mm)/msun  # convert to solar units
        #else:
        #    print("ERROR(XtransportEquation.py):" + self.errorProjectSpecific())
        #    print("ERROR(XtransportEquation.py): core mass not defined!")
        #    sys.exit()

        #self.fhtxineut = self.getRAdata(eht, 'ddx0001')[intc]/dd
        #self.fhtxiprot = self.getRAdata(eht, 'ddx0002')[intc]/dd
        #self.fhtxihe4 = self.getRAdata(eht, 'ddx0003')[intc]/dd
        #self.fhtxic12 = self.getRAdata(eht, 'ddx0004')[intc]/dd
        #self.fhtxio16 = self.getRAdata(eht, 'ddx0005')[intc]/dd
        #self.fhtxine20 = self.getRAdata(eht, 'ddx0006')[intc]/dd
        #self.fhtxina23 = self.getRAdata(eht, 'ddx0007')[intc]/dd
        #self.fhtximg24 = self.getRAdata(eht, 'ddx0008')[intc]/dd
        #self.fhtxisi28 = self.getRAdata(eht, 'ddx0009')[intc]/dd
        #self.fhtxip31 = self.getRAdata(eht, 'ddx0010')[intc]/dd
        #self.fhtxis32 = self.getRAdata(eht, 'ddx0011')[intc]/dd
        #self.fhtxis34 = self.getRAdata(eht, 'ddx0012')[intc]/dd
        #self.fhtxicl35 = self.getRAdata(eht, 'ddx0013')[intc]/dd
        #self.fhtxiar36 = self.getRAdata(eht, 'ddx0014')[intc]/dd

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nx = nx
        self.inuc = inuc
        self.element = element
        self.ddxi = ddxi
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

        if self.ig == 1:
            self.t_fht_xi = t_fht_xi
        elif self.ig == 2:
            dx = (xzn0[-1]-xzn0[0])/nx
            dumx = xzn0[0]+np.arange(1,nx,1)*dx
            t_fht_xi2 = []

            # interpolation due to non-equidistant radial grid
            for i in range(int(t_fht_xi.shape[0])):
                t_fht_xi2.append(np.interp(dumx,xzn0,t_fht_xi[i,:]))

            t_fht_xi_forspacetimediagram = np.asarray(t_fht_xi2)
            self.t_fht_xi = t_fht_xi_forspacetimediagram # for the space-time diagrams


    def plot_Xrho(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xrho stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ddxi

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title('rhoX for ' + element + " (netw: " + str(self.nnuc) + " ele)")
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{X}$')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + element + '.eps')

    def plot_X(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
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

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title('X for ' + element)
        plt.semilogy(grd1, plt1, color='brown', label=r'$\widetilde{X}$')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

        # this is an inset axes over the main axes
        # a = plt.axes([.27, .4, .44, 0.23])
        # plt.plot(grd1, plt1, color='r')
        # plt.setp(a, xlim=[xbl, xbr], ylim=[ybu, ybd], xticks=[], yticks=[])
        #        setp(a)

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\widetilde{X}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\widetilde{X}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # this is another inset axes over the main axes
        plt.rc('font', size=12.)
        a = plt.axes([0.26, 0.55, .3, .2])
        plt.plot(grd1[0:64], plt1[0:64], color='r')
        # plt.xticks([])
        # plt.yticks([])

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            plt.xlabel(setxlabel)

        plt.rc('font', size=16.)

        # plt.text(5.2, 0.6, r"$\sim 10$x", fontsize=25, color='k')

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_' + element + '.eps')

    def plot_X_with_MM(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
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

        fig, ax1 = plt.subplots(figsize=(7, 6))

        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        ax1.semilogy(grd1, plt1, color='brown', label=self.setNucNoUp(str(element)))

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\widetilde{X}$"
            ax1.set_xlabel(setxlabel)
            ax1.set_ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\widetilde{X}$" + " (netw: " + str(self.nnuc) + " ele)"
            ax1.set_xlabel(setxlabel)
            ax1.set_ylabel(setylabel)

        # show LEGEND
        ax1.legend(loc=ilg, prop={'size': 18})

        # convective boundary markers
        ax1.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        ax1.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        newMMlabel_xpos = [3.8e8, 4.7e8, 5.5e8, 6.4e8, 7.3e8, 8.5e8, 9.5e8]
        newMMlabel = self.mlabels(newMMlabel_xpos)
        ax2.set_xticks(newMMlabel_xpos)

        ax2.set_xticklabels(newMMlabel)
        ax2.set_xlabel('enclosed mass (msol)')

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_withMM' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_withMM' + element + '.eps')


    def plot_Xm_with_MM(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot X stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot

        plt0 = self.fhtxineut
        plt1 = self.fhtxiprot
        plt2 = self.fhtxihe4
        plt3 = self.fhtxic12
        plt4 = self.fhtxio16
        plt5 = self.fhtxine20
        plt6 = self.fhtxina23
        plt7 = self.fhtximg24
        plt8 = self.fhtxisi28
        plt9 = self.fhtxip31
        plt10 = self.fhtxis32
        plt11 = self.fhtxis34
        plt12 = self.fhtxicl35
        plt13 = self.fhtxiar36

        fig, ax1 = plt.subplots(figsize=(7, 6))

        to_plot = [plt0,plt1,plt2,plt3,plt4,plt5,plt6,plt7,plt8,plt9,plt10,plt11,plt12,plt13]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        #num_plots = 14

        # Have a look at the colormaps here and decide which one you'd like:
        # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        # colormap = plt.cm.gist_ncar
        #plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.gist_ncar(np.linspace(0, 1, num_plots))))

        # plot DATA
        #ax1.semilogy(grd1, plt0, label=r"neut")
        #ax1.semilogy(grd1, plt1, label=r"$^{1}$H")
        #ax1.semilogy(grd1, plt2, label=r"$^{4}$He")
        #ax1.semilogy(grd1, plt3, label=r"$^{12}$C")
        #ax1.semilogy(grd1, plt4, label=r"$^{16}$O")
        #ax1.semilogy(grd1, plt5, label=r"$^{20}$Ne")
        #ax1.semilogy(grd1, plt6, label=r"$^{23}$Na")

        ax1.semilogy(grd1, plt7, label=r"$^{24}$Mg")
        ax1.semilogy(grd1, plt8, label=r"$^{28}$Si")
        ax1.semilogy(grd1, plt9, label=r"$^{31}$P")
        ax1.semilogy(grd1, plt10, label=r"$^{32}$S")
        ax1.semilogy(grd1, plt11, label=r"$^{34}$S")
        ax1.semilogy(grd1, plt12, label=r"$^{35}$Cl")
        ax1.semilogy(grd1, plt13, label=r"$^{36}$Ar")

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\widetilde{X}$"
            ax1.set_xlabel(setxlabel)
            ax1.set_ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\widetilde{X}$"
            ax1.set_xlabel(setxlabel)
            ax1.set_ylabel(setylabel)

        # show LEGEND
        ax1.legend(loc=3, prop={'size': 14}, ncol =2)

        self.bconv = 0.44e9
        self.tconv = 0.71e9

        # convective boundary markers
        ax1.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        ax1.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        newMMlabel_xpos = [3.8e8, 4.7e8, 5.5e8, 6.4e8, 7.3e8, 8.5e8, 9.5e8]
        newMMlabel = self.mlabels(newMMlabel_xpos)
        ax2.set_xticks(newMMlabel_xpos)

        ax2.set_xticklabels(newMMlabel)
        ax2.set_xlabel('enclosed mass (msol)')

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xm_withMM_ini.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xm_withMM2_ini.eps')

    def plot_X_space_time(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot X stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        t_timec = self.t_timec

        # load x GRID
        nx = self.nx
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = np.log10(self.t_fht_xi).T
        #plt1 = self.t_fht_xi.T

        indRES = np.where((grd1 < 9.e8) & (grd1 > 4.e8))[0]

        pltMax = np.max(plt1[indRES])
        pltMin = np.min(plt1[indRES])

        pltMax = -0.2
        pltMin = -6.

        # create FIGURE
        # plt.figure(figsize=(7, 6))

        #print(t_timec[0], t_timec[-1], grd1[0], grd1[-1])

        fig, ax = plt.subplots(figsize=(14, 7))
        # fig.suptitle("log(X) (" + self.setNucNoUp(str(element))+ ")")
        fig.suptitle("X (512x512x1) (" + element + ")")

        im = ax.imshow(plt1, interpolation='bilinear', cmap=cm.jet,
                       origin='lower', extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]], aspect='auto',
                       vmax=pltMax, vmin=pltMin)

        #extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]]

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
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_space_time_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_space_time_' + element + '.eps')


    def plot_rhoX_space_time(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot X stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        t_timec = self.t_timec

        # load x GRID
        nx = self.nx
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.t_ddxi.T

        indRES = np.where((grd1 < 8.1e8) & (grd1 > 4.55e8))[0]

        pltMax = np.max(plt1[indRES])
        pltMin = np.min(plt1[indRES])

        # create FIGURE
        # plt.figure(figsize=(7, 6))

        fig, ax = plt.subplots(figsize=(14, 7))
        # fig.suptitle("rhoX (" + self.setNucNoUp(str(element)) + ") (g cm-3)")
        fig.suptitle("rhoX (" + element + ")")
        im = ax.imshow(plt1, interpolation='bilinear', cmap=cm.autumn,
                       origin='lower', extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]], aspect='auto',
                       vmax=pltMax, vmin=pltMin)

        #extent = [t_timec[0], t_timec[-1], grd1[0], grd1[-1]]

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
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_space_time_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_space_time_' + element + '.eps')

    def plot_gradX(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot grad X stratification in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.plus_gradx_fht_xi
        plt2 = self.minus_cnst_dd_fht_xi_fdil_o_fht_rxx

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        to_plot = [plt1, plt2]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title('X for ' + element)
        plt.plot(grd1, plt1, color='brown', label=r'$\partial_r \widetilde{X}$')
        plt.plot(grd1, plt2, color='r', label=r'$.$')

        # convective boundary markers
        plt.axvline(self.bconv + 0.46e8, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"$\partial_r \widetilde{X}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"$\partial_r \widetilde{X}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_gradX_' + element + '.png')

    def plot_Xtransport_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
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
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries
        to_plot = [lhs0, lhs1, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title(r"rhoX transport for " + str(element) + " (netw: " + str(self.nnuc) + " ele) " + str(self.nsdim) + "D")

        if self.ig == 1:
            plt.plot(grd1, lhs0, color='r', label=r'$-\partial_t (\overline{\rho} \widetilde{X})$')
            plt.plot(grd1, lhs1, color='cyan', label=r'$-\nabla_x (\overline{\rho} \widetilde{X} \widetilde{u}_x)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f$')
            plt.plot(grd1, rhs1, color='g', label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='r', label=r'$-\partial_t (\overline{\rho} \widetilde{X})$')
            plt.plot(grd1, lhs1, color='cyan', label=r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_r f$')
            plt.plot(grd1, rhs1, color='g', label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        plt.axvline(self.super_ad_i, linestyle=':', linewidth=0.7, color='k')
        plt.axvline(self.super_ad_o, linestyle=':', linewidth=0.7, color='k')

        # shade area one
        #ind = np.where((grd1 < 5.17e8) & (grd1 > 4.53e8))[0]

        #rinc = grd1[ind[0]]
        #routc = grd1[ind[-1]]

        #il = ind[0]
        #ir = ind[-1]

        #plt.fill([rinc, routc, routc, rinc], [ybd, ybd, ybu, ybu], 'y', edgecolor='w')

        # shade area two
        #ind = np.where((grd1 < 4.53e8) & (grd1 > self.bconv))[0]

        #rinc = grd1[ind[0]]
        #routc = grd1[ind[-1]]

        #il = ind[0]
        #ir = ind[-1]

        #plt.fill([rinc, routc, routc, rinc], [ybd, ybd, ybu, ybu], 'gold', edgecolor='w')

        # shade area three
        #ind = np.where((grd1 < 5.8e8) & (grd1 > 5.17e8))[0]

        #rinc = grd1[ind[0]]
        #routc = grd1[ind[-1]]

        #il = ind[0]
        #ir = ind[-1]

        #plt.fill([rinc, routc, routc, rinc], [ybd, ybd, ybu, ybu], 'chartreuse', edgecolor='w')

        # shade area four
        #ind = np.where((grd1 < 7.5e8) & (grd1 > 5.8e8))[0]

        #rinc = grd1[ind[0]]
        #routc = grd1[ind[-1]]

        #il = ind[0]
        #ir = ind[-1]

        #plt.fill([rinc, routc, routc, rinc], [ybd, ybd, ybu, ybu], '#CFCFCF', edgecolor='w')


        #xzn0 = np.asarray(self.xzn0)
        #idx = np.where(rhs1 == rhs1.min())[0]

        # ne20 burn min
        #plt.axvline(x=5.7e8, color='k', linewidth=1, linestyle='dotted')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 13},ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.eps')

    def plot_Xtransport_equation_integral_budget(self, laxis, xbl, xbr, ybu, ybd):
        """Plot integral budgets of composition transport equation in the model"""

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

        Sr = 4. * np.pi * rc ** 2

        int_term1 = integrate.simps(term1_sel * Sr, rc)
        int_term2 = integrate.simps(term2_sel * Sr, rc)
        int_term3 = integrate.simps(term3_sel * Sr, rc)
        int_term4 = integrate.simps(term4_sel * Sr, rc)
        int_term5 = integrate.simps(term5_sel * Sr, rc)

        fig = plt.figure(figsize=(7, 6))

        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:
            plt.ylim([ybd, ybu])

        fc = 1.

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
        ax.set_ylabel(r'g s$^{-1}$')

        # Create a title, in italics
        ax.set_title('rhoX transport budget for ' + element)

        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        group_labels = [r'$-\partial_t (\overline{\rho} \widetilde{X})$',
                        r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$',
                        r'$-\nabla_r f$', r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$', 'res']

        # Set the x tick labels to the group_labels defined above.
        ax.set_xticklabels(group_labels, fontsize=16)

        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        fig.autofmt_xdate()

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'xtransport_' + element + '_eq_bar.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'xtransport_' + element + '_eq_bar.eps')


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