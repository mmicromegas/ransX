import numpy as np
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class DivuDilatation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, data_prefix, bconv, tconv):
        super(DivuDilatation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')
        ny = self.getRAdata(eht, 'ny')
        nz = self.getRAdata(eht, 'nz')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]

        ddgg = -self.getRAdata(eht, 'ddgg')[intc]
        gamma1 = self.getRAdata(eht, 'gamma1')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uxuy = self.getRAdata(eht, 'uxuy')[intc]
        uxuz = self.getRAdata(eht, 'uxuz')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduxuy = self.getRAdata(eht, 'dduxuy')[intc]
        dduxuz = self.getRAdata(eht, 'dduxuz')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        dddivu = self.getRAdata(eht, 'dddivu')[intc]

        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        uydivu = self.getRAdata(eht, 'uydivu')[intc]
        uzdivu = self.getRAdata(eht, 'uzdivu')[intc]

        uxdivux = self.getRAdata(eht, 'uxdivux')[intc]
        uydivux = self.getRAdata(eht, 'uydivux')[intc]
        uzdivux = self.getRAdata(eht, 'uzdivux')[intc]

        uxdivuy = self.getRAdata(eht, 'uxdivuy')[intc]
        uydivuy = self.getRAdata(eht, 'uydivuy')[intc]
        uzdivuy = self.getRAdata(eht, 'uzdivuy')[intc]

        uxdivuz = self.getRAdata(eht, 'uxdivuz')[intc]
        uydivuz = self.getRAdata(eht, 'uydivuz')[intc]
        uzdivuz = self.getRAdata(eht, 'uzdivuz')[intc]

        divux = self.getRAdata(eht, 'divux')[intc]
        divuy = self.getRAdata(eht, 'divuy')[intc]
        divuz = self.getRAdata(eht, 'divuz')[intc]

        dduxdivu = self.getRAdata(eht, 'dduxdivu')[intc]
        dduydivu = self.getRAdata(eht, 'dduydivu')[intc]
        dduzdivu = self.getRAdata(eht, 'dduzdivu')[intc]

        dduxdivux = self.getRAdata(eht, 'dduxdivux')[intc]
        dduydivux = self.getRAdata(eht, 'dduydivux')[intc]
        dduzdivux = self.getRAdata(eht, 'dduzdivux')[intc]

        dduxdivuy = self.getRAdata(eht, 'dduxdivuy')[intc]
        dduydivuy = self.getRAdata(eht, 'dduydivuy')[intc]
        dduzdivuy = self.getRAdata(eht, 'dduzdivuy')[intc]

        dduxdivuz = self.getRAdata(eht, 'dduxdivuz')[intc]
        dduydivuz = self.getRAdata(eht, 'dduydivuz')[intc]
        dduzdivuz = self.getRAdata(eht, 'dduzdivuz')[intc]

        dddivux = self.getRAdata(eht, 'dddivux')[intc]
        dddivuy = self.getRAdata(eht, 'dddivuy')[intc]
        dddivuz = self.getRAdata(eht, 'dddivuz')[intc]

        dduxuxx = self.getRAdata(eht, 'dduxuxx')[intc]
        dduyuxx = self.getRAdata(eht, 'dduyuxx')[intc]
        dduzuxx = self.getRAdata(eht, 'dduzuxx')[intc]

        dduxuyy = self.getRAdata(eht, 'dduxuyy')[intc]
        dduyuyy = self.getRAdata(eht, 'dduyuyy')[intc]
        dduzuyy = self.getRAdata(eht, 'dduzuyy')[intc]

        dduxuzz = self.getRAdata(eht, 'dduxuzz')[intc]
        dduyuzz = self.getRAdata(eht, 'dduyuzz')[intc]
        dduzuzz = self.getRAdata(eht, 'dduzuzz')[intc]

        # dduxx = self.getRAdata(eht,'dduxx')[intc]
        # dduyy = self.getRAdata(eht,'dduyy')[intc]
        # dduzz = self.getRAdata(eht,'dduzz')[intc]

        uxuxx = self.getRAdata(eht, 'uxuxx')[intc]
        uyuxx = self.getRAdata(eht, 'uyuxx')[intc]
        uzuxx = self.getRAdata(eht, 'uzuxx')[intc]

        uxuyy = self.getRAdata(eht, 'uxuyy')[intc]
        uyuyy = self.getRAdata(eht, 'uyuyy')[intc]
        uzuyy = self.getRAdata(eht, 'uzuyy')[intc]

        uxuzz = self.getRAdata(eht, 'uxuzz')[intc]
        uyuzz = self.getRAdata(eht, 'uyuzz')[intc]
        uzuzz = self.getRAdata(eht, 'uzuzz')[intc]

        uxx = self.getRAdata(eht, 'uxx')[intc]
        uyy = self.getRAdata(eht, 'uyy')[intc]
        uzz = self.getRAdata(eht, 'uzz')[intc]

        pp = self.getRAdata(eht, 'pp')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
        dddivu = self.getRAdata(eht, 'dddivu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ppux = self.getRAdata(eht, 'ppux')[intc]
        uxdivu = self.getRAdata(eht, 'uxdivu')[intc]
        uxppdivu = self.getRAdata(eht, 'uxppdivu')[intc]

        ppuy = self.getRAdata(eht, 'ppuy')[intc]
        uydivu = self.getRAdata(eht, 'uydivu')[intc]
        uyppdivu = self.getRAdata(eht, 'uyppdivu')[intc]

        ppuz = self.getRAdata(eht, 'ppuz')[intc]
        uzdivu = self.getRAdata(eht, 'uzdivu')[intc]
        uzppdivu = self.getRAdata(eht, 'uzppdivu')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd

        eht_uxff = ux - fht_ux

        ###########################################
        # FULL TURBULENCE VELOCITY FIELD HYPOTHESIS
        ###########################################

        if (True):
            self.rxx = uxux - ux * ux
            self.rxy = uxuy - ux * uy
            self.rxz = uxuz - ux * uz

        if (False):
            self.rxx = dduxux / dd - ddux * ddux / (dd * dd)
            self.ryx = dduxuy / dd - ddux * dduy / (dd * dd)
            self.rzx = dduxuz / dd - ddux * dduz / (dd * dd)

        self.eht_uxf_divuf = uxdivu - ux * divu
        self.eht_uyf_divuf = uydivu - uy * divu
        self.eht_uzf_divuf = uzdivu - uz * divu

        self.eht_uxf_divuxf = uxdivux - ux * divux
        self.eht_uxf_divuyf = uxdivuy - ux * divuy
        self.eht_uxf_divuzf = uxdivuz - ux * divuz

        self.eht_uyf_divuxf = uydivux - uy * divux
        self.eht_uyf_divuyf = uydivuy - uy * divuy
        self.eht_uyf_divuzf = uydivuz - uy * divuz

        self.eht_uzf_divuxf = uzdivux - uz * divux
        self.eht_uzf_divuyf = uzdivuy - uz * divuy
        self.eht_uzf_divuzf = uzdivuz - uz * divuz

        self.eht_uxff_divuff = dduxdivu / dd - ddux * dddivu / (dd * dd)
        self.eht_uyff_divuff = dduydivu / dd - dduy * dddivu / (dd * dd)
        self.eht_uzff_divuff = dduzdivu / dd - dduz * dddivu / (dd * dd)

        self.eht_uxff_divuxff = dduxdivux / dd - ddux * dddivux / (dd * dd)
        self.eht_uxff_divuyff = dduxdivuy / dd - ddux * dddivuy / (dd * dd)
        self.eht_uxff_divuzff = dduxdivuz / dd - ddux * dddivuz / (dd * dd)

        self.eht_uyff_divuxff = dduydivux / dd - dduy * dddivux / (dd * dd)
        self.eht_uyff_divuyff = dduydivuy / dd - dduy * dddivuy / (dd * dd)
        self.eht_uyff_divuzff = dduydivuz / dd - dduy * dddivuz / (dd * dd)

        self.eht_uzff_divuxff = dduzdivux / dd - dduz * dddivux / (dd * dd)
        self.eht_uzff_divuyff = dduzdivuy / dd - dduz * dddivuy / (dd * dd)
        self.eht_uzff_divuzff = dduzdivuz / dd - dduz * dddivuz / (dd * dd)

        self.eht_uxf_uxxf = uxuxx - ux * uxx
        self.eht_uxf_uyyf = uxuyy - ux * uyy
        self.eht_uxf_uzzf = uxuzz - ux * uzz

        self.eht_uyf_uxxf = uyuxx - uy * uxx
        self.eht_uyf_uyyf = uyuyy - uy * uyy
        self.eht_uyf_uzzf = uyuzz - uy * uzz

        self.eht_uzf_uxxf = uzuxx - uz * uxx
        self.eht_uzf_uyyf = uzuyy - uz * uyy
        self.eht_uzf_uzzf = uzuzz - uz * uzz

        self.eht_uxff_divuff = dduxdivu / dd - ddux * dddivu / (dd * dd)
        self.eht_uyff_divuff = dduydivu / dd - dduy * dddivu / (dd * dd)
        self.eht_uzff_divuff = dduzdivu / dd - dduz * dddivu / (dd * dd)

        # self.eht_uxff_uxxff  = dduxuxx/dd - ddux*dduxx/(dd*dd)
        # self.eht_uxff_uyyff  = dduxuyy/dd - ddux*dduyy/(dd*dd)
        # self.eht_uxff_uzzff  = dduxuzz/dd - ddux*dduzz/(dd*dd)

        # self.eht_uyff_uxxff  = dduyuxx/dd - dduy*dduxx/(dd*dd)
        # self.eht_uyff_uyyff  = dduyuyy/dd - dduy*dduyy/(dd*dd)
        # self.eht_uyff_uzzff  = dduyuzz/dd - dduy*dduzz/(dd*dd)

        # self.eht_uzff_uxxff  = dduzuxx/dd - dduz*dduxx/(dd*dd)
        # self.eht_uzff_uyyff  = dduzuyy/dd - dduz*dduyy/(dd*dd)
        # self.eht_uzff_uzzff  = dduzuzz/dd - dduz*dduzz/(dd*dd)

        ###############################################
        # END FULL TURBULENCE VELOCITY FIELD HYPOTHESIS
        ###############################################

        self.eht_uxfppdivu = uxppdivu - ux * ppdivu
        self.ppfuxf_fht_divu = (ppux - pp * ux) * (dddivu / dd)
        self.pp_eht_uxf_divuff = pp * (uxdivu - ux * divu)
        self.eht_ppf_uxf_divuff = ppux * divu - ppux * (dddivu / dd) - pp * ux * divu + pp * ux * (dddivu / dd)

        self.eht_uyfppdivu = uyppdivu - uy * ppdivu
        self.ppfuyf_fht_divu = (ppuy - pp * uy) * (dddivu / dd)
        self.pp_eht_uyf_divuff = pp * (uydivu - uy * divu)
        self.eht_ppf_uyf_divuff = ppuy * divu - ppuy * (dddivu / dd) - pp * uy * divu + pp * uy * (dddivu / dd)

        self.eht_uzfppdivu = uzppdivu - uz * ppdivu
        self.ppfuzf_fht_divu = (ppuz - pp * uz) * (dddivu / dd)
        self.pp_eht_uzf_divuff = pp * (uzdivu - uz * divu)
        self.eht_ppf_uzf_divuff = ppuz * divu - ppuz * (dddivu / dd) - pp * uz * divu + pp * uz * (dddivu / dd)
        self.eht_divu1 = divu
        self.eht_divu2 = divux + divuy + divuz
        #print(divux)
        #print('************')
        #print(divuy)
        #print('************')
        #print(divuz)
        #print('************')

        self.fht_divu = dddivu/dd
        self.eht_divuff = self.Div(eht_uxff,xzn0)

        eht_ux = ux
        fht_ux = ddux/dd

        self.favrian_d = self.Div(fht_ux, xzn0)
        self.reynolds_d = self.Div(eht_ux, xzn0)

        # for space-time diagrams
        t_divu = self.getRAdata(eht, 'divu')
        t_timec = self.getRAdata(eht, 'timec')
        if self.ig == 1:
            self.t_divu = t_divu
        elif self.ig == 2:
            dx = (xzn0[-1]-xzn0[0])/nx
            dumx = xzn0[0]+np.arange(1,nx,1)*dx
            t_divu2 = []

            # interpolation due to non-equidistant radial grid
            for i in range(int(t_divu.shape[0])):
                t_divu2.append(np.interp(dumx,xzn0,t_divu[i,:]))

            t_divu_forspacetimediagram = np.asarray(t_divu2)
            self.t_divu = t_divu_forspacetimediagram # for the space-time diagrams


        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dd = dd
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.ig = ig

        self.pp = pp
        self.ddgg = ddgg
        self.gamma1 = gamma1
        self.fext = fext

        self.bconv = bconv
        self.tconv = tconv

        self.t_timec = t_timec

    def plot_divu(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot divu in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(DivuDilatation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.eht_divu1
        plt2 = self.eht_divu2
        plt3 = self.favrian_d
        plt4 = self.reynolds_d
        plt5 = self.fht_divu
        plt6 = self.eht_divuff

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt6]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        if self.ig == 1:
            plt.title('divu (cartesian)')
            #plt.plot(grd1, plt2, color='g', label=r"$divu2$")
            plt.plot(grd1, plt1, marker='o', color='r',markersize=6,markevery=20, label=r"+$\overline{\nabla \cdot {\bf u}}$")
            plt.plot(grd1, plt4, color='b', label=r"+$\nabla_x \overline{u}_x$")
            plt.plot(grd1, plt3, color='g', label=r"+$\nabla_x \widetilde{u}_x$")
            # plt.plot(grd1, plt5, color='m', linestyle='dotted',label=r"+$\overline{\rho \nabla \cdot {\bf u}}/\overline{\rho}$")
            plt.plot(grd1, plt6, color='c', label=r"+$\nabla_x \overline{u''}_x$")
        elif self.ig == 2:
            plt.title('divu (spherical)')
            plt.plot(grd1, plt1, color='r', label=r"$divu1$")
            plt.plot(grd1, plt2, color='g', label=r"$divu2$")

        plt.axhline(y=0., linestyle='--',color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"cm s$^{-2}$"
            plt.ylabel(setylabel)
            plt.xlabel(setxlabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"cm s$^{-2}$"
            plt.ylabel(setylabel)
            plt.xlabel(setxlabel)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # check supported file output extension
        if self.fext != "png" and self.fext != "eps":
            print("ERROR(DivuDilatation.py):" + self.errorOutputFileExtension(self.fext))
            sys.exit()

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'divu.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'divu.eps')

    def plot_divu_space_time(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot Frho space time diagram"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithMassFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

        t_timec = self.t_timec

        # load x GRID
        nx = self.nx
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.t_divu.T
        #plt1 = self.t_divu.T

        indRES = np.where((grd1 < 9.e8) & (grd1 > 4.e8))[0]

        #pltMax = np.max(plt1[indRES])
        #pltMin = np.min(plt1[indRES])

        pltMax = 5.e-5
        pltMin = -5.e-5

        # create FIGURE
        # plt.figure(figsize=(7, 6))

        #print(t_timec[0], t_timec[-1], grd1[0], grd1[-1])

        fig, ax = plt.subplots(figsize=(14, 7))
        # fig.suptitle("log(X) (" + self.setNucNoUp(str(element))+ ")")
        fig.suptitle(r"$\nabla \cdot {\bf u}$ " + str(self.nx) + ' x ' + str(self.ny) + ' x ' + str(self.nz))

        im = ax.imshow(plt1, interpolation='bilinear', cmap=cm.jet.reversed(),
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
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_divu_space_time' +'.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_divu_space_time' + '.eps')
