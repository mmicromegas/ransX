import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class FullTurbulenceVelocityFieldHypothesisY(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, data_prefix, bconv, tconv):
        super(FullTurbulenceVelocityFieldHypothesisY, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]

        pp = self.getRAdata(eht, 'pp')[intc]
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

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.dd = dd
        self.nx = nx
        self.ig = ig

        self.pp = pp
        self.ddgg = ddgg
        self.gamma1 = gamma1
        self.fext = fext

        self.bconv = bconv
        self.tconv = tconv

    def plot_ftvfhY_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot ftvfh in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(FullTurbulenceVelocityFieldHypothesisY.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.eht_uyf_uxxf
        plt2 = self.eht_uyf_uyyf
        plt3 = self.eht_uyf_uzzf
        plt4 = -self.ddgg * self.rxy / (self.gamma1 * self.pp)
        res = plt1 + plt2 + plt3 + plt4

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3, plt4]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        if self.ig == 1:
            plt.title(r"turbulence velocity field hypothesis Y")
            plt.plot(grd1, plt1, color='r', label=r"$+\overline{u'_y \nabla_x u'_x}$")
            plt.plot(grd1, plt2, color='g', label=r"$+\overline{u'_y \nabla_y u'_y}$")
            plt.plot(grd1, plt3, color='b', label=r"$+\overline{u'_y \nabla_z u'_z}$")
            plt.plot(grd1, plt4, color='m',
                     label=r"$-\overline{\rho} \ \overline{u'_y u'_x} \ \widetilde{g}_x/\Gamma_1 \ \overline{P}$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"$res$")
        elif self.ig == 2:
            plt.title(r"turbulence velocity field hypothesis $\theta$")
            plt.plot(grd1, plt1, color='r', label=r"$+\overline{u'_\theta \nabla_r u'_r}$")
            plt.plot(grd1, plt2, color='g', label=r"$+\overline{u'_\theta \nabla_\theta u'_\theta}$")
            plt.plot(grd1, plt3, color='b', label=r"$+\overline{u'_\theta \nabla_\phi u'_\phi}$")
            plt.plot(grd1, plt4, color='m',
                     label=r"$-\overline{\rho} \ \overline{u'_\theta u'_r} \ \widetilde{g}_r/\Gamma_1 \ \overline{P}$")
            plt.plot(grd1, res, color='k', linestyle='--', label=r"$res$")

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
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # check supported file output extension
        if self.fext != "png" and self.fext != "eps":
            print("ERROR(FullTurbulenceVelocityFieldHypothesisY.py):" + self.errorOutputFileExtension(self.fext))
            sys.exit()

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'full_turb_velY_field_hypothesis.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'full_turb_velY_field_hypothesis.eps')
