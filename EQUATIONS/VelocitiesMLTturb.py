import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import os
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class VelocitiesMLTturb(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, nsdim, data_prefix):
        super(VelocitiesMLTturb, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht,'xzn0')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        ux = self.getRAdata(eht,'ux')[intc]
        dd = self.getRAdata(eht,'dd')[intc]
        tt = self.getRAdata(eht,'tt')[intc]
        hh = self.getRAdata(eht,'hh')[intc]
        cp = self.getRAdata(eht,'cp')[intc]
        gg = self.getRAdata(eht,'gg')[intc]
        pp = self.getRAdata(eht,'pp')[intc]
        chit = self.getRAdata(eht,'chit')[intc]
        chid = self.getRAdata(eht,'chid')[intc]
        ddux = self.getRAdata(eht,'ddux')[intc]
        dduxux = self.getRAdata(eht,'dduxux')[intc]
        ddtt = self.getRAdata(eht,'ddtt')[intc]
        ddhh = self.getRAdata(eht,'ddhh')[intc]
        ddcp = self.getRAdata(eht,'ddcp')[intc]
        ddhhux = self.getRAdata(eht,'ddhhux')[intc]
        hhux = self.getRAdata(eht,'hhux')[intc]
        ttsq = self.getRAdata(eht,'ttsq')[intc]
        ddttsq = self.getRAdata(eht,'ddttsq')[intc]
        gamma2 = self.getRAdata(eht,'gamma2')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht,'cp')[intc]
            cv = self.getRAdata(eht,'cv')[intc]
            gamma2 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        # store time series for time derivatives
        t_timec = self.getRAdata(eht,'timec')
        t_mm = self.getRAdata(eht,'mm')

        minus_dt_mm = -self.dt(t_mm, xzn0, t_timec, intc)

        vexp1 = ddux / dd
        vexp2 = minus_dt_mm / (4. * np.pi * (xzn0 ** 2.) * dd)
        vturb = ((dduxux - ddux * ddux / dd) / dd) ** 0.5

        fht_cp = ddcp / dd

        # variance of temperature fluctuations		
        # sigmatt = (ddttsq-ddtt*ddtt/dd)/dd
        sigmatt = ttsq - tt * tt

        # T_rms fluctuations
        tt_rms = sigmatt ** 0.5

        # enthalpy flux 
        fhh = ddhhux - ddhh * ddux / dd
        # fhh = dd*(hhux - hh*ux)

        # mlt velocity		
        alphae = 0.2  # Meakin,Arnett,2007
        vmlt_1 = fhh / (alphae * dd * fht_cp * tt_rms)

        Hp = 2.e8  # this is for oburn
        alpha_mlt = 1.7
        lbd = alpha_mlt * Hp

        lntt = np.log(tt)
        lnpp = np.log(pp)

        # calculate temperature gradients		
        nabla = self.deriv(lntt, lnpp)
        nabla_ad = (gamma2 - 1.) / gamma2

        if ieos == 1:
            betaT = 0.
        elif ieos == 3:
            betaT = -chit / chid
        else:
            print("ERROR(BruntVaisalla.py): " + self.errorEos(ieos))
            sys.exit()

        vmlt_2 = gg * betaT * (nabla - nabla_ad) * ((lbd ** 2.) / (8. * Hp))
        vmlt_2 = vmlt_2.clip(min=1.)  # get rid of negative values, set to min 1.
        vmlt_2 = (vmlt_2) ** 0.5

        # this should be OS independent
        dir_model = os.path.join(os.path.realpath('.'), 'DATA_D', 'INIMODEL', 'imodel.tycho')

        data = np.loadtxt(dir_model, skiprows=26)
        nxmax = 500

        rr = data[1:nxmax, 2]
        vmlt_3 = data[1:nxmax, 8]

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ig = ig
        self.ux = ux
        self.vexp1 = vexp1
        self.vexp2 = vexp2
        self.vturb = vturb
        self.vmlt_1 = vmlt_1
        self.vmlt_2 = vmlt_2

        self.rr = rr
        self.vmlt_3 = vmlt_3
        self.fext = fext
        self.nsdim = nsdim

    def plot_velocities(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot velocities in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(VelocitiesMLTturb.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ux
        plt2 = self.vexp1
        plt3 = self.vexp2
        plt4 = self.vturb
        plt5 = self.vmlt_1 # vmlt_1 = fhh / (alphae * dd * fht_cp * tt_rms)  - REFERENCE NEEDED
        plt6 = self.vmlt_2 # vmlt_2 = gg * betaT * (nabla - nabla_ad) * ((lbd ** 2.) / (8. * Hp)) - REFERENCE NEEDED
        plt7 = self.vmlt_3 # THIS IS FROM TYCHO's initial model

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # temporary hack
        plt4 = np.nan_to_num(plt4)
        plt5 = np.nan_to_num(plt5)
        plt6 = np.nan_to_num(plt6)
        plt7 = np.nan_to_num(plt7)

        # set plot boundaries   
        to_plot = [plt4, plt5, plt6, plt7]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('velocities ' + str(self.nsdim) + "D")
        # plt.plot(grd1,plt1,color='brown',label = r'$\overline{u}_r$')
        # plt.plot(grd1,plt2,color='red',label = r'$\widetilde{u}_r$')
        # plt.plot(grd1,plt3,color='green',linestyle='--',label = r'$\overline{v}_{exp} = -\dot{M}/(4 \pi r^2 \rho)$')
        plt.plot(grd1, plt4, color='blue', label=r'$u_{turb}$')

        plt.plot(grd1,plt5,color='red',label = r'$u_{MLT} 1$')
        # plt.plot(grd1,plt6,color='g',label = r'$u_{MLT} 2$')
        # plt.plot(self.rr,plt7,color='brown',label = r'$u_{MLT} 3 inimod$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"velocity (cm s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"velocity (cm s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_velocities_turb.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_velocities_turb.eps')