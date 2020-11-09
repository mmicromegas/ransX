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

class TemperatureGradients(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, ieos, intc, data_prefix):
        super(TemperatureGradients, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf  

        pp = self.getRAdata(eht, 'pp')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        mu = self.getRAdata(eht, 'abar')[intc]
        chim = self.getRAdata(eht, 'chim')[intc]
        chit = self.getRAdata(eht, 'chit')[intc]
        gamma2 = self.getRAdata(eht, 'gamma2')[intc]
        # print(chim,chit,gamma2)

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma2 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients		
        nabla = self.deriv(lntt, lnpp)
        nabla_ad = (gamma2 - 1.) / gamma2

        if ieos == 1:
            nabla_mu = np.zeros(nx)
        else:
            nabla_mu = (chim / chit) * self.deriv(lnmu, lnpp)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nx = nx
        self.nabla = nabla
        self.nabla_ad = nabla_ad
        self.nabla_mu = nabla_mu
        self.ig = ig
        self.ieos = ieos
        self.fext = fext
        self.abar = self.getRAdata(eht, 'abar')[intc]

    def plot_nablas(self, LAXIS, bconv, tconv, super_ad_i, super_ad_o, xbl, xbr, ybu, ybd, ilg):
        """Plot temperature gradients in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureGradients.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.nabla
        plt2 = self.nabla_ad
        plt3 = self.nabla_mu

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('temperature gradients')

        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'$\nabla$')
            plt.plot(grd1, plt2, color='red', label=r'$\nabla_{ad}$')
            if self.ieos == 3:
                plt.plot(grd1, plt3, color='green', label=r'$\nabla_{\mu}$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'$\nabla$')
            plt.plot(grd1, plt2, color='red', label=r'$\nabla_{ad}$')
            if self.ieos == 3:
                plt.plot(grd1, plt3, color='green', label=r'$\nabla_{\mu}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # convective boundary markers - only super-adiatic regions
        plt.axvline(super_ad_i, linestyle=':', linewidth=0.7, color='k')
        plt.axvline(super_ad_o, linestyle=':', linewidth=0.7, color='k')


        # this is another inset axes over the main axes
        #plt.rc('font', size=12.)
        #a = plt.axes([0.24, 0.25, .3, .2])

        #ilft = 0
        #irgt = 64
        #plt.plot(grd1[ilft:irgt], self.abar[ilft:irgt], color='purple')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\nabla$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\nabla$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_nablas.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_nablas.eps')

    def plot_nablas2(self, LAXIS, bconv, tconv, super_ad_i, super_ad_o, xbl, xbr, ybu, ybd, ilg):
        """Plot temperature gradients in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(TemperatureGradients.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        to_plt1 = self.nabla
        to_plt2 = self.nabla_ad
        to_plt3 = self.abar

        # load x GRID
        grd1 = self.xzn0

        if self.ig == 1:
            xlabel_1 = r'x (cm)'
        elif self.ig == 2:
            xlabel_1 = r'r (cm)'

        ylabel_1 = r'$\nabla$'
        ylabel_3 = r"$\overline{\mu}$"

        plabel_1 = r'$\nabla$'
        plabel_2 = r'$\nabla_{ad}$'
        plabel_3 = r'$\overline{\mu}$'

        # calculate indices of grid boundaries
        xzn0 = np.asarray(self.xzn0)
        xlm = np.abs(xzn0 - xbl)
        xrm = np.abs(xzn0 - xbr)
        idxl = int(np.where(xlm == xlm.min())[0][0])
        idxr = int(np.where(xrm == xrm.min())[0][0])

        # create FIGURE
        fig, ax1 = plt.subplots(figsize=(7, 6))

        ax1.axis([xbl, xbr, ybd, ybu])
        ax1.plot(xzn0, to_plt1, color='brown', label=plabel_1)
        ax1.plot(xzn0, to_plt2, color='red', label=plabel_2)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7, prop={'size': 18})

        ax2 = ax1.twinx()
        ax2.axis([xbl, xbr, 1.79, 1.86])
        ax2.plot(xzn0, to_plt3, color='g', label=plabel_3)
        ax2.set_ylabel(ylabel_3)
        ax2.tick_params('y')
        ax2.legend(loc=1, prop={'size': 18})

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_nablas2.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_nablas2.eps')
