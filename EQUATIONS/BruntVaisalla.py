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

class BruntVaisalla(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, data_prefix):
        super(BruntVaisalla, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf 

        dd = self.getRAdata(eht, 'dd')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        gg = self.getRAdata(eht, 'gg')[intc]
        gamma1 = self.getRAdata(eht, 'gamma1')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        dlnrhodr = self.deriv(np.log(dd), xzn0)
        dlnpdr = self.deriv(np.log(pp), xzn0)
        dlnrhodrs = (1. / gamma1) * dlnpdr
        nsq = gg * (dlnrhodr - dlnrhodrs)

        chim = self.getRAdata(eht, 'chim')[intc]
        chit = self.getRAdata(eht, 'chit')[intc]
        chid = self.getRAdata(eht, 'chid')[intc]
        mu = self.getRAdata(eht, 'abar')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        gamma2 = self.getRAdata(eht, 'gamma2')[intc]

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma2 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            alpha = 0.
            delta = 0.
            phi = 0.
        elif ieos == 3:
            alpha = 1. / chid
            delta = -chit / chid
            phi = chid / chim
        else:
            print("ERROR(BruntVaisalla.py): " + self.errorEos(ieos))
            sys.exit()

        hp = -pp / self.Grad(pp, xzn0)

        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients	

        if ieos == 1:
            nabla = self.deriv(lntt, lnpp)
            nabla_ad = (gamma2 - 1.) / gamma2
            nabla_mu = np.zeros(nx)
            # Kippenhahn and Weigert, p.42
            self.nsq_version2 = (gg * delta / hp) * (nabla_ad - nabla)
        elif ieos == 3:
            nabla = self.deriv(lntt, lnpp)
            nabla_ad = (gamma2 - 1.) / gamma2
            nabla_mu = (chim / chit) * self.deriv(lnmu, lnpp)
            # Kippenhahn and Weigert, p.42 but with opposite (minus) sign at the (phi/delta)*nabla_mu
            self.nsq_version2 = (gg * delta / hp) * (nabla_ad - nabla - (phi / delta) * nabla_mu)
        else:
            print("ERROR(BruntVaisalla.py): " + self.errorEos(ieos))
            sys.exit()

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nsq = nsq
        self.ig = ig
        self.nx = nx

    def plot_bruntvaisalla(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot BruntVaisalla parameter in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(BruntVaisalla.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.nsq
        plt2 = self.nsq_version2

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Brunt-Vaisalla frequency')
        if self.ig == 1:
            plt.plot(grd1, plt1, color='r', label=r'N$^2$')
            plt.plot(grd1, np.zeros(self.nx), linestyle='--', color='k')
            # plt.plot(grd1,plt2,color='b',linestyle='--',label = r'N$^2$ version 2')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='r', label=r'N$^2$')
            plt.plot(grd1, np.zeros(self.nx), linestyle='--', color='k')
            # plt.plot(grd1,plt2,color='b',linestyle='--',label = r'N$^2$ version 2')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"N$^2$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"N$^2$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_BruntVaisalla.png')
