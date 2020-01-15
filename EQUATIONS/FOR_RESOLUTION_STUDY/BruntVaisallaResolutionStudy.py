import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class BruntVaisallaResolutionStudy(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, data_prefix):
        super(BruntVaisallaResolutionStudy, self).__init__(ig)

        # load data to list of structured arrays
        eht = []
        for file in filename:
            eht.append(np.load(file))

        # declare data lists
        dd, pp, gg, gamma1, dlnrhodr, dlnpdr, dlnrhodrs, nsq, chim, chit, chid, mu, tt, gamma2, \
        alpha, delta, phi, hp, lntt, lnpp, lnmu, nabla, nabla_ad, nabla_mu, nsq_version2, xzn0 = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        nx, ny, nz = [], [], []

        for i in range(len(filename)):
            # load grid
            xzn0.append(np.asarray(eht[i].item().get('xzn0')))

            nx.append(np.asarray(eht[i].item().get('nx')))
            ny.append(np.asarray(eht[i].item().get('ny')))
            nz.append(np.asarray(eht[i].item().get('nz')))

            # pick specific Reynolds-averaged mean fields according to:
            # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf 		

            dd.append(np.asarray(eht[i].item().get('dd')[intc]))
            pp.append(np.asarray(eht[i].item().get('pp')[intc]))
            gg.append(np.asarray(eht[i].item().get('gg')[intc]))

            # override gamma for ideal gas eos (need to be fixed in PROMPI later)
            if ieos == 1:
                cp = self.getRAdata(eht[i], 'cp')[intc]
                cv = self.getRAdata(eht[i], 'cv')[intc]
                gamma1.append(cp / cv)  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
                gamma2.append(cp / cv)  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110)
            else:
                gamma1.append(np.asarray(eht[i].item().get('gamma1')[intc]))
                gamma2.append(np.asarray(eht[i].item().get('gamma2')[intc]))

            dlnrhodr.append(self.deriv(np.log(dd[i]), xzn0[i]))
            dlnpdr.append(self.deriv(np.log(pp[i]), xzn0[i]))
            dlnrhodrs.append((1. / gamma1[i]) * dlnpdr[i])
            nsq.append(gg[i] * (dlnrhodr[i] - dlnrhodrs[i]))

            chim.append(np.asarray(eht[i].item().get('chim')[intc]))
            chit.append(np.asarray(eht[i].item().get('chit')[intc]))
            chid.append(np.asarray(eht[i].item().get('chid')[intc]))
            mu.append(np.asarray(eht[i].item().get('abar')[intc]))
            tt.append(np.asarray(eht[i].item().get('tt')[intc]))

            alpha.append(1. / chid[i])
            delta.append(-chit[i] / chid[i])
            phi.append(chid[i] / chim[i])
            hp.append(-pp[i] / self.Grad(pp[i], xzn0[i]))

            lntt.append(np.log(tt[i]))
            lnpp.append(np.log(pp[i]))
            lnmu.append(np.log(mu[i]))

            # calculate temperature gradients		
            nabla.append(self.deriv(lntt[i], lnpp[i]))
            nabla_ad.append((gamma2[i] - 1.) / gamma2[i])
            nabla_mu.append((chim[i] / chit[i]) * self.deriv(lnmu[i], lnpp[i]))

            # Kippenhahn and Weigert, p.42 but with opposite (minus) sign at the (phi/delta)*nabla_mu
            nsq_version2.append(
                (gg[i] * delta[i] / hp[i]) * (nabla_ad[i] - nabla[i] - (phi[i] / delta[i]) * nabla_mu[i]))

        # share data globally 
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nsq = nsq
        self.nsq_version2 = nsq_version2
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def plot_bruntvaisalla(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot BruntVaisalla parameter in the model"""

        if (LAXIS != 2):
            print("ERROR(BruntVaisallaResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        plt1 = self.nsq
        nx = self.nx
        ny = self.ny
        nz = self.nz

        # find maximum resolution data
        grd_maxres = self.maxresdata(grd)
        plt1_maxres = self.maxresdata(plt1)

        plt_interp = []
        for i in range(len(grd)):
            plt_interp.append(np.interp(grd_maxres, grd[i], plt1[i]))

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        plt10_tmp = plt1[0]
        plt11_tmp = plt1[0]

        plt1_foraxislimit = []
        plt1max = np.max(plt1[0])
        for plt1i in plt1:
            if (np.max(plt1i) > plt1max):
                plt1_foraxislimit = plt1i

        # set plot boundaries
        to_plot = [plt1_foraxislimit]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Brunt-Vaisalla frequency')

        for i in range(len(grd)):
            plt.plot(grd[i], plt1[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))

        # define and show x/y LABELS
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

    # find data with maximum resolution	
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
