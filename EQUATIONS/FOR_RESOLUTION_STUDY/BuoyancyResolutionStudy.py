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

class BuoyancyResolutionStudy(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, ieos, intc, data_prefix):
        super(BuoyancyResolutionStudy, self).__init__(ig)

        # load data to list of structured arrays
        eht = []
        for ffile in filename:
            eht.append(np.load(ffile))

        # declare data lists		
        xzn0, nx, ny, nz, xznr, xznl = [], [], [], [], [], []

        dd, pp, gg, gamma1, gamma2 = [], [], [], [], []

        dlnrhodr, dlnpdr, dlnrhodrs, nsq, br, dx = [], [], [], [], [], []

        for i in range(len(filename)):
            # load grid
            xzn0.append(np.asarray(eht[i].item().get('xzn0')))
            xznl.append(np.asarray(eht[i].item().get('xznl')))
            xznr.append(np.asarray(eht[i].item().get('xznr')))

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

            dx.append(xznr[i] - xznl[i])

        b = []

        # print(nsq[0],nx[0],int(nx[0]))

        for i in range(len(filename)):
            br = np.zeros(int(nx[i]))
            for ii in range(0, int(nx[i])):
                nsqf = nsq[i]
                dxf = dx[i]
                br[ii] = br[ii - 1] + nsqf[ii] * dxf[ii]
                # print(i,ii)

            b.append(br)


        # share data globally
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.b = b
        self.ig = ig

    def plot_buoyancy(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot buoyancy in the model"""

        if (LAXIS != 2):
            print("ERROR(BuoyancyResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        plt1 = self.b
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
        plt.title('Buoyancy')

        for i in range(len(grd)):
            plt.plot(grd[i], plt1[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))

        print("[WARNING] (BuoyancyResolutionStudy.py): convective boundary markers taken from 256c run, tavg = 1500 secs")
        # taken from 256cubed, tavg 1500 sec
        bconv = 4.1e8
        tconv = 9.7e8
        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$buoyancy$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$buoyancy$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_buoyancy.png')

    # find data with maximum resolution	
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
