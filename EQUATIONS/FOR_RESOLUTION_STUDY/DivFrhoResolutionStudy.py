import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class DivFrhoResolutionStudy(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(DivFrhoResolutionStudy, self).__init__(ig)

        # load data to list of structured arrays
        eht = []
        for ffile in filename:
            eht.append(self.customLoad(ffile))

        # declare data lists		
        xzn0, nx, ny, nz, tavg = [], [], [], [], []

        divfrho = []

        for i in range(len(filename)):
            # load grid
            xzn0.append(np.asarray(eht[i].item().get('xzn0')))

            nx.append(np.asarray(eht[i].item().get('nx')))
            ny.append(np.asarray(eht[i].item().get('ny')))
            nz.append(np.asarray(eht[i].item().get('nz')))
            tavg.append(np.asarray(eht[i].item().get('tavg')))

            # pick specific Reynolds-averaged mean fields according to:
            # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf 		

            divfrho.append(self.Div((np.asarray(eht[i].item().get('ddux')[intc])-
                           np.asarray(eht[i].item().get('dd')[intc])*np.asarray(eht[i].item().get('ux')[intc])),np.asarray(eht[i].item().get('xzn0'))))

        # share data globally
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.divfrho = divfrho
        self.ig = ig
        self.tavg = tavg

    def plot_divfrho(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot div TurbulentMass flux in the model"""

        if (LAXIS != 2):
            print("ERROR(DivFrhoResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        plt1 = self.divfrho
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
        plt.title(r"-Div $\overline{\rho' u'_x}$ ")

        for i in range(len(grd)):
            plt.plot(grd[i], -1.*plt1[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i])+ ' '+'(tavg = ' + str(np.int(self.tavg[i])) +' s)')

        #bbndry = grd[0][289]
        #tbndry = grd[0][316]

        #plt.text(tbndry,0.8e2,r"$\sim$0.23 Hp")

        # convective boundary
        #plt.axvline(bbndry, linestyle='--', linewidth=0.7, color='k')
        #plt.axvline(tbndry, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$-\nabla_x \overline{\rho' u'_x}}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$-\nabla_r \overline{\rho' u'_r}}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        plt.axhline(y=0., linestyle='--',color='k')

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_divfrho.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_divfrho.eps')

    # find data with maximum resolution	
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
