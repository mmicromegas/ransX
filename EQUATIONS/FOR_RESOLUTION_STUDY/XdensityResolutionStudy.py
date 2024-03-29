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

class XdensityResolutionStudy(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, inuc, element, intc, data_prefix):
        super(XdensityResolutionStudy, self).__init__(ig)

        # load data to list of structured arrays
        eht = []
        for file in filename:
            eht.append(self.customLoad(file))

        # declare data lists		
        xzn0, nx, ny, nz = [], [], [], []

        timec, dd, ddxi, fht_xi = [], [], [], []

        for i in range(len(filename)):
            # load time
            timec.append(np.asarray(eht[i].item().get('timec')[intc]))
            # load grid
            xzn0.append(np.asarray(eht[i].item().get('xzn0')))

            nx.append(np.asarray(eht[i].item().get('nx')))
            ny.append(np.asarray(eht[i].item().get('ny')))
            nz.append(np.asarray(eht[i].item().get('nz')))

            dd.append(np.asarray(eht[i].item().get('dd')[intc]))
            ddxi.append(np.asarray(eht[i].item().get('ddx' + inuc)[intc]))

            fht_xi.append(ddxi[i] / dd[i])

        # share data globally
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.inuc = inuc
        self.element = element
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.fht_xi = fht_xi
        self.ddxi = ddxi
        self.ig = ig
        self.timec = timec

    def plot_Xrho(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot fractional density in the model"""

        if (LAXIS != 2):
            print("ERROR(XdensityResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        plt1 = self.ddxi
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
        plt.title('Xrho for ' + self.element)

        plt.plot(grd[0], plt1[0], linestyle='--',label='initial')

        for i in range(1,len(grd)):
            #plt.plot(grd[i], plt1[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]) +
            #                                ' t: ' + str(np.around(self.timec[i]).astype(int)) + ' s')
            plt.plot(grd[i], plt1[i], label=str(np.around(self.timec[i]).astype(int)) + ' s')


        # define and show x/y LABELS

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\overline{\rho X_i}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\overline{\rho X_i}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + self.element + '.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + self.element + '.eps')

    # find data with maximum resolution	
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
