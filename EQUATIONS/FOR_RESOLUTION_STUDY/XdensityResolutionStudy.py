import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al
import UTILS.Tools as uT
import UTILS.Errors as eR


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XdensityResolutionStudy(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, inuc, element, intc, data_prefix):
        super(XdensityResolutionStudy, self).__init__(ig)

        # load data to list of structured arrays
        eht = []
        for file in filename:
            eht.append(np.load(file))

        # declare data lists		
        xzn0, nx, ny, nz = [], [], [], []

        dd, ddxi, fht_xi = [], [], []

        for i in range(len(filename)):
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

    def plot_X(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mass fraction in the model"""

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        fht_xi = self.fht_xi
        nx = self.nx
        ny = self.ny
        nz = self.nz

        # find maximum resolution data		
        grd_maxres = self.maxresdata(grd)
        nsq_maxres = self.maxresdata(fht_xi)

        plt_interp = []
        for i in range(len(grd)):
            plt_interp.append(np.interp(grd_maxres, grd[i], fht_xi[i]))

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('X for ' + self.element)

        for i in range(len(grd)):
            plt.plot(grd[i], fht_xi[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$X_i$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_X.png')

    def plot_Xrho(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot fractional density in the model"""

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        ddxi = self.ddxi
        nx = self.nx
        ny = self.ny
        nz = self.nz

        # find maximum resolution data		
        grd_maxres = self.maxresdata(grd)
        nsq_maxres = self.maxresdata(ddxi)

        plt_interp = []
        for i in range(len(grd)):
            plt_interp.append(np.interp(grd_maxres, grd[i], ddxi[i]))

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('Xrho for ' + self.element)

        for i in range(len(grd)):
            plt.plot(grd[i], ddxi[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho X_i}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + self.element + '.png')

    # find data with maximum resolution	
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
