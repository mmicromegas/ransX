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

class TurbulentMassFluxResolutionStudy(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(TurbulentMassFluxResolutionStudy, self).__init__(ig)

        # load data to list of structured arrays
        eht = []
        for file in filename:
            eht.append(np.load(file))

        # declare data lists		
        xzn0, nx, ny, nz = [], [], [], []

        ux, ddux, dd, fddx = [], [], [], []

        for i in range(len(filename)):
            # load grid
            xzn0.append(np.asarray(eht[i].item().get('xzn0')))

            nx.append(np.asarray(eht[i].item().get('nx')))
            ny.append(np.asarray(eht[i].item().get('ny')))
            nz.append(np.asarray(eht[i].item().get('nz')))

            # pick specific Reynolds-averaged mean fields according to:
            # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf 		

            ux.append(np.asarray(eht[i].item().get('ux')[intc]))
            dd.append(np.asarray(eht[i].item().get('dd')[intc]))
            ddux.append(np.asarray(eht[i].item().get('ddux')[intc]))
            fddx.append(ddux[i] - dd[i] * ux[i])

        # share data globally
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.fddx = fddx

    def plot_fddx(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot TurbulentMass flux in the model"""

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        fddx = self.fddx
        nx = self.nx
        ny = self.ny
        nz = self.nz

        # find maximum resolution data		
        grd_maxres = self.maxresdata(grd)
        nsq_maxres = self.maxresdata(fddx)

        plt_interp = []
        for i in range(len(grd)):
            plt_interp.append(np.interp(grd_maxres, grd[i], fddx[i]))

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('TurbulentMass Flux')

        for i in range(len(grd)):
            plt.plot(grd[i], fddx[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$f_\rho$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_TurbulentMass_flux.png')

    # find data with maximum resolution	
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
