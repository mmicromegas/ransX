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

class XResolutionStudy(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, inuc, element, intc, data_prefix):
        super(XResolutionStudy, self).__init__(ig)

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

        # get mass coordinate
        self.mm = np.asarray(eht[0].item().get('mm')[intc])

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

    def plot_X(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot mass fraction in the model"""

        if (LAXIS != 2):
            print("ERROR(XdensityResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        # load x GRID
        grd = self.xzn0

        # load DATA to plot		
        plt1 = self.fht_xi
        nx = self.nx
        ny = self.ny
        nz = self.nz

        # find maximum resolution data
        grd_maxres = self.maxresdata(grd)
        plt1_maxres = self.maxresdata(plt1)

        plt_interp = []
        for i in range(len(grd)):
            plt_interp.append(np.interp(grd_maxres, grd[i], plt1[i]))

        fig, ax1 = plt.subplots(figsize=(7, 6))

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
        plt.title('X for ' + self.element)


        #ax1.semilogy(grd[0], plt1[0], linestyle='--',label='initial')

        for i in range(len(grd)):
            #plt.semilogy(grd[i], plt1[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]) +
            #                                ' t: ' + str(round(self.timec[i])) + ' s')
            #ax1.semilogy(grd[i], plt1[i], label= str(np.around(self.timec[i]).astype(int)) + ' s')
            plt.plot(grd[i], plt1[i], label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))


        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\widetilde{X}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$X_i$ (" + self.element + " )"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 14})

        #ax2 = ax1.twiny()
        #ax2.set_xlim(ax1.get_xlim())

        #newMMlabel_xpos = [3.8e8, 4.7e8, 5.5e8, 6.4e8, 7.3e8, 8.5e8, 9.5e8]
        #newMMlabel = self.mlabels(newMMlabel_xpos)
        #ax2.set_xticks(newMMlabel_xpos)

        #ax2.set_xticklabels(newMMlabel)
        #ax2.set_xlabel('enclosed mass (msol)')


        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_' + self.element + '.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_X_' + self.element + '.eps')

    # find data with maximum resolution	
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres

    def mlabels(self, grid):
        # calculate MM labels
        xzn0 = np.asarray(self.xzn0[0])
        msun = 1.989e33  # in grams
        M_label = []
        for grd in grid:
            xlm = np.abs(xzn0 - grd)
            idx = int(np.where(xlm == xlm.min())[0][0])
            M_label.append(str(np.round(self.mm[idx]/msun,1)))

        return M_label

    def setNucNoUp(self, inpt):
        elmnt = ""
        if inpt == "neut":
            elmnt = r"neut"
        if inpt == "prot":
            elmnt = r"prot"
        if inpt == "he4":
            elmnt = r"He$^{4}$"
        if inpt == "c12":
            elmnt = r"C$^{12}$"
        if inpt == "o16":
            elmnt = r"O$^{16}$"
        if inpt == "ne20":
            elmnt = r"Ne$^{20}$"
        if inpt == "na23":
            elmnt = r"Na$^{23}$"
        if inpt == "mg24":
            elmnt = r"Mg$^{24}$"
        if inpt == "si28":
            elmnt = r"Si$^{28}$"
        if inpt == "p31":
            elmnt = r"P$^{31}$"
        if inpt == "s32":
            elmnt = r"S$^{32}$"
        if inpt == "s34":
            elmnt = r"S$^{34}$"
        if inpt == "cl35":
            elmnt = r"Cl$^{35}$"
        if inpt == "ar36":
            elmnt = r"Ar$^{36}$"
        if inpt == "ar38":
            elmnt = r"Ar$^{38}$"
        if inpt == "k39":
            elmnt = r"K$^{39}$"
        if inpt == "ca40":
            elmnt = r"Ca$^{40}$"
        if inpt == "ca42":
            elmnt = r"Ca$^{42}$"
        if inpt == "ti44":
            elmnt = r"Ti$^{44}$"
        if inpt == "ti46":
            elmnt = r"Ti$^{46}$"
        if inpt == "cr48":
            elmnt = r"Cr$^{48}$"
        if inpt == "cr50":
            elmnt = r"Cr$^{50}$"
        if inpt == "fe52":
            elmnt = r"Fe$^{52}$"
        if inpt == "fe54":
            elmnt = r"Fe$^{54}$"
        if inpt == "ni56":
            elmnt = r"Ni$^{56}$"

        return elmnt