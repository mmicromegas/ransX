import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.EVOL.ALIMITevol as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class MachNumberMeanEvolutionResolutionStudy(calc.CALCULUS, al.ALIMITevol, object):

    def __init__(self, filename, ig, data_prefix):
        super(MachNumberMeanEvolutionResolutionStudy, self).__init__(ig)

        # load data to a list of structured arrays
        eht = []
        for file in filename:
            eht.append(np.load(file))

        # declare data lists
        t_timec, t_machme = [], []
        nx, ny, nz = [], [], []

        for i in range(len(filename)):
            # load temporal evolution
            t_timec.append(np.asarray(eht[i].item().get('t_timec')))
            t_machme.append(np.asarray(eht[i].item().get('t_machMean')))

            nx.append(np.asarray(eht[i].item().get('nx')))
            ny.append(np.asarray(eht[i].item().get('ny')))
            nz.append(np.asarray(eht[i].item().get('nz')))

        # share data across the whole class
        self.t_timec = t_timec
        self.t_machme = t_machme

        self.data_prefix = data_prefix

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def plot_machmean_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):

        grd = self.t_timec
        plt1 = self.t_machme

        # load resolution
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

        if (LAXIS != 2):
            print("ERROR(MachNumberMeanEvolutionResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

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
        plt.title('mean Mach number evolution')

        for i in range(len(grd)):
            plt.plot(grd[i], plt1[i], label=str(nx[i]) + ' x ' + str(ny[i]) + ' x ' + str(nz[i]))

        # plt.plot(grd1,plt2,color='g',label = r'$epsD$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"Mach"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'machmean_evol.png')

    # find data with maximum resolution
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres