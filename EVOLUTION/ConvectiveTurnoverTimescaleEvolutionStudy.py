import numpy as np
import sys
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.EVOL.ALIMITevol import ALIMITevol
from UTILS.Tools import Tools

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ConvectiveTurnoverTimescaleEvolutionResolutionStudy(Calculus, ALIMITevol, Tools, object):

    def __init__(self, filename, ig, data_prefix):
        super(ConvectiveTurnoverTimescaleEvolutionResolutionStudy, self).__init__(ig)

        # load data to a list of structured arrays
        eht = []
        for ffile in filename:
            eht.append(self.customLoad(ffile))

        # declare data lists
        t_timec, t_tc = [], []
        nx, ny, nz = [], [], []
        tavg = []

        for i in range(len(filename)):
            # load temporal evolution
            t_timec.append(self.getRAdata(eht[i],'t_timec'))
            t_tc.append(self.getRAdata(eht[i],'t_tc'))

            nx.append(self.getRAdata(eht[i],'nx'))
            ny.append(self.getRAdata(eht[i],'ny'))
            nz.append(self.getRAdata(eht[i],'nz'))

            tavg.append(self.getRAdata(eht[i], 'tavg'))

        # share data across the whole class
        self.t_timec = t_timec
        self.t_tc = t_tc
        self.data_prefix = data_prefix

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.tavg = tavg

    def plot_tconvturn_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):

        grd = self.t_timec
        plt1 = self.t_tc
        # plt2 = self.t_epsD

        # load resolution
        nx = self.nx
        ny = self.ny
        nz = self.nz

        tavg = self.tavg
        t_tc = self.t_tc

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
            print("ERROR(ConvectiveTurnoverTimescaleEvolutionStudy.py): Only LAXIS=2 is supported.")
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
        plt.title('convective turnover timescale evolution')

        for i in range(len(grd)):
            plt.plot(grd[i], plt1[i], label=str(nx[i]) + ' x ' + str(ny[i]) + ' x ' + str(nz[i]) + ' '
                                            + '(tavg = ' + str(np.round(tavg[i],1)) + ' s = '
                                            + str(np.round(tavg[i]/np.mean(t_tc[i]),1)) + ' TOs)')

        # plt.plot(grd1,plt2,color='g',label = r'$epsD$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"timescale (s)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tconvturnover_evol_res.png')

    # find data with maximum resolution
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
