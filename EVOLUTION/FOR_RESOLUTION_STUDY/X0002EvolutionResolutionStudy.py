import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.EVOL.ALIMITevol as uEal
import UTILS.Tools as uT


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class X0002EvolutionResolutionStudy(uCalc.Calculus, uEal.ALIMITevol, uT.Tools, object):

    def __init__(self, filename, ig, data_prefix):
        super(X0002EvolutionResolutionStudy, self).__init__(ig)

        # load data to a list of structured arrays
        eht = []
        for file in filename:
            eht.append(np.load(file))

        # declare data lists
        t_timec, t_x0002mean_cnvz = [], []
        nx, ny, nz = [], [], []
        tavg, t_tc = [], []

        for i in range(len(filename)):
            # load temporal evolution
            t_timec.append(self.getRAdata(eht[i], 't_timec'))
            t_x0002mean_cnvz.append(self.getRAdata(eht[i], 't_x0002mean_cnvz'))

            nx.append(self.getRAdata(eht[i], 'nx'))
            ny.append(self.getRAdata(eht[i], 'ny'))
            nz.append(self.getRAdata(eht[i], 'nz'))

            tavg.append(self.getRAdata(eht[i], 'tavg'))
            t_tc.append(self.getRAdata(eht[i], 't_tc'))
            print(self.getRAdata(eht[i], 't_x0002mean_cnvz'))

        # share data across the whole class
        self.t_timec = t_timec
        self.t_x0002mean_cnvz = t_x0002mean_cnvz

        self.data_prefix = data_prefix

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.tavg = tavg
        self.t_tc = t_tc

    def plot_x0002_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):

        grd = self.t_timec
        plt1 = self.t_x0002mean_cnvz

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
            print("ERROR(X0002EvolutionResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        plt10_tmp = plt1[0]
        plt11_tmp = plt1[0]

        plt1_foraxislimit = []
        plt1max = np.max(plt1[0])
        for plt1i in plt1:
            if (np.max(plt1i) > plt1max):
                plt1_foraxislimit = plt1i

        # calculate indices for calculating mean for the plot label
        lmeanbndry = 300.
        umeanbndry = 500.

        il, ib = [],[]
        for i in range(len(self.t_timec)):
            tll = np.abs(np.asarray(self.t_timec[i]) - np.float(lmeanbndry))
            il.append(int(np.where(tll == tll.min())[0][0]))

            tlb = np.abs(np.asarray(self.t_timec[i]) - np.float(umeanbndry))
            ib.append(int(np.where(tlb == tlb.min())[0][0]))

        # set plot boundaries
        to_plot = [plt1_foraxislimit]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('bottom 2/3 of cnvz')

        for i in range(len(grd)):
            plotdata = plt1[i]
            grid = grd[i]
            xrate = (plotdata[ib[i]]-plotdata[il[i]])/(grid[ib[i]]-grid[il[i]])
            plt.plot(grid, plotdata, label=str(nx[i]) + ' x ' + str(ny[i]) + ' x ' + str(nz[i]) + ' '
                                            + '(tavg = ' + str(np.round(tavg[i],1)) + ' s = '
                                            + str(np.round(tavg[i]/np.mean(t_tc[i]),1)) + ' TOs, $\overline{X}$ = '
                                            + str(np.format_float_scientific(np.mean(plotdata[il[i]:ib[i]]), unique=False, precision=2))
                                            + ' $\dot{X}$ = ' + str(np.format_float_scientific(xrate, unique=False, precision=2)))
            # markers for time window for averages in label
            plt.axvline(lmeanbndry, linestyle='--', linewidth=0.7, color='k')
            plt.axvline(umeanbndry, linestyle='--', linewidth=0.7, color='k')

            # print(il[i],ib[i],np.mean(plt1[il[i]:ib[i]]))

        print('WARNING(X0002EvolutionResolutionStudy.py): mean value in the plot label calculated from-to ' + str(lmeanbndry)+'-'+str(umeanbndry) + ' s')

        #for i in range(len(grd)):
        #    plt.plot(grd[i], plt1[i], label=str(nx[i]) + ' x ' + str(ny[i]) + ' x ' + str(nz[i]))


        # plt.plot(grd1,plt2,color='g',label = r'$epsD$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"X2"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 10})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'x0002_evol_res.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'x0002_evol_res.eps')

    # find data with maximum resolution
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
