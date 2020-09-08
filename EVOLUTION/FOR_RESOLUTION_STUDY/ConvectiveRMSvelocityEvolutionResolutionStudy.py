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

class ConvectiveRMSvelocityEvolutionResolutionStudy(uCalc.Calculus, uEal.ALIMITevol, uT.Tools, object):

    def __init__(self, filename, ig, data_prefix):
        super(ConvectiveRMSvelocityEvolutionResolutionStudy, self).__init__(ig)

        # load data to a list of structured arrays
        eht = []
        for ffile in filename:
            eht.append(np.load(ffile))

        # declare data lists
        t_timec, t_urms = [], []
        nx, ny, nz = [], [], []
        tavg, t_tc = [], []

        for i in range(len(filename)):
            # load temporal evolution
            t_timec.append(self.getRAdata(eht[i], 't_timec'))
            t_urms.append(self.getRAdata(eht[i], 't_urms'))

            nx.append(self.getRAdata(eht[i], 'nx'))
            ny.append(self.getRAdata(eht[i], 'ny'))
            nz.append(self.getRAdata(eht[i], 'nz'))

            tavg.append(self.getRAdata(eht[i], 'tavg'))
            t_tc.append(self.getRAdata(eht[i], 't_tc'))

        # share data across the whole class
        self.t_timec = t_timec
        self.t_urms = t_urms
        self.data_prefix = data_prefix

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.tavg = tavg
        self.t_tc = t_tc

    def plot_turms_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):

        grd = self.t_timec
        plt1 = self.t_urms
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
            print("ERROR(ConvectiveRMSvelocityEvolutionResolutionStudy.py): Only LAXIS=2 is supported.")
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

        # calculate indices for calculating mean for the plot label
        lmeanbndry = 300.
        umeanbndry = 500.

        il, ib = [],[]
        for i in range(len(self.t_timec)):
            tll = np.abs(np.asarray(self.t_timec[i]) - np.float(lmeanbndry))
            il.append(int(np.where(tll == tll.min())[0][0]))

            tlb = np.abs(np.asarray(self.t_timec[i]) - np.float(umeanbndry))
            ib.append(int(np.where(tlb == tlb.min())[0][0]))

        # plot DATA
        plt.title('convective tke velocity evolution')

        for i in range(len(grd)):
            plotdata = plt1[i]
            plt.plot(grd[i], plotdata, label=str(nx[i]) + ' x ' + str(ny[i]) + ' x ' + str(nz[i]) + ' '
                                            + '(tavg = ' + str(np.round(tavg[i], 1)) + ' s = '
                                            + str(np.round(tavg[i] / np.mean(t_tc[i]), 1)) + ' TOs, $\overline{u}$ = '
                                            + str(np.format_float_scientific(np.mean(plotdata[il[i]:ib[i]]), unique=False, precision=1)))
            # markers for time window for averages in label
            plt.axvline(lmeanbndry, linestyle='--', linewidth=0.7, color='k')
            plt.axvline(umeanbndry, linestyle='--', linewidth=0.7, color='k')

        print('WARNING(ConvectiveRMSvelocityResolutionStudy.py): mean value in the plot label calculated from-to ' + str(lmeanbndry)+'-'+str(umeanbndry) + ' s')



        # plt.plot(grd1,plt2,color='g',label = r'$epsD$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"u (cms/s)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'urms_evol.png')

    def plot_u_vs_L(self):

        Lsun = 3.839e33 # in ergs/s
        L = [4.5e43/Lsun,4.5e44/Lsun,4.5e45/Lsun]
        u = [7.2e6,9.3e6,1.7e7]

        c1 = 1.e11
        c2 = 1.e7
        exx1 = 1./3
        LscalingLaw1 = [c2*(L[0]/c1)**exx1,c2*(L[1]/c1)**exx1,c2*(L[2]/c1)**exx1]

        exx2 = 1./5.
        LscalingLaw2 = [c2*(L[0]/c1)**exx2,c2*(L[1]/c1)**exx2,c2*(L[2]/c1)**exx2]

        print(LscalingLaw1)

        # create FIGURE
        plt.figure(figsize=(7, 6))
        plt.title('tke velocity vs luminosity')

        plt.axis([1.e43/Lsun,1.e46/Lsun,3.e6,3.e7])
        plt.loglog(L,u,label=r'prompi ccp two-layers (128x128x128)',marker='o',color='b')
        plt.loglog(L,LscalingLaw1,label=r"$10^7(L/10^{11})^{1/3}$",color='r',linestyle='--')
        plt.loglog(L,LscalingLaw2,label=r"$10^7(L/10^{11})^{1/5}$",color='g',linestyle='--')

        # define and show x/y LABELS
        setxlabel = r"L/Lsun"
        setylabel = r"u (cm/s)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=0, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'L_vs_urms_evol.png')


    # find data with maximum resolution
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
