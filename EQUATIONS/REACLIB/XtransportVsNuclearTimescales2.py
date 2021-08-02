import numpy as np
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

class XtransportVsNuclearTimescales(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, filename_reaclib, ig, inuc, element, bconv, tconv, tc, intc, data_prefix,
                 fext, tnuc, network):
        super(XtransportVsNuclearTimescales, self).__init__(ig)

        # load RANS data to structured array
        eht = self.customLoad(filename)

        # load REACLIB data	

        rcoeff_tmp = []
        rlabel = []

        # read line-by-line		
        with open(filename_reaclib) as handle:
            for lineno, line in enumerate(handle):
                if (lineno != 0) and (lineno % 3 != 0):
                    rcoeff_tmp.append(line.rstrip())
                if lineno % 3 == 0:
                    rlabel.append(line[0:52].replace(" ", ""))

                    # restructure and join every second and third coefficient line
        rcoeff = [''.join(x) for x in zip(rcoeff_tmp[0::2], rcoeff_tmp[1::2])]

        # split reaction coefficient string to individual coefficients
        n = 13
        rcoeffdict = {}
        for i in range(len(rcoeff)):
            rcoeffone = rcoeff[i]
            # parse out individual reaction rate coefficients 			
            out = [(rcoeffone[j:j + n]) for j in range(0, len(rcoeffone), n)]
            # convert to float 			 
            outfloat = []
            for k in out:
                outfloat.append(float(k))
            # store in dictionary				
            rc = {rlabel[i]: outfloat}
            rcoeffdict.update(rc)

        # print(rcoeffdict["he4ar36pk39rathrv"])
        # print(rlabel)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		

        dd = self.getRAdata(eht, 'dd')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        fxi = ddxiux - ddxi * ddux / dd

        #######################
        # Xi TRANSPORT EQUATION 
        #######################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_fht_xi = t_ddxi / t_dd

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        fxi = ddxiux - ddxi * ddux / dd

        # LHS -dq/dt 		
        self.minus_dt_dd_fht_xi = -self.dt(t_dd * t_fht_xi, xzn0, t_timec, intc)
        self.minus_dt_fht_xi = -self.dt(t_fht_xi, xzn0, t_timec, intc)

        # LHS -div(ddXiux)
        self.minus_div_eht_dd_fht_ux_fht_xi = -self.Div(dd * fht_ux * fht_xi, xzn0)

        # RHS -div fxi 
        self.minus_div_fxi = -self.Div(fxi, xzn0)

        # RHS +ddXidot 
        self.plus_ddxidot = +ddxidot

        # -res
        self.minus_resXiTransport = -(self.minus_dt_dd_fht_xi + self.minus_div_eht_dd_fht_ux_fht_xi + self.minus_div_fxi + self.plus_ddxidot)

        ###########################
        # END Xi TRANSPORT EQUATION
        ###########################

        # tau_trans = np.abs(fht_xi/self.Div(fxi/dd,xzn0))
        # tau_nuc   = np.abs(fht_xi/(ddxidot/dd))

        tau_trans = np.abs(dd*fht_xi/self.Div(fxi,xzn0))
        tau_nuc   = np.abs(dd*fht_xi/(ddxidot))
        tau_ddxi =  np.abs(dd*fht_xi/self.minus_dt_dd_fht_xi)
        tau_xi =  np.abs(fht_xi/self.minus_dt_fht_xi)

        #tau_trans = (dd*fht_xi/self.Div(fxi,xzn0))
        #tau_nuc   = (dd*fht_xi/(ddxidot))
        #tau_ddxi =  (dd*fht_xi/self.minus_dt_dd_fht_xi)
        #tau_xi =  (fht_xi/self.minus_dt_fht_xi)


        #tau_trans = np.abs(fht_xi/self.Div(fxi/dd,xzn0))
        #tau_nuc   = np.abs(fht_xi/(ddxidot/dd))
        #tau_ddxi =  np.abs(dd*fht_xi/self.minus_dt_dd_fht_xi)
        #tau_xi =  np.abs(fht_xi/self.minus_dt_fht_xi)

        # tau_trans = (fht_xi / self.Div(fxi / dd, xzn0))
        # tau_nuc = (fht_xi / (ddxidot / dd))

        # tau_trans = ddxi/self.Div(fxi,xzn0)
        # tau_nuc   = ddxi/(ddxidot)

        # Damkohler number		
        self.xda = tau_trans / tau_nuc

        # assign global data to be shared across whole class	
        self.data_prefix = data_prefix
        self.xzn0 = self.getRAdata(eht, 'xzn0')
        self.element = element
        self.inuc = inuc
        self.bconv = bconv
        self.tconv = tconv
        self.tc = tc
        self.tt = tt
        self.dd = dd
        self.rlabel = rlabel
        self.rcoeffdict = rcoeffdict

        # self.tau_trans = np.abs(tau_trans)
        # self.tau_nuc = np.abs(tau_nuc)

        self.tau_trans = tau_trans
        self.tau_nuc = tau_nuc
        self.tau_ddxi = tau_ddxi
        self.tau_xi = tau_xi

        self.fht_xi = fht_xi
        self.network = network
        self.eht = eht
        self.intc = intc
        self.fext = fext
        self.tnuc = tnuc
        self.nx = nx

    def plot_Xtransport_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xrho transport equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportVsNuclearTimescalesEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_fht_xi
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_xi

        rhs0 = self.minus_div_fxi
        rhs1 = self.plus_ddxidot

        res = self.minus_resXiTransport

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('rhoX transport for ' + element)
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='r', label=r'$-\partial_t (\overline{\rho} \widetilde{X})$')
            plt.plot(grd1, lhs1, color='cyan', label=r'$-\nabla_x (\overline{\rho} \widetilde{X} \widetilde{u}_x)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_x f$')
            plt.plot(grd1, rhs1, color='g', label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='r', label=r'$-\partial_t (\overline{\rho} \widetilde{X})$')
            plt.plot(grd1, lhs1, color='cyan', label=r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$')
            plt.plot(grd1, rhs0, color='b', label=r'$-\nabla_r f$')
            plt.plot(grd1, rhs1, color='g', label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r'x (cm)'
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r'r (cm)'
            setylabel = r"g cm$^{-3}$ s$^{-1}$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.eps')

    def plot_Xtimescales(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # Damkohler number

        if self.ig != 1 and self.ig != 2:
            print("ERROR(Xtimescales.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # convert nuc ID to string
        xnucid = str(self.inuc)

        element = self.element
        rlabel = self.rlabel
        rcoeffdict = self.rcoeffdict
        eht = self.eht
        intc = self.intc
        dd = self.dd
        network = self.network

        # yi = self.fht_xi
        # yj =
        # yk =

        # load x GRID
        grd1 = self.xzn0

        # get data
        plt0 = self.tau_trans
        plt1 = self.tau_nuc
        plt2 = self.tau_ddxi
        plt3 = self.tau_xi

        #print(plt1)


        onebody_interaction = []
        twobody_interaction = []

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries
        to_plot = []
        to_plot.append(plt0)
        to_plot.append(plt1)

        for value in onebody_interaction:
            to_plot.append(value)

        for value in twobody_interaction:
            to_plot.append(value)

        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        plt.yscale('symlog')

        # plot DATA 		
        plt.title(r"$timescales \ for \ $" + self.element)
        # plt.plot(grd1,plt0,label=r"$-\tau_{trans}^i$",color='r')
        # plt.plot(grd1,plt1,label=r"$-\tau_{nuc}^i$",color='b')

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))

        plt.plot(grd1[xlimitrange], plt0[xlimitrange], label=r"$|\tau_{trans}^i|$", color='r')
        plt.plot(grd1[xlimitrange], plt1[xlimitrange], label=r"$|\tau_{nuc}^i|$", color='b')
        plt.plot(grd1[xlimitrange], plt2[xlimitrange], label=r"$|\tau_{\rho X}^i|$", color='m')
        #plt.plot(grd1[xlimitrange], plt3[xlimitrange], label=r"$|\tau_{X}^i|$", color='brown')

        xlimitbottom = np.where(grd1 < self.bconv)
        plt.plot(grd1[xlimitbottom], plt0[xlimitbottom], '.', color='r', markersize=0.5)
        plt.plot(grd1[xlimitbottom], plt1[xlimitbottom], '.', color='b', markersize=0.5)

        xlimittop = np.where(grd1 > self.tconv)
        plt.plot(grd1[xlimittop], plt0[xlimittop], '.', color='r', markersize=0.5)
        plt.plot(grd1[xlimittop], plt1[xlimittop], '.', color='b', markersize=0.5)

        # oplot convective turnover timescale
        plt.axhline(y=self.tc, color='k', linestyle='--', linewidth=0.5)
        plt.text(self.tconv, self.tc, r"$\tau_{conv}$")

        # oplot age of the universe
        aoftheuniverse = 4.3e17  # in seconds
        plt.axhline(y=aoftheuniverse, color='k', linestyle='--', linewidth=0.5)
        plt.text(self.tconv, aoftheuniverse, r"ageOftheUniverse")

        # convective boundary markers
        # plt.axvline(self.bconv, linestyle='-', linewidth=0.7, color='k')
        # plt.axvline(self.tconv, linestyle='-', linewidth=0.7, color='k')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'xTimescales_' + element + '.png')
        #if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'xTimescales_' + element + '.eps')

        #xlim_l = 4.e8
        #xlim_r = 7.e8
        # idxl, idxr = self.idx_bndry(xlim_l, xlim_r)

        #print(idxr,self.nx)

        #self.tau_trans[idxr:self.nx] = 1.e10
        #ind_inst = np.where(self.tau_nuc < self.tc)[0]  #

        #for rr in self.xzn0:
        #    if rr < self.tconv:
        #        ind_inst = np.where((grd1 < self.tconv) & (grd1 > rr))
                #ind_inst = np.where((grd1 < 8.3e8) & (grd1 > grd1[indL]))
        #        ind_inst = ind_inst[0]

                #if len(ind_inst) != 0:
                #    print(ind_inst)

        #        indBurn = np.where(grd1 < rr)
        #        indBurn = indBurn[0]

                #print(indBurn)

        #sys.exit()

        miny = 1.e-6
        maxy = 1.e0
        plt.figure(figsize=(7, 6))
        plt.title('ne20')
        plt.axis([xbl, xbr,miny,maxy])

        #plt.figure(figsize=(7, 6))
        #lb = 1.e-5
        #ub = 1.e18
        # plt.yscale('symlog')
        #plt.axis([xbl, xbr, lb, ub])


        ne20totlum = []
        rtotlum = []
        ii = 1
        for rr in self.xzn0:
            if self.tconv > rr > self.bconv:
                ii = ii + 1
                ind_inst = np.where((grd1 < self.tconv) & (grd1 > rr))
                #ind_inst = np.where((grd1 < 8.3e8) & (grd1 > grd1[indL]))
                if len(ind_inst[0]) != 0:
                    ind_inst = ind_inst[0]

                    #print(ind_inst)

                    indBurn = np.where(grd1 < rr)
                    indBurn = indBurn[0]

                    #####################################

                    rc = self.getRAdata(eht, 'xzn0')
                    xznl = self.getRAdata(eht, 'xznl')
                    xznr = self.getRAdata(eht, 'xznr')

                    tt = self.getRAdata(eht, 'tt')[intc]
                    dd = self.getRAdata(eht, 'dd')[intc]

                    # for 25 element network
                    #xhe4 = self.data['x0003']
                    #xc12 = self.data['x0004']
                    #xo16 = self.data['x0005']
                    #xne20 = self.data['x0006']
                    #xsi28 = self.data['x0009']

                    # for 14 elements network
                    xhe4 = self.getRAdata(eht, 'x0003')[intc]
                    xc12 = self.getRAdata(eht, 'x0004')[intc]
                    xo16 = self.getRAdata(eht, 'x0005')[intc]
                    xne20 = self.getRAdata(eht, 'x0006')[intc]
                    xsi28 = self.getRAdata(eht, 'x0007')[intc]

                    #bconv = 4.2e8
                    #tconv = 9.5e8

                    Vol = 4. / 3. * np.pi * (xznr ** 3 - xznl ** 3)
                    ind = ind_inst
                    #ind = np.where((rc > bconv) & (rc < tconv))[0]
                    M = (dd * Vol)[ind].sum()

                    print(ind)

                    if 1==1:
                        Mhe4 = (dd * xhe4 * Vol)[ind].sum()
                        Mc12 = (dd * xc12 * Vol)[ind].sum()
                        Mo16 = (dd * xo16 * Vol)[ind].sum()
                        Mne20 = (dd * xne20 * Vol)[ind].sum()
                        Msi28 = (dd * xsi28 * Vol)[ind].sum()

                        xhe4inst = Mhe4/M
                        xc12inst = Mc12/M
                        xo16inst = Mo16/M
                        xne20inst = Mne20/M
                        xsi28inst = Msi28/M

                        xhe4mean = xhe4[ind].mean()
                        xc12mean = xc12[ind].mean()
                        xo16mean = xo16[ind].mean()
                        xne20mean = xne20[ind].mean()
                        xsi28mean = xsi28[ind].mean()

                        xc12rd = (xc12mean-xc12inst)/xc12inst
                        xo16rd = (xo16mean-xo16inst)/xo16inst
                        xne20rd = (xne20mean-xne20inst)/xne20inst
                        xsi28rd = (xsi28mean-xsi28inst)/xsi28inst

                        #print('Xc12 mean:' + str(xc12mean) + '  X inst. mass conserved: ' + str(xc12inst) + ' rel.diff. ' + str(np.round(xc12rd,6)))
                        #print('Xo16 mean:' + str(xo16mean) + '  X inst. mass conserved: ' + str(xo16inst) + ' rel.diff. ' + str(np.round(xo16rd,6)))
                        #print('Xne20 mean:' + str(xne20mean) + '  X inst. mass conserved: ' + str(xne20inst) + ' rel.diff. ' + str(np.round(xne20rd,6)))
                        #print('Xsi28 mean:' + str(xsi28mean) + '  X inst. mass conserved: ' + str(xsi28inst) + ' rel.diff. ' + str(np.round(xsi28rd,6)))

                        #xhe4 = np.zeros(nx)
                        #xc12 = np.zeros(nx)
                        #xo16 = np.zeros(nx)
                        #xne20 = np.zeros(nx)
                        #xsi28 = np.zeros(nx)

                        #xbl = rc[0]
                        #xbr = rc[-1]

                        #plt.figure(figsize=(7, 6))
                        #miny = 1.e-6
                        #maxy = 1.e-4
                        #plt.axis([xbl, xbr, miny, maxy])
                        #plt.semilogy(rc,xc12,color='r',label='3D non-instantaneous')
                        xc12[ind] = xc12inst
                        #plt.title(r'X(C12)')
                        #plt.semilogy(rc,xc12,color='b',label='instantaneous')
                        #plt.legend(loc=2, prop={'size': 18}, ncol=1)
                        #plt.ylabel(r"X")
                        #plt.xlabel('r (cm)')
                        #plt.show(block=False)
                        #plt.savefig('RESULTS/xc12.png')

                        #plt.figure(figsize=(7, 6))
                        #miny = 4.e-1
                        #maxy = 5.e-1
                        #plt.axis([xbl, xbr, miny, maxy])
                        #plt.semilogy(rc,xo16,color='r',label='3D non-instantaneous')
                        xo16[ind] = xo16inst
                        #plt.title(r'X(O16)')
                        #plt.semilogy(rc,xo16,color='b',label='instantaneous')
                        #plt.legend(loc=4, prop={'size': 18}, ncol=1)
                        #plt.ylabel(r"X")
                        #plt.xlabel('r (cm)')
                        #plt.show(block=False)
                        #plt.savefig('RESULTS/xo16.png')

                        #plt.figure(figsize=(7, 6))
                        #miny = 1.e-6
                        #maxy = 1.e-1
                        #plt.axis([xbl, xbr, miny, maxy])

                        #######################################
                        plt.semilogy(rc, xne20, color='r')
                        xne20[ind] = xne20inst
                        xne20[indBurn] = 0.
                        if (ii % 8) == 0:
                        #    xne20[ind] = xne20inst
                        #    xne20[indBurn] = 0.
                            plt.semilogy(rc,xne20)
                        #######################################

                        #plt.show(block=False)
                        #plt.savefig('RESULTS/xne20.png')
                        #plt.savefig('RESULTS/xne20.eps')

                        #plt.figure(figsize=(7, 6))
                        #miny = 1.e-1
                        #maxy = 8.e-1
                        #plt.axis([xbl, xbr, miny, maxy])
                        #plt.semilogy(rc,xsi28,color='r',label='3D non-instantaneous')
                        xsi28[ind] = xsi28inst
                        #plt.title(r'X(Si28)')
                        #plt.semilogy(rc,xsi28,color='b',label='instantaneous')
                        #plt.legend(loc=3, prop={'size': 18}, ncol=1)
                        #plt.ylabel(r"X")
                        #plt.xlabel('r (cm)')
                        #plt.show(block=False)
                        #plt.savefig('RESULTS/xsi28.png')


                        #print(xne20)
                        #sys.exit()

                        #  enuc = self.data['enuc1']+self.data['enuc2']


                        #xo16 = self.data['x0003']
                        #xne20 = self.data['x0004']
                        #xc12 = np.zeros(xne20.shape[0])
                        #xsi28 = np.zeros(xne20.shape[0])

                    enuc1 = self.getRAdata(eht,'enuc1')[intc]
                    enuc2 = self.getRAdata(eht,'enuc2')[intc]

                    #       ne20 > he4 + o16 (photo-d: resonance)
                    t9 = tt / 1.e9
                    # + 4.e-2*self.eht_tt[:,tt]/1.e9

                    # rate coefficients from netsu (source cf88)

                    cl = self.GETRATEcoeff(reaction='ne20_to_he4_o16_rv')
                    rate_ne20_alpha_gamma = np.exp(
                        cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                            5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

                    #       he4 + ne20 > mg24
                    cl = self.GETRATEcoeff(reaction='he4_plus_ne20_to_mg24_r')
                    rate_ne20_alpha_gamma_code = np.exp(
                        cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                            5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

                    #       o16 + o16 > p + p31 (resonance)
                    #        xo16 = self.fht_xo16[:,tt]
                    cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_p_p31_r')
                    rate_o16_o16_pchannel_r = np.exp(
                        cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                            5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

                    #       o16 + o16 > he4 + si28 (resonance)
                    #        xo16 = self.fht_xo16[:,tt]
                    cl = self.GETRATEcoeff(reaction='o16_plus_o16_to_he4_si28_r')
                    rate_o16_o16_achannel_r = np.exp(
                        cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                            5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

                    #       c12 + c12 > p + na23 (resonance)
                    cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_p_na23_r')
                    rate_c12_c12_pchannel_r = np.exp(
                        cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                            5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

                    #       c12 + c12 > he4 + ne20 (resonance)
                    cl = self.GETRATEcoeff(reaction='c12_plus_c12_to_he4_ne20_r')
                    rate_c12_c12_achannel_r = np.exp(
                        cl[0] + cl[1] * (t9 ** (-1.)) + cl[2] * (t9 ** (-1. / 3.)) + cl[3] * (t9 ** (1. / 3.)) + cl[4] * t9 + cl[
                            5] * (t9 ** (5. / 3.)) + cl[6] * np.log(t9))

                    # ANALYTIC EXPRESSIONS Caughlan & Fowler 1988

                    t9a = t9 / (1. + 0.0396 * t9)
                    c_tmp1 = (4.27e26) * (t9a ** (5. / 6.))
                    c_tmp2 = t9 ** (3. / 2.)
                    c_e_tmp1 = -84.165 / (t9a ** (1. / 3.))
                    c_e_tmp2 = -(2.12e-3) * (t9 ** 3.)

                    rate_c12_c12 = c_tmp1 / c_tmp2 * (np.exp(c_e_tmp1 + c_e_tmp2))

                    o_tmp1 = 7.1e36 / (t9 ** (2. / 3.))
                    o_c_tmp1 = -135.93 / (t9 ** (1. / 3.))
                    o_c_tmp2 = -0.629 * (t9 ** (2. / 3.))
                    o_c_tmp3 = -0.445 * (t9 ** (4. / 3.))
                    o_c_tmp4 = +0.0103 * (t9 ** 2.)

                    rate_o16_o16 = o_tmp1 * np.exp(o_c_tmp1 + o_c_tmp2 + o_c_tmp3 + o_c_tmp4)

                    n_tmp1 = 4.11e11 / (t9 ** (2. / 3.))
                    n_e_tmp1 = -46.766 / (t9 ** (1. / 3.)) - (t9 / 2.219) ** 2.
                    n_tmp2 = 1. + 0.009 * (t9 ** (1. / 3.)) + 0.882 * (t9 ** (2. / 3.)) + 0.055 * t9 + 0.749 * (
                                t9 ** (4. / 3.)) + 0.119 * (t9 ** (5. / 3.))
                    n_tmp3 = 5.27e3 / (t9 ** (3. / 2.))
                    n_e_tmp3 = -15.869 / t9

                    n_tmp4 = 6.51e3 * (t9 ** (1. / 2.))
                    n_e_tmp4 = -16.223 / t9

                    rate_alpha_gamma_cf88 = n_tmp1 * np.exp(n_e_tmp1) * n_tmp2 + n_tmp3 * np.exp(n_e_tmp3) + n_tmp4 * np.exp(
                        n_e_tmp4)

                    c1_c12 = 4.8e18
                    c1_o16 = 8.e18
                    c1_ne20 = 2.5e29
                    c1_si28 = 1.8e28

                    yc12sq = (xc12 / 12.) ** 2.
                    yo16sq = (xo16 / 16.) ** 2.
                    yne20sq = (xne20 / 20.) ** 2.

                    yo16 = xo16 / 16.

                    lag = (3.e-3) * (t9 ** (10.5))
                    lox = (2.8e-12) * (t9 / 2.) ** 33.
                    lca = (4.e-11) * (t9 ** 29.)
                    lsi = 120. * (t9 / 3.5) ** 5.

                    en_c12 = c1_c12 * yc12sq * dd * (rate_c12_c12_achannel_r + rate_c12_c12_pchannel_r)
                    en_c12_acf88 = c1_c12 * yc12sq * dd * (rate_c12_c12)
                    en_o16 = c1_o16 * yo16sq * dd * (rate_o16_o16_achannel_r + rate_o16_o16_pchannel_r)
                    en_o16_acf88 = c1_o16 * yo16sq * dd * (rate_o16_o16)
                    en_ne20 = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * rate_ne20_alpha_gamma_code * np.exp(-54.89 / t9)
                    en_ne20_acf88 = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * rate_ne20_alpha_gamma * np.exp(-54.89 / t9)
                    en_ne20_hw = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * lag * np.exp(-54.89 / t9)
                    #        en_ne20_ini = c1_ne20*(t9**(3./2.))*(yne20sq_ini/yo16)*rate_ne20_alpha_gamma_code*np.exp(-54.89/t9)
                    en_ne20_lag = c1_ne20 * (t9 ** (3. / 2.)) * (yne20sq / yo16) * lag * np.exp(-54.89 / t9)
                    en_si28 = c1_si28 * (t9 ** 3. / 2.) * xsi28 * (np.exp(-142.07 / t9)) * rate_ne20_alpha_gamma_code
                    en_si28_acf88 = c1_si28 * (t9 ** 3. / 2.) * xsi28 * (np.exp(-142.07 / t9)) * rate_ne20_alpha_gamma

                    #plt.figure(figsize=(7, 6))

                    #lb = 1.e-5
                    #ub = 1.e18

                    #plt.yscale('symlog')

                    #plt.axis([xbl, xbr, lb, ub])

                    #plt.title(r'instantaneous')
                    #plt.title(r'3D')
                    #plt.semilogy(rc, en_c12, label=r"$\dot{\epsilon}_{\rm nuc}$ (C$^{12}$)")
                    #plt.semilogy(rc, en_o16, label=r"$\dot{\epsilon}_{\rm nuc}$ (O$^{16}$)")
                    #if (ii % 8) == 0:
                    #    plt.semilogy(rc, en_ne20)
                    #plt.semilogy(rc, en_si28, label=r"$\dot{\epsilon}_{\rm nuc}$ (Si$^{28}$)")
                    # plt.semilogy(rc, en_c12 + en_o16 + en_ne20 + en_si28,label='total', color='k',linestyle='--')
                    # plt.plot(rc,enuc1,color='m',linestyle='--',label='enuc1')
                    # plt.plot(rc,enuc2,color='r',linestyle='--',label='-neut code')
                    # plt.plot(rc,enuc1-enuc2,color='b',linestyle='--',label='enuc1-enuc2')

                    en_c12tot = (en_c12 * dd * Vol)[ind].sum()
                    en_o16tot = (en_o16 * dd * Vol)[ind].sum()
                    en_ne20tot = (en_ne20 * dd * Vol)[ind].sum()
                    en_si28tot = (en_si28 * dd * Vol)[ind].sum()

                    #print('Total Enuc c12 burn:' + str(en_c12tot))
                    #print('Total Enuc o16 burn:' + str(en_o16tot))
                    print('Total Enuc ne20 burn:' + str(en_ne20tot))
                    #print('Total Enuc si28 burn:' + str(en_si28tot))

                    rtotlum.append(rr)
                    ne20totlum.append(en_ne20tot)

        plt.semilogy(rc, xne20, color='b')
        plt.legend(loc=2, prop={'size': 18}, ncol=1)
        plt.ylabel(r"X")
        plt.xlabel('r (cm)')


        #plt.legend(loc=1, prop={'size': 14}, ncol=1)
        #plt.ylabel(r"$\dot{\epsilon}_{\rm nuc}$ (erg g$^{-1}$ s$^{-1}$)")
        #plt.xlabel('r ($10^8$ cm)')
        #plt.xlabel('r (cm)')

        #axvline(x=5.65, color='k', linewidth=1)
        plt.show(block=False)
        #        text(9.,1.e6,r"ob",fontsize=42,color='k')

        plt.savefig('RESULTS/oburn14_X_inst.png')


        # create FIGURE
        plt.figure(figsize=(7, 6))
        plt.axis([xbl, xbr,1.e40,1.e49])
        plt.semilogy(rtotlum,ne20totlum,color='b',label='ne20')

        plt.axhline(y=9.4e42, color='r', linestyle='dotted',label='ne20 (3D value)')

        setxlabel = r"depth of ne20 mixing (cm)"
        setylabel = r"total luminosity (ergs/s)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 16})

        plt.show(block=False)

        plt.savefig('RESULTS/oburn14_totlum_vs_depthofne20mix.png')
        #plt.savefig('RESULTS/oburn14_nuclear_energy_gen_inst.eps')



    def GET1NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l):

        temp09 = self.tt * 1.e-9
        rate = np.exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (rate)

        return timescale

    def GET2NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l, yi, yj, yk):

        temp09 = self.tt * 1.e-9
        rate = np.exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (self.dd * yj * yk * rate / yi)

        return timescale

    def GET3NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l, yi1, yi2):

        temp09 = self.tt * 1.e-9
        rate = np.exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (self.dd * self.dd * yi1 * yi2 * rate * rate)
        # ipp = 200
        # print(timescale[ipp],self.dd[ipp],yi1[ipp],yi2[ipp],rate[ipp],c1l,c2l,c3l,c4l,c5l,c6l,c7l)

        return timescale

    def GETRATEcoeff(self, reaction):

        cl = np.zeros(7)

        if (reaction == 'c12_plus_c12_to_p_na23_r'):
            cl[0] = +0.585029E+02
            cl[1] = +0.295080E-01
            cl[2] = -0.867002E+02
            cl[3] = +0.399457E+01
            cl[4] = -0.592835E+00
            cl[5] = -0.277242E-01
            cl[6] = -0.289561E+01

        if (reaction == 'c12_plus_c12_to_he4_ne20_r'):
            cl[0] = +0.804485E+02
            cl[1] = -0.120189E+00
            cl[2] = -0.723312E+02
            cl[3] = -0.352444E+02
            cl[4] = +0.298646E+01
            cl[5] = -0.309013E+00
            cl[6] = +0.115815E+02

        if (reaction == 'he4_plus_c12_to_o16_r'):
            cl[0] = +0.142191E+03
            cl[1] = -0.891608E+02
            cl[2] = +0.220435E+04
            cl[3] = -0.238031E+04
            cl[4] = +0.108931E+03
            cl[5] = -0.531472E+01
            cl[6] = +0.136118E+04

        if (reaction == 'he4_plus_c12_to_o16_nr'):
            cl[0] = +0.184977E+02
            cl[1] = +0.482093E-02
            cl[2] = -0.332522E+02
            cl[3] = +0.333517E+01
            cl[4] = -0.701714E+00
            cl[5] = +0.781972E-01
            cl[6] = -0.280751E+01

        if (reaction == 'o16_plus_o16_to_p_p31_r'):
            cl[0] = +0.852628E+02
            cl[1] = +0.223453E+00
            cl[2] = -0.145844E+03
            cl[3] = +0.872612E+01
            cl[4] = -0.554035E+00
            cl[5] = -0.137562E+00
            cl[6] = -0.688807E+01

        if (reaction == 'o16_plus_o16_to_he4_si28_r'):
            cl[0] = +0.972435E+02
            cl[1] = -0.268514E+00
            cl[2] = -0.119324E+03
            cl[3] = -0.322497E+02
            cl[4] = +0.146214E+01
            cl[5] = -0.200893E+00
            cl[6] = +0.132148E+02

        if (reaction == 'ne20_to_he4_o16_nv'):
            cl[0] = +0.637915E+02
            cl[1] = -0.549729E+02
            cl[2] = -0.343457E+02
            cl[3] = -0.251939E+02
            cl[4] = +0.479855E+01
            cl[5] = -0.146444E+01
            cl[6] = +0.784333E+01

        if (reaction == 'ne20_to_he4_o16_rv'):
            cl[0] = +0.109310E+03
            cl[1] = -0.727584E+02
            cl[2] = +0.293664E+03
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.201193E+03

        if (reaction == 'si28_to_he4_mg24_nv1'):
            cl[0] = +0.522024E+03
            cl[1] = -0.122258E+03
            cl[2] = +0.434667E+03
            cl[3] = -0.994288E+03
            cl[4] = +0.656308E+02
            cl[5] = -0.412503E+01
            cl[6] = +0.426946E+03

        if (reaction == 'si28_to_he4_mg24_nv2'):
            cl[0] = +0.157580E+02
            cl[1] = -0.129560E+03
            cl[2] = -0.516428E+02
            cl[3] = +0.684625E+02
            cl[4] = -0.386512E+01
            cl[5] = +0.208028E+00
            cl[6] = -0.320727E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv1'):
            cl[0] = -0.906347E+01
            cl[1] = -0.241182E+02
            cl[2] = +0.373526E+01
            cl[3] = -0.664843E+01
            cl[4] = +0.254122E+00
            cl[5] = -0.588282E-02
            cl[6] = +0.191121E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv2'):
            cl[0] = +0.552169E+01
            cl[1] = -0.265651E+02
            cl[2] = +0.456462E-08
            cl[3] = -0.105997E-07
            cl[4] = +0.863175E-09
            cl[5] = -0.640626E-10
            cl[6] = -0.150000E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv3'):
            cl[0] = -0.126553E+01
            cl[1] = -0.287435E+02
            cl[2] = -0.309775E+02
            cl[3] = +0.458298E+02
            cl[4] = -0.272557E+01
            cl[5] = +0.163910E+00
            cl[6] = -0.239582E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv4'):
            cl[0] = +0.296908E+02
            cl[1] = -0.330803E+02
            cl[2] = +0.553217E+02
            cl[3] = -0.737793E+02
            cl[4] = +0.325554E+01
            cl[5] = -0.144379E+00
            cl[6] = +0.388817E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv5'):
            cl[0] = +0.128202E+02
            cl[1] = -0.376275E+02
            cl[2] = -0.487688E+02
            cl[3] = +0.549854E+02
            cl[4] = -0.270916E+01
            cl[5] = +0.142733E+00
            cl[6] = -0.319614E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv6'):
            cl[0] = +0.381739E+02
            cl[1] = -0.406821E+02
            cl[2] = -0.546650E+02
            cl[3] = +0.331135E+02
            cl[4] = -0.644696E+00
            cl[5] = -0.155955E-02
            cl[6] = -0.271330E+02

        if (reaction == 'he4_plus_o16_to_ne20_n'):
            cl[0] = +0.390340E+02
            cl[1] = -0.358600E-01
            cl[2] = -0.343457E+02
            cl[3] = -0.251939E+02
            cl[4] = +0.479855E+01
            cl[5] = -0.146444E+01
            cl[6] = +0.634333E+01

        if (reaction == 'he4_plus_o16_to_ne20_r'):
            cl[0] = +0.845522E+02
            cl[1] = -0.178214E+02
            cl[2] = +0.293664E+03
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.199693E+03

        if (reaction == 'he4_plus_ne20_to_mg24_n'):
            cl[0] = +0.321588E+02
            cl[1] = -0.151494E-01
            cl[2] = -0.446410E+02
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.193576E+01

        if (reaction == 'he4_plus_ne20_to_mg24_r'):
            cl[0] = -0.291641E+03
            cl[1] = -0.120966E+02
            cl[2] = -0.633725E+02
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.121219E+03

        if (reaction == 'mg24_to_he4_ne20_nv'):
            cl[0] = +0.569781E+02
            cl[1] = -0.108074E+03
            cl[2] = -0.446410E+02
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.343576E+01

        if (reaction == 'mg24_to_he4_ne20_rv'):
            cl[0] = -0.266822E+03
            cl[1] = -0.120156E+03
            cl[2] = -0.633725E+02
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.119719E+03

        if (reaction == 'p_plus_na23_to_he4_ne20_n'):
            cl[0] = +0.334868E+03
            cl[1] = -0.247143E+00
            cl[2] = +0.371150E+02
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r1'):
            cl[0] = +0.942806E+02
            cl[1] = -0.312034E+01
            cl[2] = +0.100052E+03
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r2'):
            cl[0] = -0.288152E+02
            cl[1] = -0.447000E+00
            cl[2] = -0.184674E-09
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01

        if (reaction == 'he4_plus_si28_to_c12_ne20_r'):
            cl[0] = -0.307762E+03
            cl[1] = -0.186722E+03
            cl[2] = +0.514197E+03
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'p_plus_p31_to_c12_ne20_r'):
            cl[0] = -0.266452E+03
            cl[1] = -0.156019E+03
            cl[2] = +0.361154E+03
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03

        if (reaction == 'c12_plus_ne20_to_p_p31_r'):
            cl[0] = -0.268136E+03
            cl[1] = -0.387624E+02
            cl[2] = +0.361154E+03
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03

        if (reaction == 'c12_plus_ne20_to_he4_si28_r'):
            cl[0] = -0.308905E+03
            cl[1] = -0.472175E+02
            cl[2] = +0.514197E+03
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'he4_plus_ne20_to_p_na23_n'):
            cl[0] = +0.335091E+03
            cl[1] = -0.278531E+02
            cl[2] = +0.371150E+02
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r1'):
            cl[0] = +0.945037E+02
            cl[1] = -0.307263E+02
            cl[2] = +0.100052E+03
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r2'):
            cl[0] = -0.285920E+02
            cl[1] = -0.280530E+02
            cl[2] = -0.184674E-09
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01

        return cl


    def getInuc(self, network, element):
        inuc_tmp = int(network.index(element))
        if inuc_tmp < 10:
            inuc = '000' + str(inuc_tmp)
        if inuc_tmp >= 10 and inuc_tmp < 100:
            inuc = '00' + str(inuc_tmp)
        if inuc_tmp >= 100 and inuc_tmp < 1000:
            inuc = '0' + str(inuc_tmp)
        return inuc


