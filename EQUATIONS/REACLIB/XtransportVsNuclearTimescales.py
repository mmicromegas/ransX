import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XtransportVsNuclearTimescales(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename_rans, filename_reaclib, ig, inuc, element, bconv, tconv, tc, intc, data_prefix,
                 fext, tnuc, network):
        super(XtransportVsNuclearTimescales, self).__init__(ig)

        # load RANS data to structured array
        eht = np.load(filename_rans)

        # load REACLIB data	

        rcoeff_tmp = []
        rlabel = []

        # read line-by-line		
        with open(filename_reaclib) as handle:
            for lineno, line in enumerate(handle):
                if (lineno <> 0) and (lineno % 3 <> 0):
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
        # if self.fext == "png":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.png')
        # if self.fext == "eps":
        #    plt.savefig('RESULTS/' + self.data_prefix + 'mean_Xtransport_' + element + '.eps')

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
        plt.plot(grd1[xlimitrange], plt3[xlimitrange], label=r"$|\tau_{X}^i|$", color='brown')

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

        if self.tnuc == 1:

            network = self.network

            ddxidict = {}
            fht_yi_list = []
            for rl in rlabel:
                if element in rl:
                    rc = rcoeffdict[rl]
                    if rl[0:2] == '1-':
                        plt.plot(grd1, self.GET1NUCtimescale(rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6]), label=rl,
                                 linestyle='--')
                    if rl[0:2] == '2-':
                        rlsplit = rl.split('-')
                        print(rlsplit)
                        for elem in rlsplit:
                            if elem in network:
                                # print('ele in network: ' + elem)
                                inuc = self.getInuc(network, elem)
                                # fht_yi_list.append((self.getRAdata(eht,'ddx'+inuc)[intc])/dd))
                                fht_yi_list.append(((self.getRAdata(eht, 'ddx' + inuc)[intc] / dd) / float(inuc)))
                                # ddxi = {ele : self.getRAdata(eht,'ddx'+inuc)[intc])}
                                # ddxidict.update(ddxi)
                        if rlsplit[3] == element:
                            plt.plot(grd1,
                                     self.GET2NUCtimescale(rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], fht_yi_list[2],
                                                           fht_yi_list[0], fht_yi_list[1]), label=rl, linestyle=':')
                        if rlsplit[4] == element:
                            plt.plot(grd1,
                                     self.GET2NUCtimescale(rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], fht_yi_list[3],
                                                           fht_yi_list[0], fht_yi_list[1]), label=rl, linestyle=':')
                        # if rlsplit[1] == element:
                        #    plt.plot(grd1,self.GET2NUCtimescale(rc[0],rc[1],rc[2],rc[3],rc[4],rc[5],rc[6],fht_yi_list[0],fht_yi_list[1],fht_yi_list[2]),label=rl,linestyle=':')
                        # if rlsplit[2] == element:
                        #    plt.plot(grd1,self.GET2NUCtimescale(rc[0],rc[1],rc[2],rc[3],rc[4],rc[5],rc[6],fht_yi_list[0],fht_yi_list[1],fht_yi_list[2]),label=rl,linestyle=':')

                    if rl[0:2] == '3-':
                        rlsplit = rl.split('-')
                        # print(rlsplit)
                        for elem in rlsplit:
                            if elem in network:
                                # print('ele in network: ' + elem)
                                inuc = self.getInuc(network, elem)
                                # fht_yi_list.append((self.getRAdata(eht,'ddx'+inuc)[intc]/dd))
                                fht_yi_list.append(((self.getRAdata(eht, 'ddx' + inuc)[intc] / dd) / float(inuc)))
                        if rlsplit[4] == element:
                            plt.plot(grd1,
                                     self.GET3NUCtimescale(rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], fht_yi_list[3],
                                                           fht_yi_list[4]), label=rl, linestyle=':')
                            # print(rc[0],rc[1],rc[2],rc[3],rc[4],rc[5],rc[6])

        elif self.tnuc == 0:
            print("MESSAGE(XtransportVsNuclearTimescalesEquation.py) : Omitting to plot nuclear timescales.")
        else:
            print("ERROR(XtransportVsNuclearTimescalesEquation.py):" + self.errorNuclTimescaleMode(self.tnuc))
            sys.exit()

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"$\tau (s)$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"$\tau (s)$"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 11})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'xTimescales_' + element + '.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'xTimescales_' + element + '.eps')

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

    def getInuc(self, network, element):
        inuc_tmp = int(network.index(element))
        if inuc_tmp < 10:
            inuc = '000' + str(inuc_tmp)
        if inuc_tmp >= 10 and inuc_tmp < 100:
            inuc = '00' + str(inuc_tmp)
        if inuc_tmp >= 100 and inuc_tmp < 1000:
            inuc = '0' + str(inuc_tmp)
        return inuc
