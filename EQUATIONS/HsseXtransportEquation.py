import numpy as np
import matplotlib.pyplot as plt
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseXtransportEquation(Calculus, SetAxisLimit, Tools, Errors, object):

    def __init__(self, filename, ig, fext, inuc, element, bconv, tconv, intc, data_prefix):
        super(HsseXtransportEquation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = self.getRAdata(eht, 'dd')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]

        ############################
        # HSSE Xi TRANSPORT EQUATION 
        ############################

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
        self.minus_dt_fht_xi = -self.dt(t_fht_xi, xzn0, t_timec, intc)

        # RHS +fht Xidot 
        self.plus_fht_xidot = +ddxidot / dd

        # RHS -(1/dd)div fxi 
        self.minus_one_o_dd_div_fxi = -(1. / dd) * self.Div(fxi, xzn0)

        # LHS -fht_ux gradx fht_xi
        self.minus_div_eht_dd_fht_ux_fht_xi = -fht_ux * self.Grad(fht_xi, xzn0)

        # -res
        self.minus_resXiTransport = -(self.minus_dt_fht_xi + self.plus_fht_xidot + self.minus_one_o_dd_div_fxi +
                                      self.minus_div_eht_dd_fht_ux_fht_xi)

        ################################
        # END HSSE Xi TRANSPORT EQUATION
        ################################

        self.tau_trans = np.abs(dd*fht_xi/self.Div(fxi,xzn0))
        self.tau_nuc   = np.abs(dd*fht_xi/(ddxidot))

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.inuc = inuc
        self.element = element
        self.ddxi = ddxi

        self.bconv = bconv
        self.tconv = tconv
        self.fext = fext
        self.nx = nx

    def plot_Xrho(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xrho stratification in the model"""

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ddxi

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('rhoX for ' + element)
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{\rho} \widetilde{X}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + element + '.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_rhoX_' + element + '.eps')

    def plot_Xtransport_equation(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xrho transport equation in the model"""

        # convert nuc ID to string
        # xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fht_xi

        rhs0 = self.plus_fht_xidot
        rhs1 = self.minus_one_o_dd_div_fxi
        rhs2 = self.minus_div_eht_dd_fht_ux_fht_xi

        res = self.minus_resXiTransport

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        element_nice = self.setNucNoUp(str(element))

        # plot DATA 
        #plt.title('hsse rhoX transport for ' + element)
        #plt.title('composition equation for ' + element)
        plt.title(str(element_nice))

        # plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t \widetilde{X}_i$')
        # plt.plot(grd1,rhs0,color='g',label=r'$+\widetilde{\dot{X}}^{\rm nuc}_i$')
        # plt.plot(grd1,rhs1,color='b',label=r'$-(1/\overline{\rho}) \nabla_r f_i$')
        # plt.plot(grd1,rhs2,color='y',label=r"$-\widetilde{u}_r \partial_r \widetilde{X}_i$")
        # plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))
        xlimitrange = np.where((grd1 > self.xzn0[0]) & (grd1 < self.xzn0[self.nx-1]))
        xlimitbottom = np.where(grd1 < self.bconv)
        xlimittop = np.where(grd1 > self.tconv)

        plt.plot(grd1[xlimitrange], lhs0[xlimitrange], color='r', label=r'$-\partial_t \widetilde{X}_i$')
        plt.plot(grd1[xlimitrange], rhs0[xlimitrange], color='g', label=r'$+\widetilde{\dot{X}}^{\rm nuc}_i$')
        plt.plot(grd1[xlimitrange], rhs1[xlimitrange], color='b', label=r'$-(1/\overline{\rho}) \nabla_r f_i$')
        plt.plot(grd1[xlimitrange], rhs2[xlimitrange], color='y',
                 label=r"$-\widetilde{u}_r \partial_r \widetilde{X}_i$")
        plt.plot(grd1[xlimitrange], res[xlimitrange], color='k', linestyle='--', label='res')

        # plt.plot(grd1[xlimitbottom], lhs0[xlimitbottom], '.', color='r', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], rhs0[xlimitbottom], '.', color='g', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], rhs1[xlimitbottom], '.', color='b', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], rhs2[xlimitbottom], '.', color='y', markersize=0.5)
        # plt.plot(grd1[xlimitbottom], res[xlimitbottom], '.', color='k', markersize=0.5)

        # plt.plot(grd1[xlimittop], lhs0[xlimittop], '.', color='r', markersize=0.5)
        # plt.plot(grd1[xlimittop], rhs0[xlimittop], '.', color='g', markersize=0.5)
        # plt.plot(grd1[xlimittop], rhs1[xlimittop], '.', color='b', markersize=0.5)
        # plt.plot(grd1[xlimittop], rhs2[xlimittop], '.', color='y', markersize=0.5)
        # plt.plot(grd1[xlimittop], res[xlimittop], '.', color='k', markersize=0.5)

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')



        if element == 's32':
            # restrict grid between self.bconv and self.tconv
            mask = (grd1 >= self.bconv * 1.05) & (grd1 <= self.tconv * 0.85)
            xnuc_grad = self.Grad(rhs0[mask], grd1[mask])

            threshold_pos = 0.1 * np.max(rhs0[mask])
            threshold_neg = 0.1 * np.min(rhs0[mask])

            ind_array_neg2 = np.where(np.diff(np.sign(xnuc_grad)) == -2)[0]
            ind_array_pos2 = np.where(np.diff(np.sign(xnuc_grad)) == 2)[0]
            ind_array = np.sort(np.concatenate((ind_array_neg2, ind_array_pos2)))
            indices_to_plot = [idx for idx in ind_array if
                               (rhs0[mask][idx] > threshold_pos or rhs0[mask][idx] < threshold_neg)]

            if len(indices_to_plot) > 0:
                for idx in indices_to_plot:
                    plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                    plt.text(grd1[mask][idx] * 1.03, rhs0[mask][idx] * 1.03,
                             '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                                                                             self.tau_nuc[mask][idx]),
                             fontsize=15, color='k')
            elif (rhs0[mask][0] > threshold_pos or rhs0[mask][0] < threshold_neg):
                plt.plot(grd1[mask][0], rhs0[mask][0], 'ko', markersize=8)
                plt.text(grd1[mask][0], rhs0[mask][0] * 1.5,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][0],
                                                                         self.tau_nuc[mask][0]),
                         fontsize=14, color='k')
        elif element == 'mg24':
            # restrict grid between self.bconv and self.tconv
            mask = (grd1 >= self.bconv * 1.015) & (grd1 <= self.tconv * 0.85)
            xnuc_grad = self.Grad(rhs0[mask], grd1[mask])

            threshold_pos = 0.02 * np.max(rhs0[mask])
            threshold_neg = 0.02 * np.min(rhs0[mask])

            ind_array_neg2 = np.where(np.diff(np.sign(xnuc_grad)) == -2)[0]
            ind_array_pos2 = np.where(np.diff(np.sign(xnuc_grad)) == 2)[0]
            ind_array = np.sort(np.concatenate((ind_array_neg2, ind_array_pos2)))
            indices_to_plot = [idx for idx in ind_array if
                               (rhs0[mask][idx] > threshold_pos or rhs0[mask][idx] < threshold_neg)]

            if len(indices_to_plot) > 0:
                #for idx in indices_to_plot:
                    # plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                    # plt.text(grd1[mask][idx] * 1.03, rhs0[mask][idx] * 1.03,
                    #          '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                    #                                                          self.tau_nuc[mask][idx]),
                    #          fontsize=15, color='k')

                idx = indices_to_plot[0]
                plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                plt.text(grd1[mask][idx] * 1.03, rhs0[mask][idx] * 1.03,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                                                                         self.tau_nuc[mask][idx]),
                         fontsize=15, color='k')

                idx = indices_to_plot[1]
                plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                plt.text(grd1[mask][idx] * 1.03, np.abs(rhs0[mask][idx]) * 5.,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                                                                         self.tau_nuc[mask][idx]),
                         fontsize=15, color='k')


            elif (rhs0[mask][0] > threshold_pos or rhs0[mask][0] < threshold_neg):
                plt.plot(grd1[mask][0], rhs0[mask][0], 'ko', markersize=8)
                plt.text(grd1[mask][0], rhs0[mask][0] * 1.5,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][0],
                                                                         self.tau_nuc[mask][0]),
                         fontsize=14, color='k')
        elif element == 'na23':
            # restrict grid between self.bconv and self.tconv
            mask = (grd1 >= self.bconv * 1.2) & (grd1 <= self.tconv * 0.85)
            xnuc_grad = self.Grad(rhs0[mask], grd1[mask])

            threshold_pos = 0.1 * np.max(rhs0[mask])
            threshold_neg = 0.1 * np.min(rhs0[mask])

            ind_array_neg2 = np.where(np.diff(np.sign(xnuc_grad)) == -2)[0]
            ind_array_pos2 = np.where(np.diff(np.sign(xnuc_grad)) == 2)[0]
            ind_array = np.sort(np.concatenate((ind_array_neg2, ind_array_pos2)))
            indices_to_plot = [idx for idx in ind_array if
                               (rhs0[mask][idx] > threshold_pos or rhs0[mask][idx] < threshold_neg)]

            if len(indices_to_plot) > 0:
                for idx in indices_to_plot:
                    plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                    plt.text(grd1[mask][idx] * 1.03, rhs0[mask][idx] * 1.03,
                             '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                                                                             self.tau_nuc[mask][idx]),
                             fontsize=15, color='k')
            elif (rhs0[mask][0] > threshold_pos or rhs0[mask][0] < threshold_neg):
                plt.plot(grd1[mask][0], rhs0[mask][0], 'ko', markersize=8)
                plt.text(grd1[mask][0], rhs0[mask][0] * 1.5,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][0],
                                                                         self.tau_nuc[mask][0]),
                         fontsize=14, color='k')
        elif element == 'ne20':
            # restrict grid between self.bconv and self.tconv
            mask = (grd1 >= self.bconv * 1.4) & (grd1 <= self.tconv * 0.85)
            xnuc_grad = self.Grad(rhs0[mask], grd1[mask])

            threshold_pos = 0.1 * np.max(rhs0[mask])
            threshold_neg = 0.1 * np.min(rhs0[mask])

            ind_array_neg2 = np.where(np.diff(np.sign(xnuc_grad)) == -2)[0]
            ind_array_pos2 = np.where(np.diff(np.sign(xnuc_grad)) == 2)[0]
            ind_array = np.sort(np.concatenate((ind_array_neg2, ind_array_pos2)))
            indices_to_plot = [idx for idx in ind_array if
                               (rhs0[mask][idx] > threshold_pos or rhs0[mask][idx] < threshold_neg)]

            if len(indices_to_plot) > 0:
                for idx in indices_to_plot:
                    plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                    plt.text(grd1[mask][idx] * 1.03, rhs0[mask][idx] * 1.03,
                             '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                                                                             self.tau_nuc[mask][idx]),
                             fontsize=15, color='k')
            elif (rhs0[mask][0] > threshold_pos or rhs0[mask][0] < threshold_neg):
                plt.plot(grd1[mask][0], rhs0[mask][0], 'ko', markersize=8)
                plt.text(grd1[mask][0], rhs0[mask][0] * 1.5,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][0],
                                                                         self.tau_nuc[mask][0]),
                         fontsize=14, color='k')
        elif element == 'ar36':
            # restrict grid between self.bconv and self.tconv
            mask = (grd1 >= self.bconv * 1.015) & (grd1 <= self.tconv * 0.85)
            xnuc_grad = self.Grad(rhs0[mask], grd1[mask])

            threshold_pos = 0.1 * np.max(rhs0[mask])
            threshold_neg = 1. * np.min(rhs0[mask])

            ind_array_neg2 = np.where(np.diff(np.sign(xnuc_grad)) == -2)[0]
            ind_array_pos2 = np.where(np.diff(np.sign(xnuc_grad)) == 2)[0]
            ind_array = np.sort(np.concatenate((ind_array_neg2, ind_array_pos2)))
            indices_to_plot = [idx for idx in ind_array if
                               (rhs0[mask][idx] > threshold_pos or rhs0[mask][idx] < threshold_neg)]

            if len(indices_to_plot) > 0:
                for idx in indices_to_plot:
                    plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                    plt.text(grd1[mask][idx] * 1.03, rhs0[mask][idx] * 1.03,
                             '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                                                                             self.tau_nuc[mask][idx]),
                             fontsize=15, color='k')
            elif (rhs0[mask][0] > threshold_pos or rhs0[mask][0] < threshold_neg):
                plt.plot(grd1[mask][0], rhs0[mask][0], 'ko', markersize=8)
                plt.text(grd1[mask][0], rhs0[mask][0] * 1.5,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][0],
                                                                         self.tau_nuc[mask][0]),
                         fontsize=14, color='k')

        else:
            # restrict grid between self.bconv and self.tconv
            mask = (grd1 >= self.bconv * 1.015) & (grd1 <= self.tconv * 0.85)
            xnuc_grad = self.Grad(rhs0[mask], grd1[mask])

            threshold_pos = 0.12 * np.max(rhs0[mask])
            threshold_neg = 0.12 * np.min(rhs0[mask])

            ind_array_neg2 = np.where(np.diff(np.sign(xnuc_grad)) == -2)[0]
            ind_array_pos2 = np.where(np.diff(np.sign(xnuc_grad)) == 2)[0]
            ind_array = np.sort(np.concatenate((ind_array_neg2, ind_array_pos2)))
            indices_to_plot = [idx for idx in ind_array if
                               (rhs0[mask][idx] > threshold_pos or rhs0[mask][idx] < threshold_neg)]

            if len(indices_to_plot) > 0:
                for idx in indices_to_plot:
                    plt.plot(grd1[mask][idx], rhs0[mask][idx], 'ko', markersize=8)
                    plt.text(grd1[mask][idx] * 1.03, rhs0[mask][idx] * 1.03,
                             '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][idx],
                                                                             self.tau_nuc[mask][idx]),
                             fontsize=15, color='k')
            elif (rhs0[mask][0] > threshold_pos or rhs0[mask][0] < threshold_neg):
                plt.plot(grd1[mask][0], rhs0[mask][0], 'ko', markersize=8)
                plt.text(grd1[mask][0], rhs0[mask][0] * 1.5,
                         '$\\tau_{trans}$=%.0fs\n$\\tau_{nuc}$=%.0fs' % (self.tau_trans[mask][0],
                                                                         self.tau_nuc[mask][0]),
                         fontsize=14, color='k')


        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"s$^{-1}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        #plt.legend(loc=ilg, prop={'size': 14}, ncol=2)
        plt.legend(loc=ilg, prop={'size': 12}, ncol=1)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == 'png':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_mean_Xtransport_' + element + '.png')
        elif self.fext == 'eps':
            plt.savefig('RESULTS/' + self.data_prefix + 'hsse_mean_Xtransport_' + element + '.eps')

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