import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XdamkohlerNumber(calc.CALCULUS, al.ALIMIT, object):

    def __init__(self, filename, ig, inuc, element, bconv, tconv, intc, data_prefix):
        super(XdamkohlerNumber, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0'))
        nx = np.asarray(eht.item().get('nx'))

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		
        # assign global data to be shared across whole class

        dd = np.asarray(eht.item().get('dd')[intc])
        ddxi = np.asarray(eht.item().get('ddx' + inuc)[intc])
        ddux = np.asarray(eht.item().get('ddux')[intc])
        ddxiux = np.asarray(eht.item().get('ddx' + inuc + 'ux')[intc])
        ddxidot = np.asarray(eht.item().get('ddx' + inuc + 'dot')[intc])

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        fxi = ddxiux - ddxi * ddux / dd

        # calculate damkohler number 		
        tau_trans = fht_xi / self.Div(fxi / dd, xzn0)
        tau_nuc = fht_xi / (ddxidot / dd)

        # Damkohler number		
        self.xda = tau_trans / tau_nuc

        self.data_prefix = data_prefix
        self.xzn0 = np.asarray(eht.item().get('xzn0'))
        self.element = element
        self.inuc = inuc
        self.bconv = bconv
        self.tconv = tconv

    def plot_Xda(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # Damkohler number

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element

        # load x GRID
        grd1 = self.xzn0

        # get data
        plt0 = self.xda

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # set plot boundaries   
        to_plot = [plt0]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 		
        plt.title(r"$Damk\"ohler \ number \ for \ $" + self.element)
        plt.plot(grd1, plt0, label=r"$D_{a}^i = \tau_{trans}^i / \tau_{nuc}^i$", color='r')

        # convective boundary markers
        plt.axvline(self.bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(self.tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$D_a^i$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 15})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'DamkohlerNo_' + element + '.png')
