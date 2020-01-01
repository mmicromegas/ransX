import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Buoyancy(calc.Calculus, al.SetAxisLimit, object):

    def __init__(self, filename, ig, ieos, intc, data_prefix):
        super(Buoyancy, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        nx = np.asarray(eht.item().get('nx'))
        xzn0 = np.asarray(eht.item().get('xzn0'))
        xznl = np.asarray(eht.item().get('xznl'))
        xznr = np.asarray(eht.item().get('xznr'))

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        pp = np.asarray(eht.item().get('pp')[intc])
        gg = np.asarray(eht.item().get('gg')[intc])
        gamma1 = np.asarray(eht.item().get('gamma1')[intc])

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if (ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        dlnrhodr = self.deriv(np.log(dd), xzn0)
        dlnpdr = self.deriv(np.log(pp), xzn0)
        dlnrhodrs = (1. / gamma1) * dlnpdr
        nsq = gg * (dlnrhodr - dlnrhodrs)

        b = np.zeros(nx)
        dx = xznr - xznl
        for i in range(0, nx):
            b[i] = b[i - 1] + nsq[i] * dx[i]

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.b = b
        self.ig = ig

    def plot_buoyancy(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot buoyancy in the model"""

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.b

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('buoyancy')
        if (self.ig == 1):
            plt.plot(grd1, plt1, color='brown', label=r'$b$')
            setxlabel = r"x (cm)"
        elif (self.ig == 2):
            plt.plot(grd1, plt1, color='brown', label=r'$b$')
            setxlabel = r"r (cm)"
        else:
            print(
                "ERROR(Buoyancy.py): geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

        # define y LABEL
        setylabel = r"$b (s^{-2}$ cm)"

        # show x/y LABELS
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_buoyancy.png')
