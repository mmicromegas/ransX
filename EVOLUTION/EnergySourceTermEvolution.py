import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.EVOL.ALIMITevol as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class EnergySourceTermEvolution(calc.Calculus, al.ALIMITevol, object):

    def __init__(self, dataout, ig, data_prefix):
        super(EnergySourceTermEvolution, self).__init__(ig)

        # load data to structured array
        eht = np.load(dataout)

        # load temporal evolution
        t_timec = np.asarray(eht.item().get('t_timec'))
        t_tenuc = np.asarray(eht.item().get('t_tenuc'))

        # share data across the whole class
        self.t_timec = t_timec
        self.data_prefix = data_prefix

        self.t_tenuc = t_tenuc

    def plot_tenuc_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # get data
        grd1 = self.t_timec
        plt1 = self.t_tenuc

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'energy source term')
        plt.plot(grd1, plt1, color='r', label=r'$tenuc$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"ergs/s"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tenuc_evol.png')
