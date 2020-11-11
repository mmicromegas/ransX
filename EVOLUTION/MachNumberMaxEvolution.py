import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.EVOL.ALIMITevol as uEal
import UTILS.Tools as uT
import UTILS.Errors as eR


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class MachNumberMaxEvolution(uCalc.Calculus, uEal.ALIMITevol, uT.Tools, eR.Errors, object):

    def __init__(self, dataout, ig, data_prefix):
        super(MachNumberMaxEvolution, self).__init__(ig)

        # load data to structured array
        eht = np.load(dataout,allow_pickle=True)

        # load temporal evolution
        t_timec = self.getRAdata(eht, 't_timec')
        t_machMax = self.getRAdata(eht, 't_machMax')

        # share data across the whole class
        self.t_timec = t_timec
        self.data_prefix = data_prefix

        self.t_machMax = t_machMax

    def plot_machmax_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # get data
        grd1 = self.t_timec
        plt1 = self.t_machMax

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'maximum Mach number')
        plt.plot(grd1, plt1, color='r', label=r'mach')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"mach"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tmachMax_evol.png')
