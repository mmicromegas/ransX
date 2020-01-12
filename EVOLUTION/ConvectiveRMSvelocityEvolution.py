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

class ConvectiveRMSvelocityEvolution(uCalc.Calculus, uEal.ALIMITevol, uT.Tools, eR.Errors, object):

    def __init__(self, dataout, ig, data_prefix):
        super(ConvectiveRMSvelocityEvolution, self).__init__(ig)

        # load data to structured array
        eht = np.load(dataout)

        # load temporal evolution
        t_timec = self.getRAdata(eht, 't_timec')
        t_urms = self.getRAdata(eht, 't_urms')

        # share data across the whole class
        self.t_timec = t_timec
        self.data_prefix = data_prefix

        self.t_urms = t_urms

    def plot_turms_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # get data
        grd1 = self.t_timec
        plt1 = self.t_urms

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title(r'convective rms velocity')
        plt.plot(grd1, plt1, color='r', label=r'u$_{rms}$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"u (cm/s)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'turms_evol.png')
