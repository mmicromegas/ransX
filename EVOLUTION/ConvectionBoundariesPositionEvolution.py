import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as calc
import UTILS.EVOL.ALIMITevol as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ConvectionBoundariesPositionEvolution(calc.Calculus, al.ALIMITevol, object):

    def __init__(self, dataout, ig, data_prefix):
        super(ConvectionBoundariesPositionEvolution, self).__init__(ig)

        # load data to structured array
        eht = np.load(dataout)

        # load temporal evolution
        t_timec = np.asarray(eht.item().get('t_timec'))
        t_xzn0inc = np.asarray(eht.item().get('t_xzn0inc'))
        t_xzn0outc = np.asarray(eht.item().get('t_xzn0outc'))

        # share data across the whole class
        self.t_timec = t_timec
        self.t_xzn0inc = t_xzn0inc
        self.t_xzn0outc = t_xzn0outc
        self.data_prefix = data_prefix

    def plot_conv_bndry_location(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        # get data
        grd1 = self.t_timec
        plt1 = self.t_xzn0inc
        plt2 = self.t_xzn0outc
        plt3 = plt2 - plt1

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('convection boundary')
        plt.plot(grd1, plt1, color='r', label=r'$inner$')
        plt.plot(grd1, plt2, color='g', label=r'$outer$')
        plt.plot(grd1, plt3, color='b', label=r'$l_c$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"cm"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'cnvzboundary_evol.png')
