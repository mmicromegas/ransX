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

class TurbulentKineticEnergyEquationEvolution(uCalc.Calculus, uEal.ALIMITevol, uT.Tools, eR.Errors, object):

    def __init__(self, dataout, ig, data_prefix):
        super(TurbulentKineticEnergyEquationEvolution, self).__init__(ig)

        # load data to structured array
        eht = np.load(dataout)

        # load temporal evolution
        t_timec = self.getRAdata(eht, 't_timec')
        t_TKEsum = self.getRAdata(eht, 't_TKEsum')
        t_epsD = self.getRAdata(eht, 't_epsD')
        t_xzn0inc = self.getRAdata(eht, 't_xzn0inc')
        t_xzn0outc = self.getRAdata(eht, 't_xzn0outc')

        t_x0002mean_cnvz = self.getRAdata(eht, 't_x0002mean_cnvz')

        # share data across the whole class
        self.t_timec = t_timec
        self.t_TKEsum = t_TKEsum
        self.t_epsD = t_epsD
        self.t_xzn0inc = t_xzn0inc
        self.t_xzn0outc = t_xzn0outc
        self.data_prefix = data_prefix

        self.t_x0002mean_cnvz = t_x0002mean_cnvz

    def plot_tke_evolution(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        grd1 = self.t_timec
        plt1 = self.t_TKEsum
        plt2 = self.t_epsD

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('turbulent kinetic energy evolution')
        plt.plot(grd1, plt1, color='r', label=r'$tke$')
        # plt.plot(grd1, plt2, color='g', label=r'$epsD$')

        # vertical markers
        t1 = 460.
        t2 = 760.
        t3 = 1060.
        plt.axvline(t1, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(t2, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(t3, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"ergs"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 16})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tke_evol.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'tke_evol.eps')

    def plot_x0002(self):
        # get data
        t_timec = self.t_timec
        t_x = self.t_x0002mean_cnvz

        # create FIGURE
        plt.figure(figsize=(7, 5))

        plt.axis([0., 1400., -0.005, 0.09])

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # plot DATA 
        plt.title(r'bottom 2/3 of cnvz ($256^3$)')
        plt.plot(t_timec, t_x, color='r', label=r'$X_2$')

        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"avg(X2)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=1, prop={'size': 14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'x0002mean_evol.png')
