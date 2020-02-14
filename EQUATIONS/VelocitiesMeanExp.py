import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class VelocitiesMeanExp(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(VelocitiesMeanExp, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        ux = self.getRAdata(eht, 'ux')[intc]
        dd = self.getRAdata(eht, 'dd')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduxux = self.getRAdata(eht, 'dduxux')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_mm = self.getRAdata(eht, 'mm')

        minus_dt_mm = -self.dt(t_mm, xzn0, t_timec, intc)

        vexp1 = ddux / dd
        vexp2 = minus_dt_mm / (4. * np.pi * (xzn0 ** 2.) * dd)
        vturb = ((dduxux - ddux * ddux / dd) / dd) ** 0.5

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ux = ux
        self.ig = ig
        self.vexp1 = vexp1
        self.vexp2 = vexp2
        self.vturb = vturb

    def plot_velocities(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot velocities in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(VelocitiesMeanExp.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ux
        plt2 = self.vexp1
        plt3 = self.vexp2
        plt4 = self.vturb

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2, plt3]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('velocities')
        plt.plot(grd1, plt1, color='brown', label=r'$\overline{u}_r$')
        plt.plot(grd1, plt2, color='red', label=r'$\widetilde{u}_r$')
        # plt.plot(grd1, plt3, color='green', linestyle='--', label=r'$\overline{v}_{exp} = -\dot{M}/(4 \pi r^2 \rho)$')
        # plt.plot(grd1,plt4,color='blue',label = r'$u_{turb}$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"velocity (cm s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"velocity (cm s$^{-1}$)"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_velocities_mean.png')
