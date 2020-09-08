import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys
from scipy import integrate

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class VelocitiesMeanExp(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, fext, intc, nsdim, data_prefix):
        super(VelocitiesMeanExp, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        nx = self.getRAdata(eht, 'nx')
        xzn0 = self.getRAdata(eht, 'xzn0')
        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')
        xznl = self.getRAdata(eht, 'xznl')
        xznr = self.getRAdata(eht, 'xznr')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        ux = self.getRAdata(eht, 'ux')[intc]
        dd = self.getRAdata(eht, 'dd')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduxux = self.getRAdata(eht, 'dduxux')[intc]

        Vol = np.zeros(nx)
        # handle volume for different geometries
        if self.ig == 1:
            surface = (yzn0[-1] - yzn0[0]) * (zzn0[-1] - zzn0[0])
            Vol = surface * (xznr - xznl)
            pmass = xzn0[0] * surface * dd[0]
        elif self.ig == 2:
            Vol = 4. / 3. * np.pi * (xznr ** 3 - xznl ** 3)

        #mm = self.getRAdata(eht, 'mm')[intc]
        #print(mm)

        t_timec = self.getRAdata(eht, 'timec')

        # store time series for time derivatives
        t_mm_l = []
        if self.ig == 1:
            #t_mm = self.getRAdata(eht, 'dd')*Vol
            #t_dd = self.getRAdata(eht, 'dd')
            #t_mm = integrate.cumtrapz(self.getRAdata(eht, 'dd')*Vol, xzn0, initial = 0.)
            t_dd = self.getRAdata(eht, 'dd')
            for i in range(t_timec.shape[0]):
                #t_mm_l.append(t_dd[i]*Vol[i])
                t_mm_l.append(integrate.cumtrapz(t_dd[i]*Vol, xzn0, initial = 0.))
                #print(t_dd[i])
                #sys.exit()
            t_mm = np.asarray(t_mm_l)
        elif self.ig == 2:
            t_dd = self.getRAdata(eht, 'dd')
            t_mm = self.getRAdata(eht, 'mm')

        minus_dt_mm = -self.dt(t_mm, xzn0, t_timec, intc)
        plus_vol_dt_dd =  Vol*self.dt(t_dd, xzn0, t_timec, intc)

        #print(minus_dt_mm)
        #print(plus_vol_dt_dd)

        vexp1 = ddux / dd
        if self.ig == 1:
            #vexp2 = (minus_dt_mm + plus_vol_dt_dd)/ (3. * (xzn0 ** 2.) * dd)
            #vexp2 = vexp2/1.e6
            xzn0_exp = np.interp(t_dd[intc+1]*Vol,t_dd[intc]*Vol,xzn0)
            xzn0_exp = np.interp(t_mm[intc+1],t_mm[intc],xzn0)
            dx = xzn0_exp - xzn0
            dt = t_timec[intc+1] - t_timec[intc]
            vexp3 = -dx/dt
        elif self.ig == 2:
            vexp2 = minus_dt_mm / (4. * np.pi * (xzn0 ** 2.) * dd)

            xzn0_exp = np.interp(t_dd[intc+1]*Vol,t_dd[intc]*Vol,xzn0)
            xzn0_exp = np.interp(t_mm[intc+1],t_mm[intc],xzn0)
            dx = xzn0_exp - xzn0
            dt = t_timec[intc+1] - t_timec[intc]
            vexp3 = -dx/dt

        vturb = ((dduxux - ddux * ddux / dd) / dd) ** 0.5

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ux = ux
        self.ig = ig
        self.vexp1 = vexp1
        #self.vexp2 = vexp2
        self.vexp3 = vexp3
        self.vturb = vturb
        self.fext = fext
        self.nsdim = nsdim

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
        #plt3 = self.vexp2
        plt4 = self.vexp3
        plt5 = self.vturb

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1, plt2]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA
        plt.title('velocities ' + str(self.nsdim) + "D")
        if self.ig == 1:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{u}_x$')
            plt.plot(grd1, plt2, color='red', label=r'$\widetilde{u}_x$')
            #plt.plot(grd1, plt3, color='green', linestyle='--',label=r'$v_{exp}$')
            plt.plot(grd1, plt2 - plt1, color='m', label=r"$-\overline{\rho' u'_x}/\overline{\rho}$")
            # plt.plot(grd1,plt4,color='blue',label = r'$u_{turb}$')
        elif self.ig == 2:
            plt.plot(grd1, plt1, color='brown', label=r'$\overline{u}_r$')
            plt.plot(grd1, plt2, color='red', label=r'$\widetilde{u}_r$')
            plt.plot(grd1, plt3, color='green', linestyle='--', label=r'$\overline{v}_{exp} = -\dot{M}/(4 \pi r^2 \rho)$')
            plt.plot(grd1, plt4, color='c', linestyle='--',label=r'$v_{exp}$')
            plt.plot(grd1, plt2-plt1, color='m', label=r"$-\overline{\rho' u'_r}/\overline{\rho}$")
            # plt.plot(grd1,plt4,color='blue',label = r'$u_{turb}$')

        plt.axhline(y=0., linestyle='dotted',color='k')

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
        plt.legend(loc=ilg, prop={'size': 15})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        if self.fext == "png":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_velocities_mean.png')
        if self.fext == "eps":
            plt.savefig('RESULTS/' + self.data_prefix + 'mean_velocities_mean.eps')