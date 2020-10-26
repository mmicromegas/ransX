import numpy as np
import matplotlib.pyplot as plt
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Tools as uT
import UTILS.Errors as eR
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class DensitySpecificVolumeCovarianceEquation(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, ig, intc, data_prefix):
        super(DensitySpecificVolumeCovarianceEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        sv = self.getRAdata(eht, 'sv')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        svux = self.getRAdata(eht, 'svux')[intc]
        svdivu = self.getRAdata(eht, 'svdivu')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_sv = self.getRAdata(eht, 'sv')

        # construct equation-specific mean fields		
        t_b = 1. - t_sv * t_dd

        fht_ux = ddux / dd
        b = 1. - sv * dd

        ##################################################
        # DENSITY-SPECIFIC VOLUME COVARIANCE or B EQUATION 
        ##################################################

        # LHS -db/dt 		
        self.minus_dt_b = self.dt(t_b, xzn0, t_timec, intc)

        # LHS -ux Grad b
        self.minus_ux_gradx_b = ux * self.Grad(b, xzn0)

        # RHS +sv Div dd uxff 
        self.plus_eht_sv_div_dd_uxff = sv * self.Div(dd * (ux - fht_ux), xzn0)

        # RHS -eht_dd Div uxf svf
        self.minus_eht_dd_div_uxf_svf = -dd * self.Div(svux - sv * ux, xzn0)

        # RHS +2 eht_dd eht svf df
        self.plus_two_eht_dd_eht_svf_df = +2. * dd * (svdivu - sv * divu)

        # -res
        self.minus_resBequation = -(self.minus_dt_b + self.minus_ux_gradx_b + self.plus_eht_sv_div_dd_uxff +
                                    self.minus_eht_dd_div_uxf_svf + self.plus_two_eht_dd_eht_svf_df)

        ######################################################
        # END DENSITY-SPECIFIC VOLUME COVARIANCE or B EQUATION 
        ######################################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.sv = sv
        self.dd = dd

    def plot_b(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot density-specific volume covariance stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(DensitySpecificVolumeCovarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = 1. - self.sv * self.dd

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [plt1]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('density-specific volume covariance')
        plt.plot(grd1, plt1, color='brown', label=r'$b$')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"b"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"b"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_b.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'mean_b.eps')

    def plot_b_equation(self, LAXIS, bconv, tconv, xbl, xbr, ybu, ybd, ilg):
        """Plot density-specific volume covariance equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(DensitySpecificVolumeCovarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_b
        lhs1 = self.minus_ux_gradx_b

        rhs0 = self.plus_eht_sv_div_dd_uxff
        rhs1 = self.minus_eht_dd_div_uxf_svf
        rhs2 = self.plus_two_eht_dd_eht_svf_df

        res = self.minus_resBequation

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   
        to_plot = [lhs0, lhs1, rhs0, rhs1, rhs2, res]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('b equation')
        if self.ig == 1:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t b$")
            plt.plot(grd1, lhs1, color='m', label=r"$-\overline{u}_x \partial_x b $")
            plt.plot(grd1, rhs0, color='b', label=r"$+v \nabla_x (\overline{v} \overline{u''_x})$")
            plt.plot(grd1, rhs1, color='g', label=r"$-v \nabla_x (\overline{\rho} \overline{(u'_x v'})$")
            plt.plot(grd1, rhs2, color='r', label=r"$+2 \overline{\rho} \overline{v'd'}$")
            plt.plot(grd1, res, color='k', linestyle='--', label='res')
        elif self.ig == 2:
            plt.plot(grd1, lhs0, color='c', label=r"$-\partial_t b$")
            plt.plot(grd1, lhs1, color='m', label=r"$-\overline{u}_r \partial_r b $")
            plt.plot(grd1, rhs0, color='b', label=r"$+v \nabla_r (\overline{v} \overline{u''_r})$")
            plt.plot(grd1, rhs1, color='g', label=r"$-v \nabla_r (\overline{\rho} \overline{(u'_r v'})$")
            plt.plot(grd1, rhs2, color='r', label=r"$+2 \overline{\rho} \overline{v'd'}$")
            plt.plot(grd1, res, color='k', linestyle='--', label='res')

        # convective boundary markers
        plt.axvline(bconv, linestyle='--', linewidth=0.7, color='k')
        plt.axvline(tconv, linestyle='--', linewidth=0.7, color='k')

        # define and show x/y LABELS
        if self.ig == 1:
            setxlabel = r"x (cm)"
            setylabel = r"b"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)
        elif self.ig == 2:
            setxlabel = r"r (cm)"
            setylabel = r"b"
            plt.xlabel(setxlabel)
            plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=ilg, prop={'size': 12}, ncol=2)

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'b_eq.png')
        plt.savefig('RESULTS/' + self.data_prefix + 'b_eq.eps')
