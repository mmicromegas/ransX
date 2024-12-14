# class for RANS ContinuityEquationWithFavrianDilatation #

import numpy as np
import sys
from scipy import integrate
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors

from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ContinuityEquationWithFavrianDilatation(Calculus, Tools, Errors, object):

    def __init__(self, filename, ig, fext, intc, nsdim, data_prefix):
        super(ContinuityEquationWithFavrianDilatation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        mm = self.getRAdata(eht, 'mm')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd

        #############################################
        # CONTINUITY EQUATION WITH FAVRIAN DILATATION
        #############################################

        # LHS -dq/dt 		
        self.minus_dt_dd = -self.dt(t_dd, xzn0, t_timec, intc)

        # LHS -fht_ux Grad dd
        self.minus_fht_ux_grad_dd = -fht_ux * self.Grad(dd, xzn0)

        # RHS -dd Div fht_ux 
        self.minus_dd_div_fht_ux = -dd * self.Div(fht_ux, xzn0)

        # -res
        self.minus_resContEquation = -(self.minus_dt_dd + self.minus_fht_ux_grad_dd + self.minus_dd_div_fht_ux)

        #################################################
        # END CONTINUITY EQUATION WITH FAVRIAN DILATATION
        #################################################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0
        self.dd = dd
        self.nx = nx
        self.ig = ig
        self.fext = fext
        self.mm = mm
        self.nsdim = nsdim

    def plot_ContinuityEquationWithFavrianDilatation(self, laxis, bconv, tconv, xbl, xbr, ybuBgr, ybdBgr, ybuEq, ybdEq, ybuBar, ybdBar,
                                                     plabel, ilg):
        """Plot rho stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        nx = self.nx
        xzn0 = self.xzn0
        yzn0 = self.yzn0
        zzn0 = self.zzn0
        nsdim = self.nsdim

        # load BACKGROUND to plot
        plt1 = self.dd

        # load EQUATION
        lhs0 = self.minus_dt_dd
        lhs1 = self.minus_fht_ux_grad_dd
        rhs0 = self.minus_dd_div_fht_ux
        res = self.minus_resContEquation

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # calculate integral budgets
        terms = [lhs0,lhs1,rhs0,res]
        int_terms = self.calcIntegralBudget(terms, xbl, xbr, nx, xzn0, yzn0, zzn0, nsdim, plabel, laxis, self.ig)

        eQterms = [r"$-$",r"$-$",
                 r"$-$", r"$+res$"]


        #eQterms = [r"$-\partial_t \overline{\rho}$",r"$-\widetilde{u}_x \partial_x \overline{\rho} $",
        #         r"$-\overline{\rho}\widetilde{d}$", r"$+res$"]
        # Plot


        fig = make_subplots(
            rows=1, cols=3)

        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[4, 5, 6],name=r"$$-\partial_t$$"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[40, 50, 60]),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
            row=1, col=3
        )

        #fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=18)))

        #fig.update_layout(legend=dict(
        #    yanchor="top",
        #    y=0.99,
        #    xanchor="left",
        #    x=0.01
        #))

        return fig
