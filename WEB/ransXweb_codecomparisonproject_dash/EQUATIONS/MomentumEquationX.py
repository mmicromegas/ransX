import numpy as np
import sys
from UTILS.Calculus import Calculus
from UTILS.Tools import Tools
from UTILS.Errors import Errors

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class MomentumEquationX(Calculus, Tools, Errors, object):

    def __init__(self, filename, plabel, code, ig, fext, intc, nsdim, data_prefix):
        super(MomentumEquationX, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        nx = self.getRAdata(eht, 'nx')
        ny = nx
        nz = nx

        xzn0 = self.getRAdata(eht, 'xzn0')
        dx = xzn0/nx
        xznl = xzn0 - dx/2.
        xznr = xzn0 + dx/2.

        yzn0 = np.linspace(0.,2.,nx)
        zzn0 = np.linspace(0.,2.,nx)

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        gg = self.getRAdata(eht, 'gg')[intc]

        ddgg = self.getRAdata(eht, 'ddgg')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_ddux = self.getRAdata(eht, 'ddux')

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        rxx = dduxux - ddux * ddux / dd

        #####################
        # X MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_ddux = -self.dt(t_ddux, xzn0, t_timec, intc)

        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_ux = -self.Div(dd * fht_ux * fht_ux, xzn0)

        # RHS -div rxx
        self.minus_div_rxx = -self.Div(rxx, xzn0)

        # RHS -G
        if self.ig == 1:
            self.minus_G = np.zeros(nx)
        elif self.ig == 2:
            self.minus_G = -(-dduyuy - dduzuz) / xzn0

        # RHS -(grad P - rho g)
        #self.minus_gradx_pp_eht_dd_eht_gg = -self.Grad(pp,xzn0) +dd*gg
        self.minus_gradx_pp_eht_dd_eht_gg = -self.Grad(pp, xzn0) + ddgg

        # for i in range(nx):
        #    print(2.*ddgg[i],dd[i]*gg[i]		

        # -res
        self.minus_resResXmomentumEquation = \
            -(self.minus_dt_ddux + self.minus_div_eht_dd_fht_ux_fht_ux + self.minus_div_rxx
              + self.minus_G + self.minus_gradx_pp_eht_dd_eht_gg)

        #########################
        # END X MOMENTUM EQUATION 
        #########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.ddux = ddux
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0
        self.dd = dd
        self.nx = nx
        self.ig = ig
        self.fext = fext
        self.nsdim = nsdim
        self.plabel = plabel
        self.code = code

    def plot_MomentumEquationX(self, laxis, bconv, tconv, xbl, xbr, ybuBgr, ybdBgr, ybuEq, ybdEq, ybuBar, ybdBar,
                                                     ilg):
        """Plot momentum stratification in the model"""

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
        plabel = self.plabel
        code = self.code

        #print(code)

        # hack so the x-axis is the same for all the codes
        xzn0_l = xzn0
        xbl_l = xbl
        xbr_l = xbr
        if code == 'PROMPI':
            xzn0_l = xzn0
            xbl_l = xbl
            xbr_l = xbr
            bconv = bconv
            tconv = tconv
        elif code == 'FLASH':
            xzn0_l = xzn0
            xbl_l = xbl
            xbr_l = xbr
            bconv = bconv
            tconv = tconv
        elif code == 'MUSIC':
            xzn0_l = xzn0 + 1.
            xbl_l = xbl + 1.
            xbr_l = xbr + 1.
            bconv = bconv + 1.
            tconv = tconv + 1.
        elif code == 'SLH':
            xzn0_l = xzn0 + 2.
            xbl_l = xbl + 2.
            xbr_l = xbr + 2.
            bconv = bconv + 2.
            tconv = tconv + 2.

        # load BACKGROUND to plot
        plt1 = self.ddux

        # load EQUATION
        lhs0 = self.minus_dt_ddux
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_ux
        rhs0 = self.minus_div_rxx
        rhs1 = self.minus_G
        rhs2 = self.minus_gradx_pp_eht_dd_eht_gg
        res = self.minus_resResXmomentumEquation

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(ContinuityEquationWithFavrianDilatation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # calculate integral budgets
        terms = [lhs0,lhs1,rhs0,rhs2,res]
        int_terms = self.calcIntegralBudget(terms, xbl_l, xbr_l, nx, xzn0_l, yzn0, zzn0, nsdim, plabel, laxis, self.ig)

        eQterms = [r"$-\partial_t ( \overline{\rho} \widetilde{u}_x )$",r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{u}_x ) $",
                   r"$-\nabla_x (\widetilde{R}_{xx})$", r"$-(\partial_x \overline{P} - \bar{\rho}\tilde{g}_x)$", r"$+res$"]

        # Plot
        title1 = code
        title2 = "momentum x" + r"$\overline{\rho} \widetilde{u}_x$"
        title3 = "integral budget"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=(title1, title2, title3))
        # 1st subplot
        fig.append_trace(
            go.Scatter(x=xzn0_l, y=lhs0, name=eQterms[0],
                       line=dict(color='cyan'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=lhs1, name=eQterms[1],
                       line=dict(color='magenta'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs0, name=eQterms[2],
                       line=dict(color='blue'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs2, name=eQterms[3],
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=res, name=eQterms[4],
                       line=dict(color='black', dash='dash'),hoverinfo='none'),
            row=1, col=1)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl_l, xbr_l], tickangle=-45, row=1, col=1,
                         tickwidth = 2,ticklen = 10, nticks = 10, showgrid = True, showline = True, linewidth = 1,
                         linecolor = 'black', mirror = True, ticks = 'outside')
        fig.update_yaxes(title_text="ccp units", range=[ybdEq, ybuEq], tickangle=-45, exponentformat='e',
                         row=1, col=1, tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside')

        # 2nd subplot
        fig.append_trace(
            go.Scatter(x=xzn0_l, y=plt1,
                       line=dict(color='blue'), hoverinfo='none',showlegend=False), row=1, col=2)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl_l, xbr_l], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10,
                         showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)
        fig.update_yaxes(title_text="ccp units", range=[ybdBgr, ybuBgr], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True, exponentformat='e',
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)

        # 3rd subplot
        fig.append_trace(go.Bar(
            x=eQterms,y=int_terms,
            orientation='v', showlegend=False, hoverinfo='none'), row=1, col=3)
        fig.update_yaxes(title_text="ccp units", range=[ybdBar, ybuBar], exponentformat='e', tickangle=-45,
                         tickwidth=2, ticks='outside',showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=3)
        fig.update_xaxes(title_text=r'', showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=3)


        # show
        fig.update_layout(height=550, width=1400, font=dict(size=14), xaxis_tickangle=-45, yaxis_tickangle=-45)
        fig.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor='rgba(0,0,0,0)',font=dict(size=18)),
                          title=r"$\partial_t ( \overline{\rho} \widetilde{u}_x) = -\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{u}_x)-\nabla_x (\widetilde{R}_{xx}) "
                                r"-(\partial_x \overline{P} - \bar{\rho}\tilde{g}_x)  + res$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]),xaxis2=dict(domain=[0.37, 0.64]),xaxis3=dict(domain=[0.74, 1.]))


        return fig
