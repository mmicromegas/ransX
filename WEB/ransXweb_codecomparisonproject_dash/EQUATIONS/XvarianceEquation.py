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

class XvarianceEquation(Calculus, Tools, Errors, object):

    def __init__(self, filename, plabel, code, ig, fext, inuc, element, bconv, tconv, intc, nsdim, data_prefix):
        super(XvarianceEquation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename,allow_pickle=True)

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

        nnuc = self.getRAdata(eht, 'nnuc')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		

        dd = self.getRAdata(eht, 'dd')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = np.zeros(nx)
        ddxisq = self.getRAdata(eht, 'ddx' + inuc + 'sq')[intc]
        ddxisqux = self.getRAdata(eht, 'ddx' + inuc + 'squx')[intc]

        ddxixidot = np.zeros(nx)

        ######################
        # Xi VARIANCE EQUATION 
        ######################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_ddxisq = self.getRAdata(eht, 'ddx' + inuc + 'sq')

        # construct equation-specific mean fields
        t_eht_dd_sigmai = t_ddxisq - t_ddxi * t_ddxi / t_dd

        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        sigmai = (ddxisq - ddxi * ddxi / dd) / dd
        fsigmai = ddxisqux - 2. * ddxiux * ddxi / dd - ddxisq * ddux / dd + 2. * ddxi * ddxi * ddux / (dd * dd)
        fxi = ddxiux - ddxi * ddux / dd

        # LHS -dq/dt 
        self.minus_dt_eht_dd_sigmai = -self.dt(t_eht_dd_sigmai, xzn0, t_timec, intc)

        # LHS -div(dduxsigmai)
        self.minus_div_eht_dd_fht_ux_sigmai = -self.Div(dd * fht_ux * sigmai, xzn0)

        # RHS -div fsigmai
        self.minus_div_fsigmai = -self.Div(fsigmai, xzn0)

        # RHS -2 fxi gradx fht_xi
        self.minus_two_fxi_gradx_fht_xi = -2. * fxi * self.Grad(fht_xi, xzn0)

        # RHS +2 xiff eht_dd xidot
        self.plus_two_xiff_eht_dd_xidot = +2. * (ddxixidot - (ddxi / dd) * ddxidot)

        # -res
        self.minus_resXiVariance = -(self.minus_dt_eht_dd_sigmai + self.minus_div_eht_dd_fht_ux_sigmai +
                                     self.minus_div_fsigmai + self.minus_two_fxi_gradx_fht_xi +
                                     self.plus_two_xiff_eht_dd_xidot)

        ##########################
        # END Xi VARIANCE EQUATION 		
        ##########################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0

        self.nx = nx
        self.inuc = inuc
        self.element = element
        self.sigmai = sigmai

        self.fht_xi = fht_xi

        self.bconv = bconv
        self.tconv = tconv

        self.ig = ig
        self.fext = fext
        self.t_timec = t_timec
        self.t_ddxi = t_ddxi
        self.ddxidot = ddxidot
        self.nsdim = nsdim
        self.code = code
        self.plabel = plabel

    def plot_XvarianceEquation(self, laxis, bconv, tconv, xbl, xbr, ybuBgr, ybdBgr, ybuEq, ybdEq, ybuBar, ybdBar,
                                                     ilg):
        """Plot rho stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(XvarianceEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        nx = self.nx
        ny = nx
        nz = nx
        xzn0 = self.xzn0
        yzn0 = self.yzn0
        zzn0 = self.zzn0
        nsdim = self.nsdim
        element = self.element
        plabel = self.plabel
        code = self.code

        # load BACKGROUND to plot
        plt1 = self.sigmai

        # load EQUATION
        lhs0 = self.minus_dt_eht_dd_sigmai
        lhs1 = self.minus_div_eht_dd_fht_ux_sigmai

        rhs0 = self.minus_div_fsigmai
        rhs1 = self.minus_two_fxi_gradx_fht_xi
        rhs2 = self.plus_two_xiff_eht_dd_xidot

        res = self.minus_resXiVariance

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

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # calculate integral budgets
        terms = [lhs0,lhs1,rhs0,rhs1,rhs2,res]
        int_terms = self.calcIntegralBudget(terms, xbl_l, xbr_l, nx, xzn0_l, yzn0, zzn0, nsdim, plabel, laxis, self.ig)

        eQterms = [r"$-\partial_t (\overline{\rho} \sigma)$", r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \sigma)$",
                   r"$-\nabla_x f^\sigma$", r"$-2 f_i \partial_x \widetilde{X}$", r"$+2 \overline{\rho X^{''} \dot{X}_{nuc}}$",  r"$+res$"]

        # Plot
        title1 = code
        title2 = "variance " + str(self.element)
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
                       line=dict(color='purple'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs0, name=eQterms[2],
                       line=dict(color='blue'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs1, name=eQterms[3],
                       line=dict(color='green'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs2, name=eQterms[4],
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=res, name=eQterms[5],
                       line=dict(color='black', dash='dash'),hoverinfo='none'),
            row=1, col=1)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl_l,xbr_l], tickangle=-45, row=1, col=1,
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

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl_l,xbr_l], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10,
                         showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)
        fig.update_yaxes(title_text="X (ccp units)", range=[ybdBgr, ybuBgr], tickangle=-45,
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
                          title=r"$\partial_t (\overline{\rho} \sigma) = -\nabla_x (\overline{\rho} \widetilde{u}_x \sigma) "
                                r"-\nabla_x f^\sigma -2 f_i \partial_x \widetilde{X}+2 \overline{\rho X^{''} \dot{X}_{nuc}}  + res$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]),xaxis2=dict(domain=[0.37, 0.64]),xaxis3=dict(domain=[0.74, 1.]))


        return fig
