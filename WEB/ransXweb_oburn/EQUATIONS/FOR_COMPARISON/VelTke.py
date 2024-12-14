import numpy as np
from UTILS.Calculus import Calculus
from UTILS.Tools import Tools
from UTILS.Errors import Errors
import sys

from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class VelTke(Calculus, Tools, Errors, object):

    def __init__(self, filename, plabel, codes, ig, intc, data_prefix):
        global fht_uyrms, fht_uxrms, fht_uzrms, xzn0
        super(VelTke, self).__init__(ig)

        # load data to list of structured arrays
        eht = []

        for ffile in filename:
            eht.append(np.load(ffile, allow_pickle=True))

        # declare data lists
        xzn0, nx = [], []
        fht_uxrms, fht_uyrms, fht_uzrms = [], [], []

        for i in range(len(filename)):
            # load grid
            nx.append(self.getRAdata(eht[i], 'nx'))
            xzn0.append(self.getRAdata(eht[i], 'xzn0'))

            # pick specific Reynolds-averaged mean fields according to:
            # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

            dd = self.getRAdata(eht[i], 'dd')[intc]
            ddux = self.getRAdata(eht[i], 'ddux')[intc]
            dduxux = self.getRAdata(eht[i], 'dduxux')[intc]
            dduy = self.getRAdata(eht[i], 'dduy')[intc]
            dduyuy = self.getRAdata(eht[i], 'dduyuy')[intc]
            dduz = self.getRAdata(eht[i], 'dduz')[intc]
            dduzuz = self.getRAdata(eht[i], 'dduzuz')[intc]

            fht_uxrms.append(((dduxux - ddux * ddux / dd) / dd) ** 0.5)
            fht_uyrms.append(((dduyuy - dduy * dduy / dd) / dd) ** 0.5)
            fht_uzrms.append(((dduzuz - dduz * dduz / dd) / dd) ** 0.5)

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ig = ig
        self.fht_uxrms = fht_uxrms
        self.fht_uyrms = fht_uyrms
        self.fht_uzrms = fht_uzrms

        self.codes = codes

        self.xzn0 = xzn0
        self.nx = nx
        self.plabel = plabel

    def plot_velTke(self, laxis, xbl, xbr, ybuBgr1, ybdBgr1, ybuBgr2, ybdBgr2, ybuBgr3, ybdBgr3, ybuBgr4, ybdBgr4):

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(plot_velTke):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        nx = self.nx
        xzn0 = self.xzn0
        codes = self.codes

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(VelTke.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # Plot
        title1 = r"$u_x^{rms}$"
        title2 = r"$u_y^{rms}$"
        title3 = r"$u_z^{rms}$"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=(title1, title2, title3))

        # 1st subplot
        #for i in range(len(xzn0)): # looping create different colors for subplots 2 and 3
        #    fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uxrms[i], name=self.codes[i], hoverinfo='none'),row=1, col=1)

        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uxrms[0], name=self.codes[0], line=dict(color='red'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uxrms[1], name=self.codes[1], line=dict(color='green'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uxrms[2], name=self.codes[2], line=dict(color='blue'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uxrms[3], name=self.codes[3], line=dict(color='magenta'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uxrms[4], name=self.codes[4], line=dict(color='brown'), hoverinfo='none'), row=1, col=1)

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl, xbr], tickangle=-45, row=1,
                         col=1,
                         tickwidth=2, ticklen=10, nticks=10, showgrid=True, showline=True, linewidth=1,
                         linecolor='black', mirror=True, ticks='outside')
        fig.update_yaxes(title_text="ccp units", range=[ybdBgr1, ybuBgr1], tickangle=-45, exponentformat='e',
                         row=1, col=1, tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside')

        # 2nd subplot
        #for i in range(len(xzn0)):
        #    fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uyrms[i], showlegend=False,hoverinfo='none'),row=1, col=2)

        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uyrms[0], line=dict(color='red'), showlegend=False,hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uyrms[1], line=dict(color='green'), showlegend=False, hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uyrms[2], line=dict(color='blue'), showlegend=False,hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uyrms[3], line=dict(color='magenta'), showlegend=False,hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uyrms[4], line=dict(color='brown'), showlegend=False,hoverinfo='none'), row=1, col=2)

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl, xbr], tickangle=-45, row=1,
                         col=2,
                         tickwidth=2, ticklen=10, nticks=10, showgrid=True, showline=True, linewidth=1,
                         linecolor='black', mirror=True, ticks='outside')
        fig.update_yaxes(title_text="ccp units", range=[ybdBgr2, ybuBgr2], tickangle=-45, exponentformat='e',
                         row=1, col=2, tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside')

        # 3rd subplot
        #for i in range(len(xzn0)):
        #    fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uzrms[i],showlegend=False, hoverinfo='none'),row=1, col=3)

        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uzrms[0], line=dict(color='red'), showlegend=False,hoverinfo='none'), row=1, col=3)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uzrms[1], line=dict(color='green'), showlegend=False,hoverinfo='none'), row=1, col=3)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uzrms[2], line=dict(color='blue'), showlegend=False,hoverinfo='none'), row=1, col=3)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uzrms[3], line=dict(color='magenta'), showlegend=False,hoverinfo='none'), row=1, col=3)
        fig.append_trace(go.Scatter(x=xzn0[0], y=self.fht_uzrms[4], line=dict(color='brown'), showlegend=False,hoverinfo='none'), row=1, col=3)

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl, xbr], tickangle=-45, row=1,
                         col=3,
                         tickwidth=2, ticklen=10, nticks=10, showgrid=True, showline=True, linewidth=1,
                         linecolor='black', mirror=True, ticks='outside')
        fig.update_yaxes(title_text="ccp units", range=[ybdBgr3, ybuBgr3], tickangle=-45, exponentformat='e',
                         row=1, col=3, tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside')

        # show
        fig.update_layout(height=550, width=1400, font=dict(size=12), xaxis_tickangle=-45, yaxis_tickangle=-45)
        fig.update_layout(
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.11, bgcolor='rgba(0,0,0,0)', font=dict(size=18)),
            title=r"$u_i^{rms} = [\widetilde{u''_i u''_i}]^{1/2} = [(\overline{\rho u_i u_i} - \overline{\rho u_i} \ \overline{\rho u_i}/ \overline{\rho})/\overline{\rho}]^{1/2}$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]), xaxis2=dict(domain=[0.37, 0.64]),
                          xaxis3=dict(domain=[0.74, 1.]))

        return fig
