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

class CompositionFlux(Calculus, Tools, Errors, object):

    def __init__(self, filename, plabel, codes, ig, intc, data_prefix):
        global fy, fx, fz, xzn0
        super(CompositionFlux, self).__init__(ig)

        # load data to list of structured arrays
        eht = []

        for ffile in filename:
            eht.append(np.load(ffile, allow_pickle=True))

        # declare data lists
        xzn0, nx = [], []
        fx0001, fx0002 = [], []

        for i in range(len(filename)):
            # load grid
            nx.append(self.getRAdata(eht[i], 'nx'))
            xzn0.append(self.getRAdata(eht[i], 'xzn0'))

            # pick specific Reynolds-averaged mean fields according to:
            # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

            dd = self.getRAdata(eht[i], 'dd')[intc]
            ddux = self.getRAdata(eht[i], 'ddux')[intc]

            ddx0001 = self.getRAdata(eht[i], 'ddx0001')[intc]
            ddx0001ux = self.getRAdata(eht[i], 'ddx0001ux')[intc]

            ddx0002 = self.getRAdata(eht[i], 'ddx0002')[intc]
            ddx0002ux = self.getRAdata(eht[i], 'ddx0002ux')[intc]

            fx0001.append(ddx0001ux - ddx0001 * ddux / dd)
            fx0002.append(ddx0002ux - ddx0002 * ddux / dd)


        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ig = ig
        self.fx0001 = fx0001
        self.fx0002 = fx0002

        self.codes = codes

        self.xzn0 = xzn0
        self.nx = nx
        self.plabel = plabel

    def plot_composition_flux(self, laxis, xbl, xbr, ybuBgr1, ybdBgr1, ybuBgr2, ybdBgr2):

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(plot_CompositionFlux):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        nx = self.nx
        xzn0 = self.xzn0
        fx0001 = self.fx0001
        fx0002 = self.fx0002

        codes = self.codes

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(CompositionFlux.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # Plot
        title1 = r"$f^x \ fluid1$"
        title2 = r"$f^x \ fluid2$"
        #title3 = r"$f^z$"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=(title1, title2))

        # 1st subplot
        #for i in range(len(xzn0)): # looping create different colors for subplots 2 and 3
        #    fig.append_trace(go.Scatter(x=xzn0[0], y=fx[i], name=codes[i], hoverinfo='none'),row=1, col=1)

        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0001[0], name=codes[0], line=dict(color='red'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0001[1], name=codes[1], line=dict(color='green'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0001[2], name=codes[2], line=dict(color='blue'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0001[3], name=codes[3], line=dict(color='magenta'), hoverinfo='none'), row=1, col=1)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0001[4], name=codes[4], line=dict(color='brown'), hoverinfo='none'), row=1, col=1)

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
        #    fig.append_trace(go.Scatter(x=xzn0[0], y=fy[i], showlegend=False,hoverinfo='none'),row=1, col=2)

        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0002[0], line=dict(color='red'), showlegend=False,hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0002[1], line=dict(color='green'), showlegend=False, hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0002[2], line=dict(color='blue'), showlegend=False,hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0002[3], line=dict(color='magenta'), showlegend=False,hoverinfo='none'), row=1, col=2)
        fig.append_trace(go.Scatter(x=xzn0[0], y=fx0002[4], line=dict(color='brown'), showlegend=False,hoverinfo='none'), row=1, col=2)

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
        #    fig.append_trace(go.Scatter(x=xzn0[0], y=fz[i],showlegend=False, hoverinfo='none'),row=1, col=3)

        # show
        fig.update_layout(height=550, width=1400, font=dict(size=12), xaxis_tickangle=-45, yaxis_tickangle=-45)
        fig.update_layout(
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.7, bgcolor='rgba(0,0,0,0)', font=dict(size=18)),
            title=r"$f^x = \overline{\rho} \widetilde{X'' u''_x} = \overline{\rho X u_x} - \overline{\rho X} \ \overline{\rho u_x}/\overline{\rho}$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]), xaxis2=dict(domain=[0.37, 0.64]),
                          xaxis3=dict(domain=[0.74, 1.]))

        return fig
