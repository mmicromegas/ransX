import numpy as np
from UTILS.Calculus import Calculus
from UTILS.SetAxisLimit import SetAxisLimit
from UTILS.Tools import Tools
from UTILS.Errors import Errors
import sys

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class SourceVel(Calculus, Tools, Errors, object):

    def __init__(self, filename, plabel, code, ig, fext, intc, nsdim, data_prefix):
        super(SourceVel, self).__init__(ig)

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

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        ux = self.getRAdata(eht,'ux')[intc]
        dd = self.getRAdata(eht,'dd')[intc]
        enuc = self.getRAdata(eht,'enuc')[intc]
        ddux = self.getRAdata(eht,'ddux')[intc]
        dduxux = self.getRAdata(eht,'dduxux')[intc]

        urms = ((dduxux - ddux * ddux / dd) / dd) ** 0.5

        eht_ux = ux
        fht_ux = ddux / dd
        f_rho = fht_ux - eht_ux # turbulent mass flux

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.ig = ig
        self.eht_ux = eht_ux
        self.fht_ux = fht_ux
        self.urms = urms
        self.enuc = enuc

        self.nsdim = nsdim
        self.code = code

        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0
        self.nx = nx
        self.plabel = plabel

    def plot_SourceVel(self, laxis, bconv, tconv, xbl, xbr, ybuBgr1, ybdBgr1, ybuBgr2, ybdBgr2, ybuBgr3, ybdBgr3,
                                                     ilg):

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(SourceVel.py):" + self.errorGeometry(self.ig))
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


        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(SourceVel.py):" + self.errorGeometry(self.ig))
            sys.exit()

        eQterms = [r"$+\overline{u}_x$",r"$+\widetilde{u}_x$",r"$-\overline{\rho' u'_x}/\overline{\rho}$"]

        # Plot
        title1 = code
        title2 = "source"
        title3 = r"$u_{rms}$"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=(title1, title2, title3))

        # 1st subplot
        fig.append_trace(
            go.Scatter(x=xzn0_l, y=self.eht_ux, name=eQterms[0],
                       line=dict(color='brown'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=self.fht_ux, name=eQterms[1],
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=self.fht_ux-self.eht_ux, name=eQterms[2],
                       line=dict(color='magenta'),hoverinfo='none'),
            row=1, col=1)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl_l, xbr_l], tickangle=-45, row=1, col=1,
                         tickwidth = 2,ticklen = 10, nticks = 10, showgrid = True, showline = True, linewidth = 1,
                         linecolor = 'black', mirror = True, ticks = 'outside')
        fig.update_yaxes(title_text="ccp units", range=[ybdBgr1, ybuBgr1], tickangle=-45, exponentformat='e',
                         row=1, col=1, tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside')

        # 2nd subplot
        fig.append_trace(
            go.Scatter(x=xzn0_l, y=self.enuc, showlegend=False,
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=2)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl_l, xbr_l], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10,
                         showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)
        fig.update_yaxes(title_text="ccp units", range=[ybdBgr2, ybuBgr2], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True, exponentformat='e',
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)

        # 3rd subplot
        fig.append_trace(
            go.Scatter(x=xzn0_l, y=self.urms,showlegend=False,
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=3)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="x (ccp units)", exponentformat='e', range=[xbl_l, xbr_l], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10,
                         showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=3)
        fig.update_yaxes(title_text="ccp units", range=[ybdBgr3, ybuBgr3], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True, exponentformat='e',
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=3)

        # show
        fig.update_layout(height=550, width=1400, font=dict(size=14), xaxis_tickangle=-45, yaxis_tickangle=-45)
        fig.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor='rgba(0,0,0,0)',font=dict(size=18)),
                          title=r"$-\overline{\rho' u'_x}/\overline{\rho} = \widetilde{u}_x - \overline{u}_x$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]),xaxis2=dict(domain=[0.37, 0.64]),xaxis3=dict(domain=[0.74, 1.]))

        return fig
