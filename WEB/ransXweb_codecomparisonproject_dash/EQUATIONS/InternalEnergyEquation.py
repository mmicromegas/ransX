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

class InternalEnergyEquation(Calculus, Tools, Errors, object):

    def __init__(self, filename, plabel, code, ig, intc, nsdim, tke_diss, data_prefix):
        super(InternalEnergyEquation, self).__init__(ig)

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

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddei = self.getRAdata(eht, 'ddei')[intc]
        ddeiux = self.getRAdata(eht, 'ddeiux')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc')[intc]
        ddenuc2 = np.zeros(nx)

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddei = self.getRAdata(eht, 'ddei')
        t_fht_ei = t_ddei / t_dd

        # construct equation-specific mean fields		
        fht_ux = ddux / dd
        fht_ei = ddei / dd
        fei = ddeiux - ddux * ddei / dd

        ##########################
        # INTERNAL ENERGY EQUATION 
        ##########################

        # LHS -dq/dt 		
        self.minus_dt_dd_fht_ei = -self.dt(t_dd * t_fht_ei, xzn0, t_timec, intc)

        # LHS -div dd fht_ux fht_ei		
        self.minus_div_dd_fht_ux_fht_ei = -self.Div(dd * fht_ux * fht_ei, xzn0)

        # RHS -div fei
        self.minus_div_fei = -self.Div(fei, xzn0)

        # RHS -div ftt (not included) heat flux
        self.minus_div_ftt = -np.zeros(nx)

        # RHS -P d = - pp Div ux
        self.minus_pp_div_ux = -pp * self.Div(ux, xzn0)

        # RHS -Wp = -eht_ppf_df
        self.minus_eht_ppf_df = -(ppdivu - pp * divu)

        # RHS source + dd enuc
        self.plus_dd_fht_enuc = ddenuc1 + ddenuc2

        # RHS dissipated turbulent kinetic energy
        self.plus_disstke = +tke_diss

        # -res
        self.minus_resEiEquation = -(self.minus_dt_dd_fht_ei + self.minus_div_dd_fht_ux_fht_ei +
                                     self.minus_div_fei + self.minus_div_ftt + self.minus_pp_div_ux + self.minus_eht_ppf_df +
                                     self.plus_dd_fht_enuc + self.plus_disstke)

        ##############################
        # END INTERNAL ENERGY EQUATION 
        ##############################

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.ig = ig
        self.dd = dd
        self.ddei = ddei

        self.nsdim = nsdim
        self.code = code

        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0
        self.nx = nx
        self.plabel = plabel

    def plot_InternalEnergyEquation(self, laxis, bconv, tconv, xbl, xbr, ybuBgr, ybdBgr, ybuEq, ybdEq, ybuBar, ybdBar,
                                                     ilg):
        """Plot turbulent kinetic energy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(InternalEnergyEquation.py):" + self.errorGeometry(self.ig))
            sys.exit()

        # load x GRID
        nx = self.nx
        xzn0 = self.xzn0
        yzn0 = self.yzn0
        zzn0 = self.zzn0
        nsdim = self.nsdim
        plabel = self.plabel
        code = self.code

        # load BACKGROUND to plot
        plt1 = self.ddei

        lhs0 = self.minus_dt_dd_fht_ei
        lhs1 = self.minus_div_dd_fht_ux_fht_ei

        rhs0 = self.minus_div_fei
        rhs1 = self.minus_div_ftt
        rhs2 = self.minus_pp_div_ux
        rhs3 = self.minus_eht_ppf_df
        rhs4 = self.plus_dd_fht_enuc
        rhs5 = self.plus_disstke

        res = self.minus_resEiEquation


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

        # calculate integral budgets
        terms = [lhs0,lhs1,rhs0,rhs1, rhs2, rhs3, rhs4, rhs5, res]
        int_terms = self.calcIntegralBudget(terms, xbl_l, xbr_l, nx, xzn0_l, yzn0, zzn0, nsdim, plabel, laxis, self.ig)

        eQterms = [r"$-\partial_t (\overline{\rho} \widetilde{\epsilon}_I )$",
                   r"$-\nabla_x (\overline{\rho}\widetilde{u}_x \widetilde{\epsilon}_I)$",
                   r"$-\nabla_x f_I $", r"$-\nabla_x f_T$ (not incl.)", r"$-\bar{P} \bar{d}$",r"$-W_P$",
                   r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}$",r"$+\varepsilon_k$",
                   r"res"]

        # Plot
        title1 = code
        title2 = "internal energy"
        title3 = "integral budget"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=(title1, title2, title3))

        # 1st subplot
        fig.append_trace(
            go.Scatter(x=xzn0_l, y=lhs0, name=eQterms[0],
                       line=dict(color='#FF6EB4'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=lhs1, name=eQterms[1],
                       line=dict(color='black'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs0, name=eQterms[2],
                       line=dict(color='#FF8C00'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs1, name=eQterms[3],
                       line=dict(color='cyan'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs2, name=eQterms[4],
                       line=dict(color='#802A2A'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs3, name=eQterms[5],
                       line=dict(color='green'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs4, name=eQterms[6],
                       line=dict(color='blue'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs5, name=eQterms[7],
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=res, name=eQterms[8],
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

        #print(int_terms)

        # 3rd subplot
        fig.append_trace(go.Bar(
            x=eQterms,y=int_terms,
            orientation='v', showlegend=False, hoverinfo='none'), row=1, col=3)
        fig.update_yaxes(title_text="ccp units", range=[ybdBar, ybuBar], exponentformat='e', tickangle=-45,
                         tickwidth=2, ticks='outside',showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=3)
        fig.update_xaxes(title_text=r'', showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=3)


        # show
        fig.update_layout(height=550, width=1400, font=dict(size=14), xaxis_tickangle=-45, yaxis_tickangle=-45)
        fig.update_layout(legend=dict(yanchor="top", y=1.02, xanchor="left", x=0.16, bgcolor='rgba(0,0,0,0)',font=dict(size=18)),
                          title=r"$\partial_t (\overline{\rho} \widetilde{\epsilon}_I)+\nabla_x (\overline{\rho}\widetilde{u}_x \widetilde{\epsilon}_I) = "
                                r"-\nabla_x f_I -\nabla_x f_T -\bar{P} \bar{d} -W_P +\overline{\rho}\widetilde{\epsilon}_{nuc} +\varepsilon_k + res$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]),xaxis2=dict(domain=[0.37, 0.64]),xaxis3=dict(domain=[0.74, 1.]))


        return fig