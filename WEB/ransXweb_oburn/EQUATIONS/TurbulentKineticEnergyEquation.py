from UTILS.Calculus import Calculus
from UTILS.Tools import Tools
from UTILS.Errors import Errors
from EQUATIONS.TurbulentKineticEnergyCalculation import TurbulentKineticEnergyCalculation
import sys

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentKineticEnergyEquation(Calculus, Tools, Errors, object):

    def __init__(self, filename, plabel, code, ig, intc, nsdim, data_prefix):
        super(TurbulentKineticEnergyEquation, self).__init__(ig)

        # instantiate turbulent kinetic energy object
        tkeF = TurbulentKineticEnergyCalculation(filename, ig, intc)

        # load all fields
        tkefields = tkeF.getTKEfield()

        nx = tkefields['nx']
        xzn0 = tkefields['xzn0']
        yzn0 = tkefields['yzn0']
        zzn0 = tkefields['zzn0']

        # LHS -dq/dt
        self.minus_dt_dd_tke = tkefields['minus_dt_dd_tke']

        # LHS -dq/dt
        self.minus_dt_dd_tke = tkefields['minus_dt_dd_tke']

        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = tkefields['minus_div_eht_dd_fht_ux_tke']

        # -div kinetic energy flux
        self.minus_div_fekx = tkefields['minus_div_fekx']

        # -div acoustic flux		
        self.minus_div_fpx = tkefields['minus_div_fpx']

        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = tkefields['plus_ax']

        # +buoyancy work
        self.plus_wb = tkefields['plus_wb']

        # +pressure dilatation
        self.plus_wp = tkefields['plus_wp']

        # -R grad u
        self.minus_r_grad_u = tkefields['minus_r_grad_u']
        # -res		
        self.minus_resTkeEquation = tkefields['minus_resTkeEquation']

        #######################################
        # END TURBULENT KINETIC ENERGY EQUATION 
        #######################################

        # assign more global data to be shared across whole class
        self.data_prefix = data_prefix
        self.ig = ig
        self.dd = tkefields['dd']
        self.tke = tkefields['tke']

        self.nsdim = nsdim
        self.code = code

        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0
        self.nx = nx
        self.plabel = plabel


    def plot_TurbulentKineticEnergyEquation(self, laxis, bconv, tconv, xbl, xbr, ybuBgr, ybdBgr, ybuEq, ybdEq, ybuBar, ybdBar,
                                                     ilg):
        """Plot turbulent kinetic energy equation in the model"""

        if self.ig != 1 and self.ig != 2:
            print("ERROR(TurbulentKineticEnergyEquation.py):" + self.errorGeometry(self.ig))
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
        plt1 = self.tke

        lhs0 = self.minus_dt_dd_tke
        lhs1 = self.minus_div_eht_dd_fht_ux_tke

        rhs0 = self.plus_wb
        rhs1 = self.plus_wp
        rhs2 = self.minus_div_fekx
        rhs3 = self.minus_div_fpx
        rhs4 = self.minus_r_grad_u

        res = self.minus_resTkeEquation


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
        terms = [lhs0,lhs1,rhs0,rhs1, rhs2, rhs3, rhs4, res]
        int_terms = self.calcIntegralBudget(terms, xbl_l, xbr_l, nx, xzn0_l, yzn0, zzn0, nsdim, plabel, laxis, self.ig)

        eQterms = [r"$-\partial_t (\overline{\rho} \widetilde{k})$",
                   r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{k})$",
                   r"$+W_b$",
                   r"$+W_p$",
                   r"$-\nabla_x f_k$",
                   r"$-\nabla_x f_P$",
                   r"$-\widetilde{R}_{xi}\partial_x \widetilde{u_i}$",
                   r"res"]


        # Plot
        title1 = code
        title2 = "turbulent kinetic energy"
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
                       line=dict(color='red'),hoverinfo='none'),
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
                       line=dict(color='magenta'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=rhs4, name=eQterms[6],
                       line=dict(color='blue'),hoverinfo='none'),
            row=1, col=1)

        fig.append_trace(
            go.Scatter(x=xzn0_l, y=res, name=eQterms[7],
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
                          title=r"$\partial_t \overline{\rho} \widetilde{k} + \nabla_x \overline{\rho} \widetilde{u}_x \widetilde{k} = -\nabla_x(f_k + f_P) - \widetilde{R}_{ix} \partial_x \widetilde{u}_i + W_b + W_P + res$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]),xaxis2=dict(domain=[0.37, 0.64]),xaxis3=dict(domain=[0.74, 1.]))


        return fig

