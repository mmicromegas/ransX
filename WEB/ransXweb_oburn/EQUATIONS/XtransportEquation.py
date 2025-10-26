# class for RANS XtransportEquation # ..

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

class XtransportEquation(Calculus, Tools, Errors, object):

    def __init__(self, filename, filename_reaclib, plabel, code, ig, fext, inuc, element, bconv, tconv, tc, intc, nsdim, data_prefix, tnuc, network):
        super(XtransportEquation, self).__init__(ig)

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

        nnuc = self.getRAdata(eht, 'nnuc')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = self.getRAdata(eht, 'dd')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]
        ddxi = self.getRAdata(eht, 'ddx' + inuc)[intc]
        ddxiux = self.getRAdata(eht, 'ddx' + inuc + 'ux')[intc]
        ddxidot = self.getRAdata(eht, 'ddx' + inuc + 'dot')[intc]


        #######################
        # Xi TRANSPORT EQUATION
        #######################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')
        t_ddxi = self.getRAdata(eht, 'ddx' + inuc)
        t_fht_xi = t_ddxi / t_dd

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_xi = ddxi / dd
        fxi = ddxiux - ddxi * ddux / dd

        # LHS -dq/dt
        self.minus_dt_dd_fht_xi = -self.dt(t_dd * t_fht_xi, xzn0, t_timec, intc)

        # LHS -div(ddXiux)
        self.minus_div_eht_dd_fht_ux_fht_xi = -self.Div(dd * fht_ux * fht_xi, xzn0)
        self.minus_dt_fht_xi = -self.dt(t_fht_xi, xzn0, t_timec, intc)

        # RHS -div fxi
        self.minus_div_fxi = -self.Div(fxi, xzn0)

        # RHS +ddXidot
        self.plus_ddxidot = +ddxidot

        # -res
        self.minus_resXiTransport = -(self.minus_dt_dd_fht_xi + self.minus_div_eht_dd_fht_ux_fht_xi + self.minus_div_fxi + self.plus_ddxidot)

        ###########################
        # END Xi TRANSPORT EQUATION
        ###########################

        # load REACLIB data

        rcoeff_tmp = []
        rlabel = []

        # read line-by-line
        with open(filename_reaclib) as handle:
            for lineno, line in enumerate(handle):
                if (lineno != 0) and (lineno % 3 != 0):
                    rcoeff_tmp.append(line.rstrip())
                if lineno % 3 == 0:
                    rlabel.append(line[0:52].replace(" ", ""))

                    # restructure and join every second and third coefficient line
        rcoeff = [''.join(x) for x in zip(rcoeff_tmp[0::2], rcoeff_tmp[1::2])]

        # split reaction coefficient string to individual coefficients
        n = 13
        rcoeffdict = {}
        for i in range(len(rcoeff)):
            rcoeffone = rcoeff[i]
            # parse out individual reaction rate coefficients
            out = [(rcoeffone[j:j + n]) for j in range(0, len(rcoeffone), n)]
            # convert to float
            outfloat = []
            for k in out:
                outfloat.append(float(k))
            # store in dictionary
            rc = {rlabel[i]: outfloat}
            rcoeffdict.update(rc)


        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = xzn0
        self.yzn0 = yzn0
        self.zzn0 = zzn0

        self.nx = nx
        self.inuc = inuc
        self.element = element
        self.ddxi = ddxi

        self.fht_xi = fht_xi

        self.bconv = bconv
        self.tconv = tconv

        self.ig = ig
        self.fext = fext
        self.t_timec = t_timec
        self.t_fht_xi = t_fht_xi
        self.t_ddxi = t_ddxi
        self.ddxidot = ddxidot
        self.nnuc = nnuc
        self.nsdim = nsdim
        self.code = code
        self.plabel = plabel

        # tau_trans = np.abs(fht_xi/self.Div(fxi/dd,xzn0))
        # tau_nuc   = np.abs(fht_xi/(ddxidot/dd))

        tau_trans = np.abs(dd*fht_xi/self.Div(fxi,xzn0))
        tau_nuc   = np.abs(dd*fht_xi/(ddxidot))

        #print(ddxidot)
        #sys.exit()
        tau_ddxi =  np.abs(dd*fht_xi/self.minus_dt_dd_fht_xi)
        tau_xi =  np.abs(fht_xi/self.minus_dt_fht_xi)

        #tau_trans = (dd*fht_xi/self.Div(fxi,xzn0))
        #tau_nuc   = (dd*fht_xi/(ddxidot))
        #tau_ddxi =  (dd*fht_xi/self.minus_dt_dd_fht_xi)
        #tau_xi =  (fht_xi/self.minus_dt_fht_xi)


        #tau_trans = np.abs(fht_xi/self.Div(fxi/dd,xzn0))
        #tau_nuc   = np.abs(fht_xi/(ddxidot/dd))
        #tau_ddxi =  np.abs(dd*fht_xi/self.minus_dt_dd_fht_xi)
        #tau_xi =  np.abs(fht_xi/self.minus_dt_fht_xi)

        # tau_trans = (fht_xi / self.Div(fxi / dd, xzn0))
        # tau_nuc = (fht_xi / (ddxidot / dd))

        # tau_trans = ddxi/self.Div(fxi,xzn0)
        # tau_nuc   = ddxi/(ddxidot)

        # Damkohler number
        self.xda = tau_trans / tau_nuc

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix
        self.xzn0 = self.getRAdata(eht, 'xzn0')
        self.element = element
        self.inuc = inuc
        self.bconv = bconv
        self.tconv = tconv
        self.tc = tc
        self.tt = tt
        self.dd = dd
        self.rlabel = rlabel
        self.rcoeffdict = rcoeffdict

        # self.tau_trans = np.abs(tau_trans)
        # self.tau_nuc = np.abs(tau_nuc)

        self.tau_trans = tau_trans
        self.tau_nuc = tau_nuc
        self.tau_ddxi = tau_ddxi
        self.tau_xi = tau_xi

        self.fht_xi = fht_xi
        self.network = network
        self.eht = eht
        self.intc = intc
        self.fext = fext
        self.tnuc = tnuc
        self.nx = nx


    def plot_XtransportEquation(self, laxis, bconv, tconv, xbl, xbr, ybuBgr, ybdBgr, ybuEq, ybdEq, ybuBar, ybdBar,
                                                     ilg):
        """Plot rho stratification in the model"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(XtransportEquation.py):" + self.errorGeometry(self.ig))
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
        plt1 = self.fht_xi

        # load EQUATION
        lhs0 = self.minus_dt_dd_fht_xi
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_xi

        rhs0 = self.minus_div_fxi
        rhs1 = self.plus_ddxidot

        res = self.minus_resXiTransport

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
        terms = [lhs0,lhs1,rhs0,rhs1,res]
        int_terms = self.calcIntegralBudget(terms, xbl_l, xbr_l, nx, xzn0_l, yzn0, zzn0, nsdim, plabel, laxis, self.ig)

        eQterms = [r"$-\partial_t (\overline{\rho} \widetilde{X})$",r"$-\nabla_x (\overline{\rho} \widetilde{X} \widetilde{u}_x )$",
                 r"$-\nabla_x f$", r"$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$", r"$+res$"]

        # Plot
        title1 = code
        title2 = "mass fraction " +  str(self.element)
        title3 = "integral budget"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=(title1, title2, title3))

        # 1st subplot
        fig.add_trace(
            go.Scatter(x=xzn0_l, y=lhs0, name=eQterms[0],
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=xzn0_l, y=lhs1, name=eQterms[1],
                       line=dict(color='cyan'),hoverinfo='none'),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=xzn0_l, y=rhs0, name=eQterms[2],
                       line=dict(color='blue'),hoverinfo='none'),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=xzn0_l, y=rhs1, name=eQterms[3],
                       line=dict(color='green'),hoverinfo='none'),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=xzn0_l, y=res, name=eQterms[4],
                       line=dict(color='black', dash='dash'),hoverinfo='none'),
            row=1, col=1)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="r (cm)", exponentformat='e', range=[xbl_l,xbr_l], tickangle=-45, row=1, col=1,
                         tickwidth = 2,ticklen = 10, nticks = 10, showgrid = True, showline = True, linewidth = 1,
                         linecolor = 'black', mirror = True, ticks = 'outside')
        fig.update_yaxes(title_text=r"$\mbox{g cm}^{-3} \mbox{s}^{-1}$", range=[ybdEq, ybuEq], tickangle=-45, exponentformat='e',
                         row=1, col=1, tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside')

        # 2nd subplot
        fig.add_trace(
            go.Scatter(x=xzn0_l, y=plt1,
                       line=dict(color='blue'), hoverinfo='none',showlegend=False), row=1, col=2)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="r (cm)", exponentformat='e', range=[xbl_l,xbr_l], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10,
                         showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)
        fig.update_yaxes(title_text=r"$\widetilde{X}$", range=[ybdBgr, ybuBgr], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True, exponentformat='e',
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)

        # 3rd subplot
        fig.add_trace(go.Bar(
            x=eQterms,y=int_terms,
            orientation='v', showlegend=False, hoverinfo='none'), row=1, col=3)
        fig.update_yaxes(title_text=r"$\mbox{g s}^{-1}$", range=[ybdBar, ybuBar], exponentformat='e', tickangle=-45,
                         tickwidth=2, ticks='outside',showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=3)
        fig.update_xaxes(title_text=r'', showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=3)

        # show
        fig.update_layout(height=550, width=1400, font=dict(size=14), xaxis_tickangle=-45, yaxis_tickangle=-45)
        fig.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.12, bgcolor='rgba(0,0,0,0)',font=dict(size=18)),
                          title=r"$\partial_t (\overline{\rho} \widetilde{X}) = -\nabla_x (\overline{\rho} \widetilde{X} \widetilde{u}_x) "
                                r"-\nabla_x f "
                                r"+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc} + res$")
        fig.update_layout(xaxis=dict(domain=[0, 0.27]),xaxis2=dict(domain=[0.37, 0.64]),xaxis3=dict(domain=[0.74, 1.]))


        return fig


    def plot_Xtimescales(self, laxis, bconv, tconv, xbl, xbr, ybuEq, ybdEq,ilg):



        if self.ig != 1 and self.ig != 2:
            print("ERROR(Xtimescales.py):" + self.errorGeometry(self.ig))
            sys.exit()

        xzn0_l = self.xzn0
        xbl_l = xbl
        xbr_l = xbr
        bconv = self.bconv
        tconv = self.tconv

        # convert nuc ID to string
        xnucid = str(self.inuc)

        element = self.element
        rlabel = self.rlabel
        rcoeffdict = self.rcoeffdict
        eht = self.eht
        intc = self.intc
        dd = self.dd
        network = self.network

        # yi = self.fht_xi
        # yj =
        # yk =

        # load x GRID
        grd1 = self.xzn0

        # get data
        plt0 = self.tau_trans
        plt1 = self.tau_nuc
        plt2 = self.tau_ddxi
        plt3 = self.tau_xi



        eQterms = [r"$-\partial_t (\overline{\rho} \widetilde{X})$",r"$-\nabla_x (\overline{\rho} \widetilde{X} \widetilde{u}_x )$",
                 r"$-\nabla_x f$", r"$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$", r"$+res$"]

        # Plot
        title1 = 'code'
        title2 = "mass fraction " +  str(self.element)
        title3 = "integral budget"

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=(title1, title2, title3))

        # 1st subplot
        fig.add_trace(
            go.Scatter(x=xzn0_l, y=plt0, name=eQterms[0],
                       line=dict(color='red'),hoverinfo='none'),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=xzn0_l, y=plt1, name=eQterms[1],
                       line=dict(color='cyan'),hoverinfo='none'),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=xzn0_l, y=plt2, name=eQterms[2],
                       line=dict(color='blue'),hoverinfo='none'),
            row=1, col=1)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="r (cm)", exponentformat='e', range=[xbl_l,xbr_l], tickangle=-45, row=1, col=1,
                         tickwidth = 2,ticklen = 10, nticks = 10, showgrid = True, showline = True, linewidth = 1,
                         linecolor = 'black', mirror = True, ticks = 'outside')
        fig.update_yaxes(title_text=r"$\mbox{g cm}^{-3} \mbox{s}^{-1}$", range=[ybdEq, ybuEq], tickangle=-45, exponentformat='e',
                         row=1, col=1, tickwidth=2,
                         ticklen=10, nticks=10, showgrid=True,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside')

        # 2nd subplot
        fig.add_trace(
            go.Scatter(x=xzn0_l, y=plt1,
                       line=dict(color='blue'), hoverinfo='none',showlegend=False), row=1, col=2)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="r (cm)", exponentformat='e', range=[xbl_l,xbr_l], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10,
                         showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)
        #fig.update_yaxes(title_text=r"$\widetilde{X}$", range=[ybdBgr, ybuBgr], tickangle=-45,
        #                 tickwidth=2,
        #                 ticklen=10, nticks=10, showgrid=True, exponentformat='e',
        #                 showline=True, linewidth=1, linecolor='black', mirror=True,
        #                 ticks='outside', row=1, col=2)

        # 3rd subplot
        fig.add_trace(
            go.Scatter(x=xzn0_l, y=plt1,
                       line=dict(color='blue'), hoverinfo='none',showlegend=False), row=1, col=2)

        fig.add_vline(bconv,line_width=1, line_dash="dot", line_color="black")
        fig.add_vline(tconv,line_width=1, line_dash="dot", line_color="black")

        fig.update_xaxes(title_text="r (cm)", exponentformat='e', range=[xbl_l,xbr_l], tickangle=-45,
                         tickwidth=2,
                         ticklen=10, nticks=10,
                         showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                         ticks='outside', row=1, col=2)
        #fig.update_yaxes(title_text=r"$\widetilde{X}$", range=[ybdBgr, ybuBgr], tickangle=-45,
        #                 tickwidth=2,
        #                 ticklen=10, nticks=10, showgrid=True, exponentformat='e',
        #                 showline=True, linewidth=1, linecolor='black', mirror=True,
        #                 ticks='outside', row=1, col=2)




        return fig



    def GET1NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l):

        temp09 = self.tt * 1.e-9
        rate = np.exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (rate)

        return timescale

    def GET2NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l, yi, yj, yk):

        temp09 = self.tt * 1.e-9
        rate = np.exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (self.dd * yj * yk * rate / yi)

        return timescale

    def GET3NUCtimescale(self, c1l, c2l, c3l, c4l, c5l, c6l, c7l, yi1, yi2):

        temp09 = self.tt * 1.e-9
        rate = np.exp(c1l + c2l * (temp09 ** (-1.)) + c3l * (temp09 ** (-1. / 3.)) + c4l * (
                temp09 ** (1. / 3.)) + c5l * temp09 + c6l * (temp09 ** (5. / 3.)) + c7l * np.log(temp09))
        timescale = 1. / (self.dd * self.dd * yi1 * yi2 * rate * rate)
        # ipp = 200
        # print(timescale[ipp],self.dd[ipp],yi1[ipp],yi2[ipp],rate[ipp],c1l,c2l,c3l,c4l,c5l,c6l,c7l)

        return timescale

    def GETRATEcoeff(self, reaction):

        cl = np.zeros(7)

        if (reaction == 'c12_plus_c12_to_p_na23_r'):
            cl[0] = +0.585029E+02
            cl[1] = +0.295080E-01
            cl[2] = -0.867002E+02
            cl[3] = +0.399457E+01
            cl[4] = -0.592835E+00
            cl[5] = -0.277242E-01
            cl[6] = -0.289561E+01

        if (reaction == 'c12_plus_c12_to_he4_ne20_r'):
            cl[0] = +0.804485E+02
            cl[1] = -0.120189E+00
            cl[2] = -0.723312E+02
            cl[3] = -0.352444E+02
            cl[4] = +0.298646E+01
            cl[5] = -0.309013E+00
            cl[6] = +0.115815E+02

        if (reaction == 'he4_plus_c12_to_o16_r'):
            cl[0] = +0.142191E+03
            cl[1] = -0.891608E+02
            cl[2] = +0.220435E+04
            cl[3] = -0.238031E+04
            cl[4] = +0.108931E+03
            cl[5] = -0.531472E+01
            cl[6] = +0.136118E+04

        if (reaction == 'he4_plus_c12_to_o16_nr'):
            cl[0] = +0.184977E+02
            cl[1] = +0.482093E-02
            cl[2] = -0.332522E+02
            cl[3] = +0.333517E+01
            cl[4] = -0.701714E+00
            cl[5] = +0.781972E-01
            cl[6] = -0.280751E+01

        if (reaction == 'o16_plus_o16_to_p_p31_r'):
            cl[0] = +0.852628E+02
            cl[1] = +0.223453E+00
            cl[2] = -0.145844E+03
            cl[3] = +0.872612E+01
            cl[4] = -0.554035E+00
            cl[5] = -0.137562E+00
            cl[6] = -0.688807E+01

        if (reaction == 'o16_plus_o16_to_he4_si28_r'):
            cl[0] = +0.972435E+02
            cl[1] = -0.268514E+00
            cl[2] = -0.119324E+03
            cl[3] = -0.322497E+02
            cl[4] = +0.146214E+01
            cl[5] = -0.200893E+00
            cl[6] = +0.132148E+02

        if (reaction == 'ne20_to_he4_o16_nv'):
            cl[0] = +0.637915E+02
            cl[1] = -0.549729E+02
            cl[2] = -0.343457E+02
            cl[3] = -0.251939E+02
            cl[4] = +0.479855E+01
            cl[5] = -0.146444E+01
            cl[6] = +0.784333E+01

        if (reaction == 'ne20_to_he4_o16_rv'):
            cl[0] = +0.109310E+03
            cl[1] = -0.727584E+02
            cl[2] = +0.293664E+03
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.201193E+03

        if (reaction == 'si28_to_he4_mg24_nv1'):
            cl[0] = +0.522024E+03
            cl[1] = -0.122258E+03
            cl[2] = +0.434667E+03
            cl[3] = -0.994288E+03
            cl[4] = +0.656308E+02
            cl[5] = -0.412503E+01
            cl[6] = +0.426946E+03

        if (reaction == 'si28_to_he4_mg24_nv2'):
            cl[0] = +0.157580E+02
            cl[1] = -0.129560E+03
            cl[2] = -0.516428E+02
            cl[3] = +0.684625E+02
            cl[4] = -0.386512E+01
            cl[5] = +0.208028E+00
            cl[6] = -0.320727E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv1'):
            cl[0] = -0.906347E+01
            cl[1] = -0.241182E+02
            cl[2] = +0.373526E+01
            cl[3] = -0.664843E+01
            cl[4] = +0.254122E+00
            cl[5] = -0.588282E-02
            cl[6] = +0.191121E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv2'):
            cl[0] = +0.552169E+01
            cl[1] = -0.265651E+02
            cl[2] = +0.456462E-08
            cl[3] = -0.105997E-07
            cl[4] = +0.863175E-09
            cl[5] = -0.640626E-10
            cl[6] = -0.150000E+01

        if (reaction == 'he4_plus_si28_to_p_p31_rv3'):
            cl[0] = -0.126553E+01
            cl[1] = -0.287435E+02
            cl[2] = -0.309775E+02
            cl[3] = +0.458298E+02
            cl[4] = -0.272557E+01
            cl[5] = +0.163910E+00
            cl[6] = -0.239582E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv4'):
            cl[0] = +0.296908E+02
            cl[1] = -0.330803E+02
            cl[2] = +0.553217E+02
            cl[3] = -0.737793E+02
            cl[4] = +0.325554E+01
            cl[5] = -0.144379E+00
            cl[6] = +0.388817E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv5'):
            cl[0] = +0.128202E+02
            cl[1] = -0.376275E+02
            cl[2] = -0.487688E+02
            cl[3] = +0.549854E+02
            cl[4] = -0.270916E+01
            cl[5] = +0.142733E+00
            cl[6] = -0.319614E+02

        if (reaction == 'he4_plus_si28_to_p_p31_rv6'):
            cl[0] = +0.381739E+02
            cl[1] = -0.406821E+02
            cl[2] = -0.546650E+02
            cl[3] = +0.331135E+02
            cl[4] = -0.644696E+00
            cl[5] = -0.155955E-02
            cl[6] = -0.271330E+02

        if (reaction == 'he4_plus_o16_to_ne20_n'):
            cl[0] = +0.390340E+02
            cl[1] = -0.358600E-01
            cl[2] = -0.343457E+02
            cl[3] = -0.251939E+02
            cl[4] = +0.479855E+01
            cl[5] = -0.146444E+01
            cl[6] = +0.634333E+01

        if (reaction == 'he4_plus_o16_to_ne20_r'):
            cl[0] = +0.845522E+02
            cl[1] = -0.178214E+02
            cl[2] = +0.293664E+03
            cl[3] = -0.384974E+03
            cl[4] = +0.202380E+02
            cl[5] = -0.100379E+01
            cl[6] = +0.199693E+03

        if (reaction == 'he4_plus_ne20_to_mg24_n'):
            cl[0] = +0.321588E+02
            cl[1] = -0.151494E-01
            cl[2] = -0.446410E+02
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.193576E+01

        if (reaction == 'he4_plus_ne20_to_mg24_r'):
            cl[0] = -0.291641E+03
            cl[1] = -0.120966E+02
            cl[2] = -0.633725E+02
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.121219E+03

        if (reaction == 'mg24_to_he4_ne20_nv'):
            cl[0] = +0.569781E+02
            cl[1] = -0.108074E+03
            cl[2] = -0.446410E+02
            cl[3] = -0.833867E+01
            cl[4] = +0.241631E+01
            cl[5] = -0.778056E+00
            cl[6] = +0.343576E+01

        if (reaction == 'mg24_to_he4_ne20_rv'):
            cl[0] = -0.266822E+03
            cl[1] = -0.120156E+03
            cl[2] = -0.633725E+02
            cl[3] = +0.394643E+03
            cl[4] = -0.362432E+02
            cl[5] = +0.264060E+01
            cl[6] = -0.119719E+03

        if (reaction == 'p_plus_na23_to_he4_ne20_n'):
            cl[0] = +0.334868E+03
            cl[1] = -0.247143E+00
            cl[2] = +0.371150E+02
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r1'):
            cl[0] = +0.942806E+02
            cl[1] = -0.312034E+01
            cl[2] = +0.100052E+03
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'p_plus_na23_to_he4_ne20_r2'):
            cl[0] = -0.288152E+02
            cl[1] = -0.447000E+00
            cl[2] = -0.184674E-09
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01

        if (reaction == 'he4_plus_si28_to_c12_ne20_r'):
            cl[0] = -0.307762E+03
            cl[1] = -0.186722E+03
            cl[2] = +0.514197E+03
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'p_plus_p31_to_c12_ne20_r'):
            cl[0] = -0.266452E+03
            cl[1] = -0.156019E+03
            cl[2] = +0.361154E+03
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03

        if (reaction == 'c12_plus_ne20_to_p_p31_r'):
            cl[0] = -0.268136E+03
            cl[1] = -0.387624E+02
            cl[2] = +0.361154E+03
            cl[3] = -0.926430E+02
            cl[4] = -0.998738E+01
            cl[5] = +0.892737E+00
            cl[6] = +0.161042E+03

        if (reaction == 'c12_plus_ne20_to_he4_si28_r'):
            cl[0] = -0.308905E+03
            cl[1] = -0.472175E+02
            cl[2] = +0.514197E+03
            cl[3] = -0.200896E+03
            cl[4] = -0.642713E+01
            cl[5] = +0.758256E+00
            cl[6] = +0.236359E+03

        if (reaction == 'he4_plus_ne20_to_p_na23_n'):
            cl[0] = +0.335091E+03
            cl[1] = -0.278531E+02
            cl[2] = +0.371150E+02
            cl[3] = -0.478518E+03
            cl[4] = +0.190867E+03
            cl[5] = -0.136026E+03
            cl[6] = +0.979858E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r1'):
            cl[0] = +0.945037E+02
            cl[1] = -0.307263E+02
            cl[2] = +0.100052E+03
            cl[3] = -0.193413E+03
            cl[4] = +0.123467E+02
            cl[5] = -0.781799E+00
            cl[6] = +0.890392E+02

        if (reaction == 'he4_plus_ne20_to_p_na23_r2'):
            cl[0] = -0.285920E+02
            cl[1] = -0.280530E+02
            cl[2] = -0.184674E-09
            cl[3] = +0.614357E-09
            cl[4] = -0.658195E-10
            cl[5] = +0.593159E-11
            cl[6] = -0.150000E+01

        return cl


    def getInuc(self, network, element):
        inuc_tmp = int(network.index(element))
        if inuc_tmp < 10:
            inuc = '000' + str(inuc_tmp)
        if inuc_tmp >= 10 and inuc_tmp < 100:
            inuc = '00' + str(inuc_tmp)
        if inuc_tmp >= 100 and inuc_tmp < 1000:
            inuc = '0' + str(inuc_tmp)
        return inuc


