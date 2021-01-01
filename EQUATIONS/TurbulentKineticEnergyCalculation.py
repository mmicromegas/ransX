import numpy as np
import UTILS.Calculus as uCalc
import UTILS.Tools as uT


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentKineticEnergyCalculation(uCalc.Calculus, uT.Tools, object):

    def __init__(self, filename, ig, intc):
        super(TurbulentKineticEnergyCalculation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        nx = self.getRAdata(eht, 'nx')
        ny = self.getRAdata(eht, 'ny')
        nz = self.getRAdata(eht, 'nz')

        xzn0 = self.getRAdata(eht, 'xzn0')
        xznl = self.getRAdata(eht, 'xznl')
        xznr = self.getRAdata(eht, 'xznr')

        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        # dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduxuy = self.getRAdata(eht, 'dduxuy')[intc]
        dduxuz = self.getRAdata(eht, 'dduxuz')[intc]
        dduyux = dduxuy
        dduzux = dduxuz

        dduxuxux = self.getRAdata(eht, 'dduxuxux')[intc]
        dduxuyuy = self.getRAdata(eht, 'dduxuyuy')[intc]
        dduxuzuz = self.getRAdata(eht, 'dduxuzuz')[intc]

        ddekux = self.getRAdata(eht, 'ddekux')[intc]
        ddek = self.getRAdata(eht, 'ddek')[intc]

        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        ppdivux = self.getRAdata(eht, 'ppdivux')[intc]
        ppdivuy = self.getRAdata(eht, 'ppdivuy')[intc]
        ppdivuz = self.getRAdata(eht, 'ppdivuz')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        divux = self.getRAdata(eht, 'divux')[intc]
        divuy = self.getRAdata(eht, 'divuy')[intc]
        divuz = self.getRAdata(eht, 'divuz')[intc]

        ppux = self.getRAdata(eht, 'ppux')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]

        ###################################
        # TURBULENT KINETIC ENERGY EQUATION
        ###################################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        t_ddux = self.getRAdata(eht, 'ddux')
        t_dduy = self.getRAdata(eht, 'dduy')
        t_dduz = self.getRAdata(eht, 'dduz')

        t_dduxux = self.getRAdata(eht, 'dduxux')
        t_dduyuy = self.getRAdata(eht, 'dduyuy')
        t_dduzuz = self.getRAdata(eht, 'dduzuz')

        t_uxffuxff = t_dduxux / t_dd - t_ddux * t_ddux / (t_dd * t_dd)
        t_uyffuyff = t_dduyuy / t_dd - t_dduy * t_dduy / (t_dd * t_dd)
        t_uzffuzff = t_dduzuz / t_dd - t_dduz * t_dduz / (t_dd * t_dd)

        t_tke = 0.5 * (t_uxffuxff + t_uyffuyff + t_uzffuzff)

        t_tkex = 0.5 * (t_uxffuxff)
        t_tkey = 0.5 * (t_uyffuyff)
        t_tkez = 0.5 * (t_uzffuzff)

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd

        fht_ek = ddek / dd

        uxffuxff = (dduxux / dd - ddux * ddux / (dd * dd))
        uyffuyff = (dduyuy / dd - dduy * dduy / (dd * dd))
        uzffuzff = (dduzuz / dd - dduz * dduz / (dd * dd))

        tke = 0.5 * (uxffuxff + uyffuyff + uzffuzff)

        tkex = 0.5 * (uxffuxff)
        tkey = 0.5 * (uyffuyff)
        tkez = 0.5 * (uzffuzff)

        # fekx = ddekux - fht_ek * fht_ux

        dduxffuxffuxff = dduxuxux - 2. * dduxux * fht_ux - dduxux * fht_ux - dd * fht_ux * fht_ux * fht_ux - dd * fht_ux * fht_ux * fht_ux
        dduyffuyffuxff = dduxuyuy - 2. * dduyux * fht_uy - dduyuy * fht_ux - dd * fht_uy * fht_uy * fht_ux - dd * fht_uy * fht_uy * fht_ux
        dduzffuzffuxff = dduxuzuz - 2. * dduzux * fht_uz - dduzuz * fht_ux - dd * fht_uz * fht_uz * fht_ux - dd * fht_uz * fht_uz * fht_ux

        fekx = 0.5 * (dduxffuxffuxff + dduyffuyffuxff + dduzffuzffuxff)

        fekxx = 0.5 * (dduxffuxffuxff)
        fekxy = 0.5 * (dduyffuyffuxff)
        fekxz = 0.5 * (dduzffuzffuxff)

        fpx = ppux - pp * ux

        # LHS -dq/dt
        self.minus_dt_dd_tke = -self.dt(t_dd * t_tke, xzn0, t_timec, intc)

        self.minus_dt_dd_tkex = -self.dt(t_dd * t_tkex, xzn0, t_timec, intc)
        self.minus_dt_dd_tkey = -self.dt(t_dd * t_tkey, xzn0, t_timec, intc)
        self.minus_dt_dd_tkez = -self.dt(t_dd * t_tkez, xzn0, t_timec, intc)


        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = -self.Div(dd * fht_ux * tke, xzn0)

        self.minus_div_eht_dd_fht_ux_tkex = -self.Div(dd * fht_ux * tkex, xzn0)
        self.minus_div_eht_dd_fht_ux_tkey = -self.Div(dd * fht_ux * tkey, xzn0)
        self.minus_div_eht_dd_fht_ux_tkez = -self.Div(dd * fht_ux * tkez, xzn0)

        # -div kinetic energy flux
        self.minus_div_fekx = -self.Div(fekx, xzn0)

        self.minus_div_fekxx = -self.Div(fekxx, xzn0)
        self.minus_div_fekxy = -self.Div(fekxy, xzn0)
        self.minus_div_fekxz = -self.Div(fekxz, xzn0)

        # -div acoustic flux
        self.minus_div_fpx = -self.Div(fpx, xzn0)

        # RHS warning ax = overline{+u''_x}
        self.plus_ax = -ux + fht_ux

        # +buoyancy work
        self.plus_wb = self.plus_ax * self.Grad(pp, xzn0)

        # +pressure dilatation
        self.plus_wp = ppdivu - pp * divu

        self.plus_wpx = ppdivux - pp * divux
        self.plus_wpy = ppdivuy - pp * divuy
        self.plus_wpz = ppdivuz - pp * divuz

        # -R grad u

        rxx = dduxux - ddux * ddux / dd
        rxy = dduxuy - ddux * dduy / dd
        rxz = dduxuz - ddux * dduz / dd

        self.minus_r_grad_u = -(rxx * self.Grad(ddux / dd, xzn0) +
                                rxy * self.Grad(dduy / dd, xzn0) +
                                rxz * self.Grad(dduz / dd, xzn0))

        self.minus_rxx_grad_ux = -(rxx * self.Grad(ddux / dd, xzn0))
        self.minus_rxy_grad_uy = -(rxy * self.Grad(dduy / dd, xzn0))
        self.minus_rxz_grad_uz = -(rxz * self.Grad(dduz / dd, xzn0))

        # -res
        self.minus_resTkeEquation = - (self.minus_dt_dd_tke + self.minus_div_eht_dd_fht_ux_tke +
                                       self.plus_wb + self.plus_wp + self.minus_div_fekx +
                                       self.minus_div_fpx + self.minus_r_grad_u)

        self.minus_resTkeEquationX = - (self.minus_dt_dd_tkex + self.minus_div_eht_dd_fht_ux_tkex +
                                       self.plus_wb + self.plus_wpx + self.minus_div_fekxx +
                                       self.minus_div_fpx + self.minus_rxx_grad_ux)

        self.minus_resTkeEquationY = - (self.minus_dt_dd_tkey + self.minus_div_eht_dd_fht_ux_tkey +
                                       + self.plus_wpy + self.minus_div_fekxy + self.minus_rxy_grad_uy)

        self.minus_resTkeEquationZ = - (self.minus_dt_dd_tkez + self.minus_div_eht_dd_fht_ux_tkez +
                                       + self.plus_wpz + self.minus_div_fekxz + self.minus_rxz_grad_uz)


        #######################################
        # END TURBULENT KINETIC ENERGY EQUATION
        #######################################

        # assign global data to be shared across whole class
        self.xzn0 = xzn0
        self.xznl = xznl
        self.xznr = xznr

        self.yzn0 = yzn0
        self.zzn0 = zzn0

        self.tke = tke

        self.tkex = tkex
        self.tkey = tkey
        self.tkez = tkez

        self.t_tkex = t_tkex
        self.t_tkey = t_tkey
        self.t_tkez = t_tkez

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dd = dd
        self.pp = pp
        self.uxux = uxux
        self.t_timec = t_timec  # for the space-time diagrams

        if self.ig == 1:
            self.t_tke = t_tke
        elif self.ig == 2:
            dx = (xzn0[-1] - xzn0[0]) / nx
            dumx = xzn0[0] + np.arange(1, nx, 1) * dx
            t_tke2 = []

            # interpolation due to non-equidistant radial grid
            for i in range(int(t_tke.shape[0])):
                t_tke2.append(np.interp(dumx, xzn0, t_tke[i, :]))

            t_tke_forspacetimediagram = np.asarray(t_tke2)
            self.t_tke = t_tke_forspacetimediagram  # for the space-time diagrams

    def getTKEfield(self):
        # return fields
        field = {'dd': self.dd, 'tke': self.tke, 'xzn0': self.xzn0, 'yzn0': self.yzn0, 'zzn0': self.zzn0,
                 'tkex': self.tkex,
                 'tkey': self.tkey,
                 'tkez': self.tkez,
                 't_tkex': self.t_tkex,
                 't_tkey': self.t_tkey,
                 't_tkez': self.t_tkez,
                 'minus_dt_dd_tke': self.minus_dt_dd_tke,
                 'minus_dt_dd_tkex': self.minus_dt_dd_tkex,
                 'minus_dt_dd_tkey': self.minus_dt_dd_tkey,
                 'minus_dt_dd_tkez': self.minus_dt_dd_tkez,
                 'minus_div_eht_dd_fht_ux_tke': self.minus_div_eht_dd_fht_ux_tke,
                 'minus_div_eht_dd_fht_ux_tkex': self.minus_div_eht_dd_fht_ux_tkex,
                 'minus_div_eht_dd_fht_ux_tkey': self.minus_div_eht_dd_fht_ux_tkey,
                 'minus_div_eht_dd_fht_ux_tkez': self.minus_div_eht_dd_fht_ux_tkez,
                 'minus_div_fekx': self.minus_div_fekx,
                 'minus_div_fekxx': self.minus_div_fekxx,
                 'minus_div_fekxy': self.minus_div_fekxy,
                 'minus_div_fekxz': self.minus_div_fekxz,
                 'minus_div_fpx': self.minus_div_fpx,
                 'plus_ax': self.plus_ax,
                 'plus_wb': self.plus_wb,
                 'plus_wp': self.plus_wp,
                 'plus_wpx': self.plus_wpx,
                 'plus_wpy': self.plus_wpy,
                 'plus_wpz': self.plus_wpz,
                 'minus_r_grad_u': self.minus_r_grad_u,
                 'minus_rxx_grad_ux': self.minus_rxx_grad_ux,
                 'minus_rxy_grad_uy': self.minus_rxy_grad_uy,
                 'minus_rxz_grad_uz': self.minus_rxz_grad_uz,
                 'minus_resTkeEquation': self.minus_resTkeEquation,
                 'minus_resTkeEquationX': self.minus_resTkeEquationX,
                 'minus_resTkeEquationY': self.minus_resTkeEquationY,
                 'minus_resTkeEquationZ': self.minus_resTkeEquationZ,
                 'nx': self.nx, 't_timec': self.t_timec, 't_tke': self.t_tke}

        return {'dd': field['dd'], 'tke': field['tke'], 'xzn0': field['xzn0'], 'yzn0': field['yzn0'],
                'zzn0': field['zzn0'],
                'tkex': field['tkex'],
                'tkey': field['tkey'],
                'tkez': field['tkez'],
                't_tkex': field['t_tkex'],
                't_tkey': field['t_tkey'],
                't_tkez': field['t_tkez'],
                'minus_dt_dd_tke': field['minus_dt_dd_tke'],
                'minus_dt_dd_tkex': field['minus_dt_dd_tkex'],
                'minus_dt_dd_tkey': field['minus_dt_dd_tkey'],
                'minus_dt_dd_tkez': field['minus_dt_dd_tkez'],
                'minus_div_eht_dd_fht_ux_tke': field['minus_div_eht_dd_fht_ux_tke'],
                'minus_div_eht_dd_fht_ux_tkex': field['minus_div_eht_dd_fht_ux_tkex'],
                'minus_div_eht_dd_fht_ux_tkey': field['minus_div_eht_dd_fht_ux_tkey'],
                'minus_div_eht_dd_fht_ux_tkez': field['minus_div_eht_dd_fht_ux_tkez'],
                'minus_div_fekx': field['minus_div_fekx'],
                'minus_div_fekxx': field['minus_div_fekxx'],
                'minus_div_fekxy': field['minus_div_fekxy'],
                'minus_div_fekxz': field['minus_div_fekxz'],
                'minus_div_fpx': field['minus_div_fpx'],
                'plus_ax': field['plus_ax'],
                'plus_wb': field['plus_wb'],
                'plus_wp': field['plus_wp'],
                'plus_wpx': field['plus_wpx'],
                'plus_wpy': field['plus_wpy'],
                'plus_wpz': field['plus_wpz'],
                'minus_r_grad_u': field['minus_r_grad_u'],
                'minus_rxx_grad_ux': field['minus_rxx_grad_ux'],
                'minus_rxy_grad_uy': field['minus_rxy_grad_uy'],
                'minus_rxz_grad_uz': field['minus_rxz_grad_uz'],
                'minus_resTkeEquation': field['minus_resTkeEquation'],
                'minus_resTkeEquationX': field['minus_resTkeEquationX'],
                'minus_resTkeEquationY': field['minus_resTkeEquationY'],
                'minus_resTkeEquationZ': field['minus_resTkeEquationZ'],
                'nx': field['nx'], 't_timec': field['t_timec'], 't_tke': field['t_tke']}
