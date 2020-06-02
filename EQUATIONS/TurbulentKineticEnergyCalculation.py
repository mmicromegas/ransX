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
        eht = np.load(filename)

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

        ddekux = self.getRAdata(eht, 'ddekux')[intc]
        ddek = self.getRAdata(eht, 'ddek')[intc]

        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        divu = self.getRAdata(eht, 'divu')[intc]
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

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fht_ek = ddek / dd

        uxffuxff = (dduxux / dd - ddux * ddux / (dd * dd))
        uyffuyff = (dduyuy / dd - dduy * dduy / (dd * dd))
        uzffuzff = (dduzuz / dd - dduz * dduz / (dd * dd))

        tke = 0.5 * (uxffuxff + uyffuyff + uzffuzff)

        fekx = ddekux - fht_ek * fht_ux
        fpx = ppux - pp * ux

        # LHS -dq/dt
        self.minus_dt_dd_tke = -self.dt(t_dd * t_tke, xzn0, t_timec, intc)

        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = -self.Div(dd * fht_ux * tke, xzn0)

        # -div kinetic energy flux
        self.minus_div_fekx = -self.Div(fekx, xzn0)

        # -div acoustic flux
        self.minus_div_fpx = -self.Div(fpx, xzn0)

        # RHS warning ax = overline{+u''_x}
        self.plus_ax = -ux + fht_ux

        # +buoyancy work
        self.plus_wb = self.plus_ax * self.Grad(pp, xzn0)

        # +pressure dilatation
        self.plus_wp = ppdivu - pp * divu

        # -R grad u

        rxx = dduxux - ddux * ddux / dd
        rxy = dduxuy - ddux * dduy / dd
        rxz = dduxuz - ddux * dduz / dd

        self.minus_r_grad_u = -(rxx * self.Grad(ddux / dd, xzn0) +
                                rxy * self.Grad(dduy / dd, xzn0) +
                                rxz * self.Grad(dduz / dd, xzn0))

        # -res
        self.minus_resTkeEquation = - (self.minus_dt_dd_tke + self.minus_div_eht_dd_fht_ux_tke +
                                       self.plus_wb + self.plus_wp + self.minus_div_fekx +
                                       self.minus_div_fpx + self.minus_r_grad_u)

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

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dd = dd
        self.pp = pp
        self.uxux = uxux
        self.t_timec = t_timec # for the space-time diagrams

        if self.ig == 1:
            self.t_tke = t_tke
        elif self.ig == 2:
            dx = (xzn0[-1]-xzn0[0])/nx
            dumx = xzn0[0]+np.arange(1,nx,1)*dx
            t_tke2 = []

            # interpolation due to non-equidistant radial grid
            for i in range(int(t_tke.shape[0])):
                t_tke2.append(np.interp(dumx,xzn0,t_tke[i,:]))

            t_tke_forspacetimediagram = np.asarray(t_tke2)
            self.t_tke = t_tke_forspacetimediagram # for the space-time diagrams

    def getTKEfield(self):
        # return fields
        field = {'dd': self.dd, 'tke': self.tke, 'xzn0': self.xzn0, 'minus_dt_dd_tke': self.minus_dt_dd_tke,
                 'minus_div_eht_dd_fht_ux_tke': self.minus_div_eht_dd_fht_ux_tke,
                 'minus_div_fekx': self.minus_div_fekx,
                 'minus_div_fpx': self.minus_div_fpx,
                 'plus_ax': self.plus_ax,
                 'plus_wb': self.plus_wb,
                 'plus_wp': self.plus_wp,
                 'minus_r_grad_u': self.minus_r_grad_u,
                 'minus_resTkeEquation': self.minus_resTkeEquation,
                 'nx': self.nx, 't_timec': self.t_timec,'t_tke': self.t_tke}

        return {'dd': field['dd'], 'tke': field['tke'], 'xzn0': field['xzn0'],
                'minus_dt_dd_tke': field['minus_dt_dd_tke'],
                'minus_div_eht_dd_fht_ux_tke': field['minus_div_eht_dd_fht_ux_tke'],
                'minus_div_fekx': field['minus_div_fekx'],
                'minus_div_fpx': field['minus_div_fpx'],
                'plus_ax': field['plus_ax'],
                'plus_wb': field['plus_wb'],
                'plus_wp': field['plus_wp'],
                'minus_r_grad_u': field['minus_r_grad_u'],
                'minus_resTkeEquation': field['minus_resTkeEquation'],
                'nx': field['nx'], 't_timec': field['t_timec'],'t_tke': field['t_tke']}
