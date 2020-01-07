import numpy as np
import UTILS.Calculus as calc


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentKineticEnergyCalculation(calc.Calculus, object):

    def __init__(self, filename, ig, ieos, intc):
        super(TurbulentKineticEnergyCalculation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        timec = eht.item().get('timec')[intc]
        tavg = np.asarray(eht.item().get('tavg'))
        trange = np.asarray(eht.item().get('trange'))

        # load grid
        nx = np.asarray(eht.item().get('nx'))
        ny = np.asarray(eht.item().get('ny'))
        nz = np.asarray(eht.item().get('nz'))

        xzn0 = np.asarray(eht.item().get('xzn0'))
        xznl = np.asarray(eht.item().get('xznl'))
        xznr = np.asarray(eht.item().get('xznr'))

        yzn0 = np.asarray(eht.item().get('yzn0'))
        zzn0 = np.asarray(eht.item().get('zzn0'))

        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])
        pp = np.asarray(eht.item().get('pp')[intc])

        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduxuy = np.asarray(eht.item().get('dduxuy')[intc])
        dduxuz = np.asarray(eht.item().get('dduxuz')[intc])

        ddekux = np.asarray(eht.item().get('ddekux')[intc])
        ddek = np.asarray(eht.item().get('ddek')[intc])

        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])
        divu = np.asarray(eht.item().get('divu')[intc])
        ppux = np.asarray(eht.item().get('ppux')[intc])

        uxux = np.asarray(eht.item().get('uxux')[intc])

        ###################################
        # TURBULENT KINETIC ENERGY EQUATION
        ###################################

        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))
        t_dd = np.asarray(eht.item().get('dd'))

        t_ddux = np.asarray(eht.item().get('ddux'))
        t_dduy = np.asarray(eht.item().get('dduy'))
        t_dduz = np.asarray(eht.item().get('dduz'))

        t_dduxux = np.asarray(eht.item().get('dduxux'))
        t_dduyuy = np.asarray(eht.item().get('dduyuy'))
        t_dduzuz = np.asarray(eht.item().get('dduzuz'))

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

        self.minus_r_grad_u = -(rxx * self.Grad(ddux / dd, xzn0) + \
                                rxy * self.Grad(dduy / dd, xzn0) + \
                                rxz * self.Grad(dduz / dd, xzn0))

        # -res
        self.minus_resTkeEquation = - (self.minus_dt_dd_tke + self.minus_div_eht_dd_fht_ux_tke + \
                                       self.plus_wb + self.plus_wp + self.minus_div_fekx + \
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

    def getTKEfield(self):
        # return fields
        field = {'dd':self.dd, 'tke': self.tke, 'xzn0': self.xzn0, 'minus_dt_dd_tke': self.minus_dt_dd_tke,
        'minus_div_eht_dd_fht_ux_tke': self.minus_div_eht_dd_fht_ux_tke,
        'minus_div_fekx': self.minus_div_fekx,
        'minus_div_fpx': self.minus_div_fpx,
        'plus_ax': self.plus_ax,
        'plus_wb': self.plus_wb,
        'plus_wp': self.plus_wp,
        'minus_r_grad_u': self.minus_r_grad_u,
        'minus_resTkeEquation': self.minus_resTkeEquation}

        return{'dd': field['dd'],'tke': field['tke'], 'xzn0': field['xzn0'],'minus_dt_dd_tke': field['minus_dt_dd_tke'],
               'minus_div_eht_dd_fht_ux_tke': field['minus_div_eht_dd_fht_ux_tke'],
               'minus_div_fekx': field['minus_div_fekx'],
               'minus_div_fpx': field['minus_div_fpx'],
               'plus_ax': field['plus_ax'],
               'plus_wb': field['plus_wb'],
               'plus_wp': field['plus_wp'],
               'minus_r_grad_u': field['minus_r_grad_u'],
               'minus_resTkeEquation': field['minus_resTkeEquation']}


