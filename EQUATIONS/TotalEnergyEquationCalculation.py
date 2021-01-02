import numpy as np
from UTILS.Calculus import Calculus
from UTILS.Tools import Tools


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TotalEnergyEquationCalculation(Calculus, Tools, object):

    def __init__(self, filename, ig, intc):
        super(TotalEnergyEquationCalculation, self).__init__(ig)

        # load data to structured array
        eht = self.customLoad(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')
        nx = self.getRAdata(eht, 'nx')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]

        ddux = self.getRAdata(eht, 'ddux')[intc]
        dduy = self.getRAdata(eht, 'dduy')[intc]
        dduz = self.getRAdata(eht, 'dduz')[intc]

        dduxux = self.getRAdata(eht, 'dduxux')[intc]
        dduyuy = self.getRAdata(eht, 'dduyuy')[intc]
        dduzuz = self.getRAdata(eht, 'dduzuz')[intc]
        dduxuy = self.getRAdata(eht, 'dduxuy')[intc]
        dduxuz = self.getRAdata(eht, 'dduxuz')[intc]

        ddekux = self.getRAdata(eht, 'ddekux')[intc]
        ddek = self.getRAdata(eht, 'ddek')[intc]

        ddei = self.getRAdata(eht, 'ddei')[intc]
        ddeiux = self.getRAdata(eht, 'ddeiux')[intc]

        divu = self.getRAdata(eht, 'divu')[intc]
        ppdivu = self.getRAdata(eht, 'ppdivu')[intc]
        ppux = self.getRAdata(eht, 'ppux')[intc]

        ddenuc1 = self.getRAdata(eht, 'ddenuc1')[intc]
        ddenuc2 = self.getRAdata(eht, 'ddenuc2')[intc]

        #######################
        # TOTAL ENERGY EQUATION
        #######################

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        t_ddei = self.getRAdata(eht, 'ddei')

        t_ddux = self.getRAdata(eht, 'ddux')
        t_dduy = self.getRAdata(eht, 'dduy')
        t_dduz = self.getRAdata(eht, 'dduz')

        t_dduxux = self.getRAdata(eht, 'dduxux')
        t_dduyuy = self.getRAdata(eht, 'dduyuy')
        t_dduzuz = self.getRAdata(eht, 'dduzuz')

        t_uxux = self.getRAdata(eht, 'uxux')
        t_uyuy = self.getRAdata(eht, 'uyuy')
        t_uzuz = self.getRAdata(eht, 'uzuz')

        t_fht_ek = 0.5 * (t_dduxux + t_dduyuy + t_dduzuz) / t_dd
        t_fht_ei = t_ddei / t_dd

        # construct equation-specific mean fields
        # fht_ek = 0.5*(dduxux + dduyuy + dduzuz)/dd
        fht_ek = ddek / dd
        fht_ux = ddux / dd
        fht_ei = ddei / dd

        fei = ddeiux - ddux * ddei / dd
        fekx = ddekux - fht_ux * fht_ek
        fpx = ppux - pp * ux

        # LHS -dq/dt
        self.minus_dt_eht_dd_fht_ek = -self.dt(t_dd * t_fht_ek, xzn0, t_timec, intc)
        self.minus_dt_eht_dd_fht_ei = -self.dt(t_dd * t_fht_ei, xzn0, t_timec, intc)
        self.minus_dt_eht_dd_fht_et = self.minus_dt_eht_dd_fht_ek + \
                                      self.minus_dt_eht_dd_fht_ei

        # LHS -div dd ux te
        self.minus_div_eht_dd_fht_ux_fht_ek = -self.Div(dd * fht_ux * fht_ek, xzn0)
        self.minus_div_eht_dd_fht_ux_fht_ei = -self.Div(dd * fht_ux * fht_ei, xzn0)
        self.minus_div_eht_dd_fht_ux_fht_et = self.minus_div_eht_dd_fht_ux_fht_ek + \
                                              self.minus_div_eht_dd_fht_ux_fht_ei

        # RHS -div fei
        self.minus_div_fei = -self.Div(fei, xzn0)

        # RHS -div ftt (not included) heat flux
        self.minus_div_ftt = -np.zeros(nx)

        # -div kinetic energy flux
        self.minus_div_fekx = -self.Div(fekx, xzn0)

        # -div acoustic flux
        self.minus_div_fpx = -self.Div(fpx, xzn0)

        # RHS warning ax = overline{+u''_x}
        self.plus_ax = -ux + fht_ux

        # +buoyancy work
        self.plus_wb = self.plus_ax * self.Grad(pp, xzn0)

        # RHS -P d = - eht_pp Div eht_ux
        self.minus_pp_div_ux = -pp * self.Div(ux, xzn0)

        # -R grad u

        rxx = dduxux - ddux * ddux / dd
        rxy = dduxuy - ddux * dduy / dd
        rxz = dduxuz - ddux * dduz / dd

        self.minus_r_grad_u = -(rxx * self.Grad(ddux / dd, xzn0) + \
                                rxy * self.Grad(dduy / dd, xzn0) + \
                                rxz * self.Grad(dduz / dd, xzn0))

        # +dd Dt fht_ui_fht_ui_o_two
        t_fht_ux = t_ddux / t_dd
        t_fht_uy = t_dduy / t_dd
        t_fht_uz = t_dduz / t_dd

        fht_ux = ddux / dd
        fht_uy = dduy / dd
        fht_uz = dduz / dd

        self.plus_dd_Dt_fht_ui_fht_ui_o_two = \
            +self.dt(t_dd * (t_fht_ux ** 2. + t_fht_uy ** 2. + t_fht_uz ** 2.), xzn0, t_timec, intc) - \
            self.Div(dd * fht_ux * (fht_ux ** 2. + fht_uy ** 2. + fht_uz ** 2.), xzn0) / 2.

        # RHS source + dd enuc
        self.plus_dd_fht_enuc = ddenuc1 + ddenuc2

        # -res
        self.minus_resTeEquation = - (self.minus_dt_eht_dd_fht_et + self.minus_div_eht_dd_fht_ux_fht_et +
                                      self.minus_div_fei + self.minus_div_ftt + self.minus_div_fekx +
                                      self.minus_div_fpx + self.minus_r_grad_u + self.minus_pp_div_ux +
                                      self.plus_wb + self.plus_dd_fht_enuc + self.plus_dd_Dt_fht_ui_fht_ui_o_two)

        ###########################
        # END TOTAL ENERGY EQUATION
        ###########################

    def getTotalEnergyEquationField(self):
        # return fields
        field = {'minus_resTeEquation': self.minus_resTeEquation}

        return {'minus_resTeEquation': field['minus_resTeEquation']}
