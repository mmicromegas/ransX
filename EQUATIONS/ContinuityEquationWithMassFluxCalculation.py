import numpy as np
import UTILS.Calculus as uCalc
import UTILS.Tools as uT


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ContinuityEquationWithMassFluxCalculation(uCalc.Calculus, uT.Tools, object):

    def __init__(self, filename, ig, intc):
        super(ContinuityEquationWithMassFluxCalculation, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        # load grid
        xzn0 = self.getRAdata(eht, 'xzn0')

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd = self.getRAdata(eht, 'dd')[intc]
        ux = self.getRAdata(eht, 'ux')[intc]
        ddux = self.getRAdata(eht, 'ddux')[intc]

        # store time series for time derivatives
        t_timec = self.getRAdata(eht, 'timec')
        t_dd = self.getRAdata(eht, 'dd')

        # t_mm    = self.getRAdata(eht,'mm'))
        # minus_dt_mm = -self.dt(t_mm,xzn0,t_timec,intc)
        # fht_ux = minus_dt_mm/(4.*np.pi*(xzn0**2.)*dd)

        # construct equation-specific mean fields
        fht_ux = ddux / dd
        fdd = ddux - dd * ux

        ####################################
        # CONTINUITY EQUATION WITH MASS FLUX
        ####################################

        # LHS -dq/dt
        self.minus_dt_dd = -self.dt(t_dd, xzn0, t_timec, intc)

        # LHS -fht_ux Grad dd
        self.minus_fht_ux_grad_dd = -fht_ux * self.Grad(dd, xzn0)

        # RHS -Div fdd
        self.minus_div_fdd = -self.Div(fdd, xzn0)

        # RHS +fdd_o_dd gradx dd
        self.plus_fdd_o_dd_gradx_dd = +(fdd / dd) * self.Grad(dd, xzn0)

        # RHS -dd Div ux
        self.minus_dd_div_ux = -dd * self.Div(ux, xzn0)

        # -res
        self.minus_resContEquation = -(self.minus_dt_dd + self.minus_fht_ux_grad_dd + self.minus_div_fdd +
                                       self.plus_fdd_o_dd_gradx_dd + self.minus_dd_div_ux)

        ########################################
        # END CONTINUITY EQUATION WITH MASS FLUX
        ########################################

    def getCONTfield(self):
        # return fields
        field = {'minus_resContEquation': self.minus_resContEquation}

        return {'minus_resContEquation': field['minus_resContEquation']}
