from EQUATIONS.ContinuityEquationWithMassFlux import ContinuityEquationWithMassFlux
from EQUATIONS.ContinuityEquationWithFavrianDilatation import ContinuityEquationWithFavrianDilatation

from EQUATIONS.MomentumEquationX import MomentumEquationX
from EQUATIONS.MomentumEquationY import MomentumEquationY
from EQUATIONS.MomentumEquationZ import MomentumEquationZ

from EQUATIONS.ReynoldsStressXXequation import ReynoldsStressXXequation
from EQUATIONS.ReynoldsStressYYequation import ReynoldsStressYYequation
from EQUATIONS.ReynoldsStressZZequation import ReynoldsStressZZequation

from EQUATIONS.TurbulentKineticEnergyEquation import TurbulentKineticEnergyEquation
from EQUATIONS.TurbulentKineticEnergyEquationRadial import TurbulentKineticEnergyEquationRadial
from EQUATIONS.TurbulentKineticEnergyEquationHorizontal import TurbulentKineticEnergyEquationHorizontal

from EQUATIONS.InternalEnergyEquation import InternalEnergyEquation
from EQUATIONS.InternalEnergyFluxEquation import InternalEnergyFluxEquation
from EQUATIONS.InternalEnergyVarianceEquation import InternalEnergyVarianceEquation

from EQUATIONS.KineticEnergyEquation import KineticEnergyEquation
from EQUATIONS.TotalEnergyEquation import TotalEnergyEquation

from EQUATIONS.EntropyEquation import EntropyEquation
from EQUATIONS.EntropyFluxEquation import EntropyFluxEquation
from EQUATIONS.EntropyVarianceEquation import EntropyVarianceEquation

from EQUATIONS.PressureEquation import PressureEquation

from EQUATIONS.PressureFluxXequation import PressureFluxXequation
from EQUATIONS.PressureFluxYequation import PressureFluxYequation
from EQUATIONS.PressureFluxZequation import PressureFluxZequation

from EQUATIONS.PressureVarianceEquation import PressureVarianceEquation

from EQUATIONS.TemperatureEquation import TemperatureEquation
from EQUATIONS.TemperatureFluxEquation import TemperatureFluxEquation
from EQUATIONS.TemperatureVarianceEquation import TemperatureVarianceEquation

from EQUATIONS.EnthalpyEquation import EnthalpyEquation
from EQUATIONS.EnthalpyFluxEquation import EnthalpyFluxEquation
from EQUATIONS.EnthalpyVarianceEquation import EnthalpyVarianceEquation

from EQUATIONS.DensityVarianceEquation import DensityVarianceEquation
from EQUATIONS.TurbulentMassFluxEquation import TurbulentMassFluxEquation
from EQUATIONS.DensitySpecificVolumeCovarianceEquation import DensitySpecificVolumeCovarianceEquation

from EQUATIONS.XtransportEquation import XtransportEquation

from EQUATIONS.XfluxXequation import XfluxXequation
from EQUATIONS.XfluxYequation import XfluxYequation
from EQUATIONS.XfluxZequation import XfluxZequation

from EQUATIONS.XvarianceEquation import XvarianceEquation
from EQUATIONS.Xdiffusivity import Xdiffusivity
from EQUATIONS.XdamkohlerNumber import XdamkohlerNumber

from EQUATIONS.AbarTransportEquation import AbarTransportEquation
from EQUATIONS.ZbarTransportEquation import ZbarTransportEquation

from EQUATIONS.AbarFluxTransportEquation import AbarFluxTransportEquation
from EQUATIONS.ZbarFluxTransportEquation import ZbarFluxTransportEquation

from EQUATIONS.TemperatureDensity import TemperatureDensity
from EQUATIONS.PressureInternalEnergy import PressureInternalEnergy
from EQUATIONS.NuclearEnergyProduction import NuclearEnergyProduction
from EQUATIONS.Gravity import Gravity
from EQUATIONS.TemperatureGradients import TemperatureGradients
from EQUATIONS.Degeneracy import Degeneracy
from EQUATIONS.VelocitiesMeanExp import VelocitiesMeanExp
from EQUATIONS.VelocitiesMLTturb import VelocitiesMLTturb
from EQUATIONS.RelativeRMSflct import RelativeRMSflct
from EQUATIONS.AbarZbar import AbarZbar
from EQUATIONS.BruntVaisalla import BruntVaisalla
from EQUATIONS.Buoyancy import Buoyancy

# import classes for hydrodynamic stellar structure equations
from EQUATIONS.HsseContinuityEquation import HsseContinuityEquation
from EQUATIONS.HsseMomentumEquationX import HsseMomentumEquationX
from EQUATIONS.HsseTemperatureEquation import HsseTemperatureEquation
from EQUATIONS.HsseLuminosityEquation import HsseLuminosityEquation
from EQUATIONS.HsseXtransportEquation import HsseXtransportEquation

# from class for full turbulence velocity field hypothesis
from EQUATIONS.FullTurbulenceVelocityFieldHypothesisX import FullTurbulenceVelocityFieldHypothesisX
from EQUATIONS.FullTurbulenceVelocityFieldHypothesisY import FullTurbulenceVelocityFieldHypothesisY
from EQUATIONS.FullTurbulenceVelocityFieldHypothesisZ import FullTurbulenceVelocityFieldHypothesisZ

from EQUATIONS.UxfpdIdentity import UxfpdIdentity
from EQUATIONS.UyfpdIdentity import UyfpdIdentity
from EQUATIONS.UzfpdIdentity import UzfpdIdentity

from EQUATIONS.DivuDilatation import DivuDilatation

import matplotlib.pyplot as plt


class MasterPlot():

    def __init__(self, params):
        self.params = params
        #plt.close()

    def execRho(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransCONT = ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],
                                                                    params.getForProp('prop')['ig'],
                                                                    params.getForProp('prop')['fext'],
                                                                    params.getForProp('prop')['intc'],
                                                                    params.getForProp('prop')['nsdim'],
                                                                    params.getForProp('prop')['prefix'])

        # plot density
        ransCONT.plot_rho(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('rho')['xbl'],
                          params.getForEqs('rho')['xbr'],
                          params.getForEqs('rho')['ybu'],
                          params.getForEqs('rho')['ybd'],
                          params.getForEqs('rho')['ilg'])

        # ransCONT.plot_mm_vs_MM(params.getForProp('prop')['laxis'],
        #                       params.getForEqs('rho')['xbl'],
        #                       params.getForEqs('rho')['xbr'],
        #                       params.getForEqs('rho')['ybu'],
        #                       params.getForEqs('rho')['ybd'],
        #                       params.getForEqs('rho')['ilg'])

    def execContEq(self, wxStudio, bconv, tconv):
        params = self.params

        # instantiate 
        ransCONT = ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],
                                                                    params.getForProp('prop')['ig'],
                                                                    params.getForProp('prop')['fext'],
                                                                    params.getForProp('prop')['intc'],
                                                                    params.getForProp('prop')['nsdim'],
                                                                    params.getForProp('prop')['prefix'])

        # plot continuity equation
        if wxStudio[0]:
            ransCONT.plot_continuity_equation(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              wxStudio[1],
                                              wxStudio[2],
                                              wxStudio[3],
                                              wxStudio[4],
                                              params.getForEqs('conteq')['ilg'])
        else:
            ransCONT.plot_continuity_equation(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              params.getForEqs('conteq')['xbl'],
                                              params.getForEqs('conteq')['xbr'],
                                              params.getForEqs('conteq')['ybu'],
                                              params.getForEqs('conteq')['ybd'],
                                              params.getForEqs('conteq')['ilg'])

    def execContEqBar(self):
        params = self.params

        # instantiate 
        ransCONT = ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],
                                                                    params.getForProp('prop')['ig'],
                                                                    params.getForProp('prop')['fext'],
                                                                    params.getForProp('prop')['intc'],
                                                                    params.getForProp('prop')['nsdim'],
                                                                    params.getForProp('prop')['prefix'])

        # plot continuity equation integral budget					       
        ransCONT.plot_continuity_equation_integral_budget(params.getForProp('prop')['laxis'],
                                                          params.getForEqsBar('conteqBar')['xbl'],
                                                          params.getForEqsBar('conteqBar')['xbr'],
                                                          params.getForEqsBar('conteqBar')['ybu'],
                                                          params.getForEqsBar('conteqBar')['ybd'])

    def execContFddEq(self, wxStudio, bconv, tconv):
        params = self.params

        # instantiate 
        ransCONTfdd = ContinuityEquationWithMassFlux(params.getForProp('prop')['eht_data'],
                                                             params.getForProp('prop')['ig'],
                                                             params.getForProp('prop')['fext'],
                                                             params.getForProp('prop')['intc'],
                                                             params.getForProp('prop')['nsdim'],
                                                             params.getForProp('prop')['prefix'])

        # plot continuity equation
        if wxStudio[0]:
            ransCONTfdd.plot_continuity_equation(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              wxStudio[1],
                                              wxStudio[2],
                                              wxStudio[3],
                                              wxStudio[4],
                                              params.getForEqs('conteq')['ilg'])
        else:
            ransCONTfdd.plot_continuity_equation(wxStudio[0], params.getForProp('prop')['laxis'],
                                             bconv, tconv,
                                             params.getForEqs('conteqfdd')['xbl'],
                                             params.getForEqs('conteqfdd')['xbr'],
                                             params.getForEqs('conteqfdd')['ybu'],
                                             params.getForEqs('conteqfdd')['ybd'],
                                             params.getForEqs('conteqfdd')['ilg'])

        # ransCONTfdd.plot_Frho_space_time(params.getForProp('prop')['laxis'],
        #                                 bconv, tconv,
        #                                 params.getForEqs('conteqfdd')['xbl'],
        #                                 params.getForEqs('conteqfdd')['xbr'],
        #                                 params.getForEqs('conteqfdd')['ybu'],
        #                                 params.getForEqs('conteqfdd')['ybd'],
        #                                 params.getForEqs('conteqfdd')['ilg'])

    def execContFddEqBar(self):
        params = self.params

        # instantiate 
        ransCONTfdd = ContinuityEquationWithMassFlux(params.getForProp('prop')['eht_data'],
                                                             params.getForProp('prop')['ig'],
                                                             params.getForProp('prop')['fext'],
                                                             params.getForProp('prop')['intc'],
                                                             params.getForProp('prop')['nsdim'],
                                                             params.getForProp('prop')['prefix'])

        # plot continuity equation integral budget					       
        ransCONTfdd.plot_continuity_equation_integral_budget(params.getForProp('prop')['laxis'],
                                                             params.getForEqsBar('conteqfddBar')['xbl'],
                                                             params.getForEqsBar('conteqfddBar')['xbr'],
                                                             params.getForEqsBar('conteqfddBar')['ybu'],
                                                             params.getForEqsBar('conteqfddBar')['ybd'])

    def execHssContEq(self, bconv, tconv):
        params = self.params

        # instantiate 
        ranshssecont = HsseContinuityEquation(params.getForProp('prop')['eht_data'],
                                                       params.getForProp('prop')['ig'],
                                                       params.getForProp('prop')['ieos'],
                                                       params.getForProp('prop')['fext'],
                                                       params.getForProp('prop')['intc'],
                                                       params.getForProp('prop')['prefix'],
                                                       bconv, tconv)

        # plot continuity equation						       
        ranshssecont.plot_continuity_equation(params.getForProp('prop')['laxis'],
                                              params.getForEqs('cteqhsse')['xbl'],
                                              params.getForEqs('cteqhsse')['xbr'],
                                              params.getForEqs('cteqhsse')['ybu'],
                                              params.getForEqs('cteqhsse')['ybd'],
                                              params.getForEqs('cteqhsse')['ilg'])

        # plot continuity equation alternative						       
        ranshssecont.plot_continuity_equation_2(params.getForProp('prop')['laxis'],
                                                params.getForEqs('cteqhsse')['xbl'],
                                                params.getForEqs('cteqhsse')['xbr'],
                                                params.getForEqs('cteqhsse')['ybu'],
                                                params.getForEqs('cteqhsse')['ybd'],
                                                params.getForEqs('cteqhsse')['ilg'])

        # plot continuity equation alternative simplified						       
        ranshssecont.plot_continuity_equation_3(params.getForProp('prop')['laxis'],
                                                params.getForEqs('cteqhsse')['xbl'],
                                                params.getForEqs('cteqhsse')['xbr'],
                                                params.getForEqs('cteqhsse')['ybu'],
                                                params.getForEqs('cteqhsse')['ybd'],
                                                params.getForEqs('cteqhsse')['ilg'])

        # plot continuity equation alternative simplified - cracking on velocities

        #        ranshssecont.plot_velocities(params.getForProp('prop')['laxis'],\
        #                                             params.getForEqs('cteqhsse')['xbl'],\
        #                                             params.getForEqs('cteqhsse')['xbr'],\
        #                                             params.getForEqs('cteqhsse')['ybu'],\
        #                                             params.getForEqs('cteqhsse')['ybd'],\
        #                                             params.getForEqs('cteqhsse')['ilg'])

        ranshssecont.plot_dilatation_flux(params.getForProp('prop')['laxis'],
                                          params.getForEqs('cteqhsse')['xbl'],
                                          params.getForEqs('cteqhsse')['xbr'],
                                          params.getForEqs('cteqhsse')['ybu'],
                                          params.getForEqs('cteqhsse')['ybd'],
                                          params.getForEqs('cteqhsse')['ilg'])

    #        ranshssecont.plot_mass_flux_acceleration(params.getForProp('prop')['laxis'],\
    #                                             params.getForEqs('cteqhsse')['xbl'],\
    #                                             params.getForEqs('cteqhsse')['xbr'],\
    #                                             params.getForEqs('cteqhsse')['ybu'],\
    #                                             params.getForEqs('cteqhsse')['ybd'],\
    #                                             params.getForEqs('cteqhsse')['ilg'])

    def execHssMomxEq(self, bconv, tconv):
        params = self.params

        # instantiate 
        ranshssemomx = HsseMomentumEquationX(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['ieos'],
                                                      params.getForProp('prop')['fext'],
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['prefix'],
                                                      bconv, tconv)

        # plot hsse momentm equation						       
        ranshssemomx.plot_momentum_equation_x(params.getForProp('prop')['laxis'],
                                              params.getForEqs('mxeqhsse')['xbl'],
                                              params.getForEqs('mxeqhsse')['xbr'],
                                              params.getForEqs('mxeqhsse')['ybu'],
                                              params.getForEqs('mxeqhsse')['ybd'],
                                              params.getForEqs('mxeqhsse')['ilg'])

        # plot hsse momentm equation alternative						       
        ranshssemomx.plot_momentum_equation_x_2(params.getForProp('prop')['laxis'],
                                                params.getForEqs('mxeqhsse')['xbl'],
                                                params.getForEqs('mxeqhsse')['xbr'],
                                                params.getForEqs('mxeqhsse')['ybu'],
                                                params.getForEqs('mxeqhsse')['ybd'],
                                                params.getForEqs('mxeqhsse')['ilg'])

        # plot hsse momentm equation alternative simplified						       
        ranshssemomx.plot_momentum_equation_x_3(params.getForProp('prop')['laxis'],
                                                params.getForEqs('mxeqhsse')['xbl'],
                                                params.getForEqs('mxeqhsse')['xbr'],
                                                params.getForEqs('mxeqhsse')['ybu'],
                                                params.getForEqs('mxeqhsse')['ybd'],
                                                params.getForEqs('mxeqhsse')['ilg'])

    def execHssTempEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 
        ranshssetemp = HsseTemperatureEquation(params.getForProp('prop')['eht_data'],
                                                        params.getForProp('prop')['ig'],
                                                        params.getForProp('prop')['ieos'],
                                                        params.getForProp('prop')['fext'],
                                                        params.getForProp('prop')['intc'],
                                                        tke_diss, bconv, tconv,
                                                        params.getForProp('prop')['prefix'])

        # plot hsse temperature equation						       
        ranshssetemp.plot_tt_equation(params.getForProp('prop')['laxis'],
                                      params.getForEqs('tpeqhsse')['xbl'],
                                      params.getForEqs('tpeqhsse')['xbr'],
                                      params.getForEqs('tpeqhsse')['ybu'],
                                      params.getForEqs('tpeqhsse')['ybd'],
                                      params.getForEqs('tpeqhsse')['ilg'])

        # plot hsse temperature equation alternative						       
        ranshssetemp.plot_tt_equation_2(params.getForProp('prop')['laxis'],
                                        params.getForEqs('tpeqhsse')['xbl'],
                                        params.getForEqs('tpeqhsse')['xbr'],
                                        params.getForEqs('tpeqhsse')['ybu'],
                                        params.getForEqs('tpeqhsse')['ybd'],
                                        params.getForEqs('tpeqhsse')['ilg'])

        # plot hsse temperature equation alternative simplified						       
        ranshssetemp.plot_tt_equation_3(params.getForProp('prop')['laxis'],
                                        params.getForEqs('tpeqhsse')['xbl'],
                                        params.getForEqs('tpeqhsse')['xbr'],
                                        params.getForEqs('tpeqhsse')['ybu'],
                                        params.getForEqs('tpeqhsse')['ybd'],
                                        params.getForEqs('tpeqhsse')['ilg'])

    def execHssLumiEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 
        ranshsselumi = HsseLuminosityEquation(params.getForProp('prop')['eht_data'],
                                                       params.getForProp('prop')['ig'],
                                                       params.getForProp('prop')['ieos'],
                                                       params.getForProp('prop')['fext'],
                                                       params.getForProp('prop')['intc'],
                                                       tke_diss, bconv, tconv,
                                                       params.getForProp('prop')['prefix'])

        # plot hsse luminosity equation						       
        # ranshsselumi.plot_luminosity_equation(params.getForProp('prop')['laxis'],
        #                                      params.getForEqs('lueqhsse')['xbl'],
        #                                      params.getForEqs('lueqhsse')['xbr'],
        #                                      params.getForEqs('lueqhsse')['ybu'],
        #                                      params.getForEqs('lueqhsse')['ybd'],
        #                                      params.getForEqs('lueqhsse')['ilg'])

        # plot hsse luminosity equation exact						       
        ranshsselumi.plot_luminosity_equation_exact(params.getForProp('prop')['laxis'],
                                                    params.getForEqs('lueqhsse')['xbl'],
                                                    params.getForEqs('lueqhsse')['xbr'],
                                                    params.getForEqs('lueqhsse')['ybu'],
                                                    params.getForEqs('lueqhsse')['ybd'],
                                                    params.getForEqs('lueqhsse')['ilg'])

        # plot hsse luminosity equation exact 2						       
        ranshsselumi.plot_luminosity_equation_exact2(params.getForProp('prop')['laxis'],
                                                     params.getForEqs('lueqhsse')['xbl'],
                                                     params.getForEqs('lueqhsse')['xbr'],
                                                     params.getForEqs('lueqhsse')['ybu'],
                                                     params.getForEqs('lueqhsse')['ybd'],
                                                     params.getForEqs('lueqhsse')['ilg'])

        # plot hsse luminosity equation alternative
        # ranshsselumi.plot_luminosity_equation_2(params.getForProp('prop')['laxis'],
        #                                        params.getForEqs('lueqhsse')['xbl'],
        #                                        params.getForEqs('lueqhsse')['xbr'],
        #                                        params.getForEqs('lueqhsse')['ybu'],
        #                                        params.getForEqs('lueqhsse')['ybd'],
        #                                        params.getForEqs('lueqhsse')['ilg'])

        # plot hsse luminosity equation alternative simplified						       
        # ranshsselumi.plot_luminosity_equation_3(params.getForProp('prop')['laxis'],
        #                                        params.getForEqs('lueqhsse')['xbl'],
        #                                        params.getForEqs('lueqhsse')['xbr'],
        #                                        params.getForEqs('lueqhsse')['ybu'],
        #                                        params.getForEqs('lueqhsse')['ybd'],
        #                                        params.getForEqs('lueqhsse')['ilg'])

    def execHssCompEq(self, inuc, element, x, bconv, tconv):
        params = self.params

        # instantiate 
        ranshssecomp = HsseXtransportEquation(params.getForProp('prop')['eht_data'],
                                                       params.getForProp('prop')['ig'],
                                                       params.getForProp('prop')['fext'],
                                                       inuc, element, bconv, tconv,
                                                       params.getForProp('prop')['intc'],
                                                       params.getForProp('prop')['prefix'])

        ranshssecomp.plot_Xtransport_equation(params.getForProp('prop')['laxis'],
                                              params.getForEqs(x)['xbl'],
                                              params.getForEqs(x)['xbr'],
                                              params.getForEqs(x)['ybu'],
                                              params.getForEqs(x)['ybd'],
                                              params.getForEqs(x)['ilg'])

    def execXrho(self, inuc, element, x, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate 		
        ransXtra = XtransportEquation(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['plabel'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           inuc, element, bconv, tconv, super_ad_i, super_ad_o,
                                           params.getForProp('prop')['intc'],
                                           params.getForProp('prop')['nsdim'],
                                           params.getForProp('prop')['prefix'])

        ransXtra.plot_Xrho(params.getForProp('prop')['laxis'],
                           params.getForEqs(x)['xbl'],
                           params.getForEqs(x)['xbr'],
                           params.getForEqs(x)['ybu'],
                           params.getForEqs(x)['ybd'],
                           params.getForEqs(x)['ilg'])

        # ransXtra.plot_X(params.getForProp('prop')['laxis'], \
        #                params.getForEqs(x)['xbl'], \
        #                params.getForEqs(x)['xbr'], \
        #                params.getForEqs(x)['ybu'], \
        #                params.getForEqs(x)['ybd'], \
        #                params.getForEqs(x)['ilg'])

        # ransXtra.plot_gradX(params.getForProp('prop')['laxis'],\
        #                   params.getForEqs(x)['xbl'],\
        #                   params.getForEqs(x)['xbr'],\
        #                   params.getForEqs(x)['ybu'],\
        #                   params.getForEqs(x)['ybd'],\
        #                   params.getForEqs(x)['ilg'])

    def execX(self, inuc, element, x, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransXtra = XtransportEquation(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['plabel'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           inuc, element, bconv, tconv, super_ad_i, super_ad_o,
                                           params.getForProp('prop')['intc'],
                                           params.getForProp('prop')['nsdim'],
                                           params.getForProp('prop')['prefix'])

        if params.getForProp('prop')['plabel'] == "oburn":

            ransXtra.plot_X_with_MM(params.getForProp('prop')['laxis'],
                                    params.getForEqs(x)['xbl'],
                                    params.getForEqs(x)['xbr'],
                                    params.getForEqs(x)['ybu'],
                                    params.getForEqs(x)['ybd'],
                                    params.getForEqs(x)['ilg'])

        else:
            ransXtra.plot_X(params.getForProp('prop')['laxis'],
                            params.getForEqs(x)['xbl'],
                            params.getForEqs(x)['xbr'],
                            params.getForEqs(x)['ybu'],
                            params.getForEqs(x)['ybd'],
                            params.getForEqs(x)['ilg'])

        #ransXtra.plot_X_space_time(params.getForProp('prop')['laxis'],
        #                           params.getForEqs(x)['xbl'],
        #                           params.getForEqs(x)['xbr'],
        #                           params.getForEqs(x)['ybu'],
        #                           params.getForEqs(x)['ybd'],
        #                           params.getForEqs(x)['ilg'])

        #ransXtra.plot_rhoX_space_time(params.getForProp('prop')['laxis'],
        #                              params.getForEqs(x)['xbl'],
        #                              params.getForEqs(x)['xbr'],
        #                              params.getForEqs(x)['ybu'],
        #                              params.getForEqs(x)['ybd'],
        #                              params.getForEqs(x)['ilg'])

        # ransXtra.plot_Xm_with_MM(params.getForProp('prop')['laxis'],
        #                    params.getForEqs(x)['xbl'],
        #                    params.getForEqs(x)['xbr'],
        #                    params.getForEqs(x)['ybu'],
        #                    params.getForEqs(x)['ybd'],
        #                    params.getForEqs(x)['ilg'])

    def execXtrsEq(self, inuc, element, x, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate 
        ransXtra = XtransportEquation(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['plabel'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           inuc, element, bconv, tconv, super_ad_i, super_ad_o,
                                           params.getForProp('prop')['intc'],
                                           params.getForProp('prop')['nsdim'],
                                           params.getForProp('prop')['prefix'])

        ransXtra.plot_Xtransport_equation(params.getForProp('prop')['laxis'],
                                          params.getForEqs(x)['xbl'],
                                          params.getForEqs(x)['xbr'],
                                          params.getForEqs(x)['ybu'],
                                          params.getForEqs(x)['ybd'],
                                          params.getForEqs(x)['ilg'])

    def execXtrsEqBar(self, inuc, element, x, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate 
        ransXtra = XtransportEquation(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['plabel'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           inuc, element, bconv, tconv, super_ad_i, super_ad_o,
                                           params.getForProp('prop')['intc'],
                                           params.getForProp('prop')['nsdim'],
                                           params.getForProp('prop')['prefix'])

        # plot X transport equation integral budget					       
        ransXtra.plot_Xtransport_equation_integral_budget(params.getForProp('prop')['laxis'],
                                                          params.getForEqsBar(x)['xbl'],
                                                          params.getForEqsBar(x)['xbr'],
                                                          params.getForEqsBar(x)['ybu'],
                                                          params.getForEqsBar(x)['ybd'])

    def execXflxx(self, inuc, element, x, bconv, tconv, tke_diss, tauL, cnvz_in_hp):
        params = self.params

        # instantiate 		
        ransXflxx = XfluxXequation(params.getForProp('prop')['eht_data'],
                                         params.getForProp('prop')['ig'],
                                         params.getForProp('prop')['ieos'],
                                         params.getForProp('prop')['fext'],
                                         inuc, element, bconv, tconv, tke_diss, tauL, cnvz_in_hp,
                                         params.getForProp('prop')['intc'],
                                         params.getForProp('prop')['nsdim'],
                                         params.getForProp('prop')['prefix'])

        # ransXflxx.plot_XfluxX(params.getForProp('prop')['laxis'],
        #                      params.getForEqs(x)['xbl'],
        #                      params.getForEqs(x)['xbr'],
        #                      params.getForEqs(x)['ybu'],
        #                      params.getForEqs(x)['ybd'],
        #                      params.getForEqs(x)['ilg'])

        ransXflxx.plot_alphaX(params.getForProp('prop')['laxis'],
                              params.getForEqs(x)['xbl'],
                              params.getForEqs(x)['xbr'],
                              params.getForEqs(x)['ybu'],
                              params.getForEqs(x)['ybd'],
                              params.getForEqs(x)['ilg'])

        # ransXflxx.plot_XfluxxX(params.getForProp('prop')['laxis'],
        #                      params.getForEqs(x)['xbl'],
        #                      params.getForEqs(x)['xbr'],
        #                      params.getForEqs(x)['ybu'],
        #                      params.getForEqs(x)['ybd'],
        #                      params.getForEqs(x)['ilg'])

        # ransXflxx.plot_XfluxXRogers1989(params.getForProp('prop')['laxis'],
        #                                params.getForEqs(x)['xbl'],
        #                                params.getForEqs(x)['xbr'],
        #                                params.getForEqs(x)['ybu'],
        #                                params.getForEqs(x)['ybd'],
        #                                params.getForEqs(x)['ilg'])

        # ransXflxx.plot_Xflux_gradient(params.getForProp('prop')['laxis'],
        #                              params.getForEqs(x)['xbl'],
        #                              params.getForEqs(x)['xbr'],
        #                              params.getForEqs(x)['ybu'],
        #                              params.getForEqs(x)['ybd'],
        #                              params.getForEqs(x)['ilg'])

        # ransXflxx.plot_XfluxX2(params.getForProp('prop')['laxis'],
        #                      params.getForEqs(x)['xbl'],
        #                      params.getForEqs(x)['xbr'],
        #                      params.getForEqs(x)['ybu'],
        #                      params.getForEqs(x)['ybd'],
        #                      params.getForEqs(x)['ilg'])

    def execXflxXeq(self, inuc, element, x, bconv, tconv, tke_diss, tauL, cnvz_in_hp):
        params = self.params

        # instantiate 
        ransXflxx = XfluxXequation(params.getForProp('prop')['eht_data'],
                                         params.getForProp('prop')['ig'],
                                         params.getForProp('prop')['ieos'],
                                         params.getForProp('prop')['fext'],
                                         inuc, element, bconv, tconv, tke_diss, tauL, cnvz_in_hp,
                                         params.getForProp('prop')['intc'],
                                         params.getForProp('prop')['nsdim'],
                                         params.getForProp('prop')['prefix'])

        ransXflxx.plot_XfluxX_equation(params.getForProp('prop')['laxis'],
                                       params.getForEqs(x)['xbl'],
                                       params.getForEqs(x)['xbr'],
                                       params.getForEqs(x)['ybu'],
                                       params.getForEqs(x)['ybd'],
                                       params.getForEqs(x)['ilg'])

        # ransXflxx.plot_XfluxX_equation2(params.getForProp('prop')['laxis'], \
        #                                params.getForEqs(x)['xbl'], \
        #                                params.getForEqs(x)['xbr'], \
        #                                params.getForEqs(x)['ybu'], \
        #                                params.getForEqs(x)['ybd'], \
        #                                params.getForEqs(x)['ilg'])

    def execXflxy(self, inuc, element, x, bconv, tconv, tke_diss, tauL):
        params = self.params

        # instantiate 		
        ransXflxy = XfluxYequation(params.getForProp('prop')['eht_data'],
                                         params.getForProp('prop')['ig'],
                                         inuc, element, bconv, tconv, tke_diss, tauL,
                                         params.getForProp('prop')['intc'],
                                         params.getForProp('prop')['prefix'])

        ransXflxy.plot_XfluxY(params.getForProp('prop')['laxis'],
                              params.getForEqs(x)['xbl'],
                              params.getForEqs(x)['xbr'],
                              params.getForEqs(x)['ybu'],
                              params.getForEqs(x)['ybd'],
                              params.getForEqs(x)['ilg'])

    def execXflxYeq(self, inuc, element, x, bconv, tconv, tke_diss, tauL):
        params = self.params

        # instantiate 
        ransXflxy = XfluxYequation(params.getForProp('prop')['eht_data'],
                                         params.getForProp('prop')['ig'],
                                         inuc, element, bconv, tconv, tke_diss, tauL,
                                         params.getForProp('prop')['intc'],
                                         params.getForProp('prop')['prefix'])

        ransXflxy.plot_XfluxY_equation(params.getForProp('prop')['laxis'],
                                       params.getForEqs(x)['xbl'],
                                       params.getForEqs(x)['xbr'],
                                       params.getForEqs(x)['ybu'],
                                       params.getForEqs(x)['ybd'],
                                       params.getForEqs(x)['ilg'])

    def execXflxz(self, inuc, element, x, bconv, tconv, tke_diss, tauL):
        params = self.params

        # instantiate 		
        ransXflxz = XfluxZequation(params.getForProp('prop')['eht_data'],
                                         params.getForProp('prop')['ig'],
                                         inuc, element, bconv, tconv, tke_diss, tauL,
                                         params.getForProp('prop')['intc'],
                                         params.getForProp('prop')['prefix'])

        ransXflxz.plot_XfluxZ(params.getForProp('prop')['laxis'],
                              params.getForEqs(x)['xbl'],
                              params.getForEqs(x)['xbr'],
                              params.getForEqs(x)['ybu'],
                              params.getForEqs(x)['ybd'],
                              params.getForEqs(x)['ilg'])

    def execXflxZeq(self, inuc, element, x, bconv, tconv, tke_diss, tauL):
        params = self.params

        # instantiate 
        ransXflxz = XfluxZequation(params.getForProp('prop')['eht_data'],
                                         params.getForProp('prop')['ig'],
                                         inuc, element, bconv, tconv, tke_diss, tauL,
                                         params.getForProp('prop')['intc'],
                                         params.getForProp('prop')['prefix'])

        ransXflxz.plot_XfluxZ_equation(params.getForProp('prop')['laxis'],
                                       params.getForEqs(x)['xbl'],
                                       params.getForEqs(x)['xbr'],
                                       params.getForEqs(x)['ybu'],
                                       params.getForEqs(x)['ybd'],
                                       params.getForEqs(x)['ilg'])

    def execXvar(self, inuc, element, x, bconv, tconv):
        params = self.params
        tauL = 1.

        # instantiate 		
        ransXvar = XvarianceEquation(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          inuc, element, tauL, bconv, tconv,
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['nsdim'],
                                          params.getForProp('prop')['prefix'])

        ransXvar.plot_Xvariance(params.getForProp('prop')['laxis'],
                                params.getForEqs(x)['xbl'],
                                params.getForEqs(x)['xbr'],
                                params.getForEqs(x)['ybu'],
                                params.getForEqs(x)['ybd'],
                                params.getForEqs(x)['ilg'])

    def execXvarEq(self, inuc, element, x, tauL, bconv, tconv):
        params = self.params

        # instantiate 
        ransXvar = XvarianceEquation(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          inuc, element, tauL, bconv, tconv,
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['nsdim'],
                                          params.getForProp('prop')['prefix'])

        ransXvar.plot_Xvariance_equation(params.getForProp('prop')['laxis'],
                                         params.getForEqs(x)['xbl'],
                                         params.getForEqs(x)['xbr'],
                                         params.getForEqs(x)['ybu'],
                                         params.getForEqs(x)['ybd'],
                                         params.getForEqs(x)['ilg'])

    def execDiff(self, inuc, element, x, lc, uconv, bconv, tconv, tke_diss, tauL, super_ad_i, super_ad_o, cnvz_in_hp):
        params = self.params

        # instantiate 
        ransXdiff = Xdiffusivity(params.getForProp('prop')['eht_data'],
                                       params.getForProp('prop')['ig'],
                                       params.getForProp('prop')['fext'],
                                       params.getForProp('prop')['ieos'],
                                       inuc, element, lc, uconv, bconv, tconv, cnvz_in_hp,
                                       tke_diss, tauL, super_ad_i, super_ad_o,
                                       params.getForProp('prop')['intc'],
                                       params.getForProp('prop')['prefix'])

        # ransXdiff.plot_X_Ediffusivity(params.getForProp('prop')['laxis'],
        #                              params.getForEqs(x)['xbl'],
        #                              params.getForEqs(x)['xbr'],
        #                              params.getForEqs(x)['ybu'],
        #                              params.getForEqs(x)['ybd'],
        #                              params.getForEqs(x)['ilg'])

        ransXdiff.plot_X_Ediffusivity2(params.getForProp('prop')['laxis'],
                                       params.getForEqs(x)['xbl'],
                                       params.getForEqs(x)['xbr'],
                                       params.getForEqs(x)['ybu'],
                                       params.getForEqs(x)['ybd'],
                                       params.getForEqs(x)['ilg'])

    def execXda(self, inuc, element, x, bconv, tconv):
        params = self.params

        # instantiate 
        ransXda = XdamkohlerNumber(params.getForProp('prop')['eht_data'],
                                       params.getForProp('prop')['ig'],
                                       inuc, element, bconv, tconv,
                                       params.getForProp('prop')['intc'],
                                       params.getForProp('prop')['prefix'])

        ransXda.plot_Xda(params.getForProp('prop')['laxis'],
                         params.getForEqs(x)['xbl'],
                         params.getForEqs(x)['xbr'],
                         params.getForEqs(x)['ybu'],
                         params.getForEqs(x)['ybd'],
                         params.getForEqs(x)['ilg'])

    def execTke(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate 		
        ransTke = TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['intc'],
                                                     params.getForProp('prop')['nsdim'],
                                                     kolmdissrate, bconv, tconv,
                                                     super_ad_i, super_ad_o,
                                                     params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy			   
        ransTke.plot_tke(params.getForProp('prop')['laxis'],
                         bconv, tconv,
                         params.getForEqs('tkie')['xbl'],
                         params.getForEqs('tkie')['xbr'],
                         params.getForEqs('tkie')['ybu'],
                         params.getForEqs('tkie')['ybd'],
                         params.getForEqs('tkie')['ilg'])

        #ransTke.plot_TKE_space_time(params.getForProp('prop')['laxis'],
        #                            params.getForEqs('tkeeq')['xbl'],
        #                            params.getForEqs('tkeeq')['xbr'],
        #                            params.getForEqs('tkeeq')['ybu'],
        #                            params.getForEqs('tkeeq')['ybd'],
        #                            params.getForEqs('tkeeq')['ilg'])

        # plot turbulent kinetic energy evolution	   
        # ransTke.plot_tke_evolution()

        # plot evolution of convection boundaries	   
        # ransTke.plot_conv_bndry_location()

    def execTkeEq(self, wxStudio, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate 		
        ransTke = TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['intc'],
                                                     params.getForProp('prop')['nsdim'],
                                                     kolmdissrate, bconv, tconv,
                                                     super_ad_i, super_ad_o,
                                                     params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy equation
        if wxStudio[0]:
            ransTke.plot_momentum_equation_x(wxStudio[0], params.getForProp('prop')['laxis'],
                                              wxStudio[1],
                                              wxStudio[2],
                                              wxStudio[3],
                                              wxStudio[4],
                                              params.getForEqs('conteq')['ilg'])
        else:
            ransTke.plot_tke_equation(wxStudio[0], params.getForProp('prop')['laxis'],
                                      params.getForEqs('tkeeq')['xbl'],
                                      params.getForEqs('tkeeq')['xbr'],
                                      params.getForEqs('tkeeq')['ybu'],
                                      params.getForEqs('tkeeq')['ybd'],
                                      params.getForEqs('tkeeq')['ilg'])


    def execTkeEqBar(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransTke = TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['intc'],
                                                     params.getForProp('prop')['nsdim'],
                                                     kolmdissrate, bconv, tconv,
                                                     super_ad_i, super_ad_o,
                                                     params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy equation
        ransTke.plot_tke_equation_integral_budget(params.getForProp('prop')['laxis'],
                                                  params.getForEqs('tkeeqBar')['xbl'],
                                                  params.getForEqs('tkeeqBar')['xbr'],
                                                  params.getForEqs('tkeeqBar')['ybu'],
                                                  params.getForEqs('tkeeqBar')['ybd'])

    def execTkeRadial(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransTkeR = TurbulentKineticEnergyEquationRadial(params.getForProp('prop')['eht_data'],
                                                             params.getForProp('prop')['ig'],
                                                             params.getForProp('prop')['intc'],
                                                             params.getForProp('prop')['nsdim'],
                                                             kolmdissrate, bconv, tconv,
                                                             super_ad_i, super_ad_o,
                                                             params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy
        ransTkeR.plot_tkeRadial(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('tkieR')['xbl'],
                                params.getForEqs('tkieR')['xbr'],
                                params.getForEqs('tkieR')['ybu'],
                                params.getForEqs('tkieR')['ybd'],
                                params.getForEqs('tkieR')['ilg'])

        #ransTkeR.plot_TKEradial_space_time(params.getForProp('prop')['laxis'],
        #                                   params.getForEqs('tkeReq')['xbl'],
        #                                   params.getForEqs('tkeReq')['xbr'],
        #                                   params.getForEqs('tkeReq')['ybu'],
        #                                   params.getForEqs('tkeReq')['ybd'],
        #                                   params.getForEqs('tkeReq')['ilg'])

        # plot turbulent kinetic energy evolution
        # ransTke.plot_tke_evolution()

        # plot evolution of convection boundaries
        # ransTke.plot_conv_bndry_location()

    def execTkeEqRadial(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransTkeR = TurbulentKineticEnergyEquationRadial(params.getForProp('prop')['eht_data'],
                                                             params.getForProp('prop')['ig'],
                                                             params.getForProp('prop')['intc'],
                                                             params.getForProp('prop')['nsdim'],
                                                             kolmdissrate, bconv, tconv,
                                                             super_ad_i, super_ad_o,
                                                             params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy equation
        ransTkeR.plot_tkeRadial_equation(params.getForProp('prop')['laxis'],
                                         params.getForEqs('tkeReq')['xbl'],
                                         params.getForEqs('tkeReq')['xbr'],
                                         params.getForEqs('tkeReq')['ybu'],
                                         params.getForEqs('tkeReq')['ybd'],
                                         params.getForEqs('tkeReq')['ilg'])

    def execTkeEqRadialBar(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransTkeR = TurbulentKineticEnergyEquationRadial(params.getForProp('prop')['eht_data'],
                                                             params.getForProp('prop')['ig'],
                                                             params.getForProp('prop')['intc'],
                                                             params.getForProp('prop')['nsdim'],
                                                             kolmdissrate, bconv, tconv,
                                                             super_ad_i, super_ad_o,
                                                             params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy equation
        ransTkeR.plot_tkeRadial_equation_integral_budget(params.getForProp('prop')['laxis'],
                                                         params.getForEqs('tkeReqBar')['xbl'],
                                                         params.getForEqs('tkeReqBar')['xbr'],
                                                         params.getForEqs('tkeReqBar')['ybu'],
                                                         params.getForEqs('tkeReqBar')['ybd'])

    def execTkeHorizontal(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransTkeH = TurbulentKineticEnergyEquationHorizontal(params.getForProp('prop')['eht_data'],
                                                                 params.getForProp('prop')['ig'],
                                                                 params.getForProp('prop')['intc'],
                                                                 params.getForProp('prop')['nsdim'],
                                                                 kolmdissrate, bconv, tconv,
                                                                 super_ad_i, super_ad_o,
                                                                 params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy
        ransTkeH.plot_tkeHorizontal(params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('tkieH')['xbl'],
                                    params.getForEqs('tkieH')['xbr'],
                                    params.getForEqs('tkieH')['ybu'],
                                    params.getForEqs('tkieH')['ybd'],
                                    params.getForEqs('tkieH')['ilg'])

        #ransTkeH.plot_TKEhorizontal_space_time(params.getForProp('prop')['laxis'],
        #                                       params.getForEqs('tkeHeq')['xbl'],
        #                                       params.getForEqs('tkeHeq')['xbr'],
        #                                       params.getForEqs('tkeHeq')['ybu'],
        #                                       params.getForEqs('tkeHeq')['ybd'],
        #                                       params.getForEqs('tkeHeq')['ilg'])

        # plot turbulent kinetic energy evolution
        # ransTke.plot_tke_evolution()

        # plot evolution of convection boundaries
        # ransTke.plot_conv_bndry_location()

    def execTkeEqHorizontal(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransTkeH = TurbulentKineticEnergyEquationHorizontal(params.getForProp('prop')['eht_data'],
                                                                 params.getForProp('prop')['ig'],
                                                                 params.getForProp('prop')['intc'],
                                                                 params.getForProp('prop')['nsdim'],
                                                                 kolmdissrate, bconv, tconv,
                                                                 super_ad_i, super_ad_o,
                                                                 params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy equation
        ransTkeH.plot_tkeHorizontal_equation(params.getForProp('prop')['laxis'],
                                             params.getForEqs('tkeHeq')['xbl'],
                                             params.getForEqs('tkeHeq')['xbr'],
                                             params.getForEqs('tkeHeq')['ybu'],
                                             params.getForEqs('tkeHeq')['ybd'],
                                             params.getForEqs('tkeHeq')['ilg'])

    def execTkeEqHorizontalBar(self, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate
        ransTkeH = TurbulentKineticEnergyEquationHorizontal(params.getForProp('prop')['eht_data'],
                                                                 params.getForProp('prop')['ig'],
                                                                 params.getForProp('prop')['intc'],
                                                                 params.getForProp('prop')['nsdim'],
                                                                 kolmdissrate, bconv, tconv,
                                                                 super_ad_i, super_ad_o,
                                                                 params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy equation
        ransTkeH.plot_tkeHorizontal_equation_integral_budget(params.getForProp('prop')['laxis'],
                                                             params.getForEqs('tkeHeqBar')['xbl'],
                                                             params.getForEqs('tkeHeqBar')['xbr'],
                                                             params.getForEqs('tkeHeqBar')['ybu'],
                                                             params.getForEqs('tkeHeqBar')['ybd'])

    def execMomx(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransMomx = MomentumEquationX(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['nsdim'],
                                          params.getForProp('prop')['prefix'])

        ransMomx.plot_momentum_x(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('momex')['xbl'],
                                 params.getForEqs('momex')['xbr'],
                                 params.getForEqs('momex')['ybu'],
                                 params.getForEqs('momex')['ybd'],
                                 params.getForEqs('momex')['ilg'])

    def execMomxEq(self, wxStudio, bconv, tconv):
        params = self.params

        # instantiate 		
        ransMomx = MomentumEquationX(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['nsdim'],
                                          params.getForProp('prop')['prefix'])

        if wxStudio[0]:
            ransMomx.plot_momentum_equation_x(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              wxStudio[1],
                                              wxStudio[2],
                                              wxStudio[3],
                                              wxStudio[4],
                                              params.getForEqs('conteq')['ilg'])
        else:
            ransMomx.plot_momentum_equation_x(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              params.getForEqs('momxeq')['xbl'],
                                              params.getForEqs('momxeq')['xbr'],
                                              params.getForEqs('momxeq')['ybu'],
                                              params.getForEqs('momxeq')['ybd'],
                                              params.getForEqs('momxeq')['ilg'])

    def execMomy(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransMomy = MomentumEquationY(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomy.plot_momentum_y(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('momey')['xbl'],
                                 params.getForEqs('momey')['xbr'],
                                 params.getForEqs('momey')['ybu'],
                                 params.getForEqs('momey')['ybd'],
                                 params.getForEqs('momey')['ilg'])

    def execMomyEq(self, wxStudio, bconv, tconv):
        params = self.params

        # instantiate 		
        ransMomy = MomentumEquationY(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        if wxStudio[0]:
            ransMomy.plot_momentum_equation_y(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              wxStudio[1],
                                              wxStudio[2],
                                              wxStudio[3],
                                              wxStudio[4],
                                              params.getForEqs('conteq')['ilg'])
        else:
            ransMomy.plot_momentum_equation_y(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              params.getForEqs('momyeq')['xbl'],
                                              params.getForEqs('momyeq')['xbr'],
                                              params.getForEqs('momyeq')['ybu'],
                                              params.getForEqs('momyeq')['ybd'],
                                              params.getForEqs('momyeq')['ilg'])

    def execMomz(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransMomz = MomentumEquationZ(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomz.plot_momentum_z(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('momez')['xbl'],
                                 params.getForEqs('momez')['xbr'],
                                 params.getForEqs('momez')['ybu'],
                                 params.getForEqs('momez')['ybd'],
                                 params.getForEqs('momez')['ilg'])

    def execMomzEq(self, wxStudio, bconv, tconv):
        params = self.params

        # instantiate 		
        ransMomz = MomentumEquationZ(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        if wxStudio[0]:
            ransMomz.plot_momentum_equation_z(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              wxStudio[1],
                                              wxStudio[2],
                                              wxStudio[3],
                                              wxStudio[4],
                                              params.getForEqs('conteq')['ilg'])
        else:
            ransMomz.plot_momentum_equation_z(wxStudio[0], params.getForProp('prop')['laxis'],
                                              bconv, tconv,
                                              params.getForEqs('momzeq')['xbl'],
                                              params.getForEqs('momzeq')['xbr'],
                                              params.getForEqs('momzeq')['ybu'],
                                              params.getForEqs('momzeq')['ybd'],
                                              params.getForEqs('momzeq')['ilg'])

    def execEi(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransEi = InternalEnergyEquation(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           params.getForProp('prop')['intc'],
                                           tke_diss,
                                           params.getForProp('prop')['prefix'])

        ransEi.plot_ei(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('eint')['xbl'],
                       params.getForEqs('eint')['xbr'],
                       params.getForEqs('eint')['ybu'],
                       params.getForEqs('eint')['ybd'],
                       params.getForEqs('eint')['ilg'])

    def execEiEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransEi = InternalEnergyEquation(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           params.getForProp('prop')['intc'],
                                           tke_diss,
                                           params.getForProp('prop')['prefix'])

        ransEi.plot_ei_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('eieq')['xbl'],
                                params.getForEqs('eieq')['xbr'],
                                params.getForEqs('eieq')['ybu'],
                                params.getForEqs('eieq')['ybd'],
                                params.getForEqs('eieq')['ilg'])

    def execEiFlx(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransEiFlx = InternalEnergyFluxEquation(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    tke_diss,
                                                    params.getForProp('prop')['prefix'])

        ransEiFlx.plot_fei(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('eintflx')['xbl'],
                           params.getForEqs('eintflx')['xbr'],
                           params.getForEqs('eintflx')['ybu'],
                           params.getForEqs('eintflx')['ybd'],
                           params.getForEqs('eintflx')['ilg'])

    def execEiFlxEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransEiFlx = InternalEnergyFluxEquation(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    tke_diss,
                                                    params.getForProp('prop')['prefix'])

        ransEiFlx.plot_fei_equation(params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('eiflxeq')['xbl'],
                                    params.getForEqs('eiflxeq')['xbr'],
                                    params.getForEqs('eiflxeq')['ybu'],
                                    params.getForEqs('eiflxeq')['ybd'],
                                    params.getForEqs('eiflxeq')['ilg'])

        ransEiFlx.plot_fei_equation2(params.getForProp('prop')['laxis'],
                                     bconv, tconv,
                                     params.getForEqs('eiflxeq')['xbl'],
                                     params.getForEqs('eiflxeq')['xbr'],
                                     params.getForEqs('eiflxeq')['ybu'],
                                     params.getForEqs('eiflxeq')['ybd'],
                                     params.getForEqs('eiflxeq')['ilg'])

    def execHHflx(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransHHflx = EnthalpyFluxEquation(params.getForProp('prop')['eht_data'],
                                              params.getForProp('prop')['ig'],
                                              params.getForProp('prop')['ieos'],
                                              params.getForProp('prop')['intc'],
                                              tke_diss,
                                              params.getForProp('prop')['prefix'])

        ransHHflx.plot_fhh(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('enthflx')['xbl'],
                           params.getForEqs('enthflx')['xbr'],
                           params.getForEqs('enthflx')['ybu'],
                           params.getForEqs('enthflx')['ybd'],
                           params.getForEqs('enthflx')['ilg'])

    def execHHflxEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransHHflx = EnthalpyFluxEquation(params.getForProp('prop')['eht_data'],
                                              params.getForProp('prop')['ig'],
                                              params.getForProp('prop')['ieos'],
                                              params.getForProp('prop')['intc'],
                                              tke_diss,
                                              params.getForProp('prop')['prefix'])

        ransHHflx.plot_fhh_equation(params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('hhflxeq')['xbl'],
                                    params.getForEqs('hhflxeq')['xbr'],
                                    params.getForEqs('hhflxeq')['ybu'],
                                    params.getForEqs('hhflxeq')['ybd'],
                                    params.getForEqs('hhflxeq')['ilg'])

    def execHHvar(self, bconv, tconv):
        params = self.params
        tke_diss = 0.
        tauL = 1.

        # instantiate 		
        ransHHvar = EnthalpyVarianceEquation(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['ieos'],
                                                     params.getForProp('prop')['intc'],
                                                     tke_diss, tauL,
                                                     params.getForProp('prop')['prefix'])

        ransHHvar.plot_sigma_hh(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('enthvar')['xbl'],
                                params.getForEqs('enthvar')['xbr'],
                                params.getForEqs('enthvar')['ybu'],
                                params.getForEqs('enthvar')['ybd'],
                                params.getForEqs('enthvar')['ilg'])

    def execHHvarEq(self, tke_diss, tauL, bconv, tconv):
        params = self.params

        # instantiate 		
        ransHHvar = EnthalpyVarianceEquation(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['ieos'],
                                                     params.getForProp('prop')['intc'],
                                                     tke_diss, tauL,
                                                     params.getForProp('prop')['prefix'])

        ransHHvar.plot_sigma_hh_equation(params.getForProp('prop')['laxis'],
                                         bconv, tconv,
                                         params.getForEqs('hhvareq')['xbl'],
                                         params.getForEqs('hhvareq')['xbr'],
                                         params.getForEqs('hhvareq')['ybu'],
                                         params.getForEqs('hhvareq')['ybd'],
                                         params.getForEqs('hhvareq')['ilg'])

    def execEiVar(self, bconv, tconv):
        params = self.params
        tke_diss = 0.
        tauL = 1.

        # instantiate 		
        ransEiVar = InternalEnergyVarianceEquation(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['ieos'],
                                                           params.getForProp('prop')['intc'],
                                                           tke_diss, tauL,
                                                           params.getForProp('prop')['prefix'])

        ransEiVar.plot_sigma_ei(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('eintvar')['xbl'],
                                params.getForEqs('eintvar')['xbr'],
                                params.getForEqs('eintvar')['ybu'],
                                params.getForEqs('eintvar')['ybd'],
                                params.getForEqs('eintvar')['ilg'])

    def execEiVarEq(self, tke_diss, tauL, bconv, tconv):
        params = self.params

        # instantiate 		
        ransEiVar = InternalEnergyVarianceEquation(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['ieos'],
                                                           params.getForProp('prop')['intc'],
                                                           tke_diss, tauL,
                                                           params.getForProp('prop')['prefix'])

        ransEiVar.plot_sigma_ei_equation(params.getForProp('prop')['laxis'],
                                         bconv, tconv,
                                         params.getForEqs('eivareq')['xbl'],
                                         params.getForEqs('eivareq')['xbr'],
                                         params.getForEqs('eivareq')['ybu'],
                                         params.getForEqs('eivareq')['ybd'],
                                         params.getForEqs('eivareq')['ilg'])

    def execSS(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransSS = EntropyEquation(params.getForProp('prop')['eht_data'],
                                    params.getForProp('prop')['ig'],
                                    params.getForProp('prop')['fext'],
                                    params.getForProp('prop')['intc'],
                                    params.getForProp('prop')['nsdim'],
                                    tke_diss,
                                    params.getForProp('prop')['prefix'])

        ransSS.plot_ss(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('entr')['xbl'],
                       params.getForEqs('entr')['xbr'],
                       params.getForEqs('entr')['ybu'],
                       params.getForEqs('entr')['ybd'],
                       params.getForEqs('entr')['ilg'])

    def execSSeq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransSS = EntropyEquation(params.getForProp('prop')['eht_data'],
                                    params.getForProp('prop')['ig'],
                                    params.getForProp('prop')['fext'],
                                    params.getForProp('prop')['intc'],
                                    params.getForProp('prop')['nsdim'],
                                    tke_diss,
                                    params.getForProp('prop')['prefix'])

        ransSS.plot_ss_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('sseq')['xbl'],
                                params.getForEqs('sseq')['xbr'],
                                params.getForEqs('sseq')['ybu'],
                                params.getForEqs('sseq')['ybd'],
                                params.getForEqs('sseq')['ilg'])

    def execSSflx(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransSSflx = EntropyFluxEquation(params.getForProp('prop')['eht_data'],
                                             params.getForProp('prop')['ig'],
                                             params.getForProp('prop')['intc'],
                                             tke_diss,
                                             params.getForProp('prop')['prefix'])

        ransSSflx.plot_fss(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('entrflx')['xbl'],
                           params.getForEqs('entrflx')['xbr'],
                           params.getForEqs('entrflx')['ybu'],
                           params.getForEqs('entrflx')['ybd'],
                           params.getForEqs('entrflx')['ilg'])

    def execSSflxEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransSSflx = EntropyFluxEquation(params.getForProp('prop')['eht_data'],
                                             params.getForProp('prop')['ig'],
                                             params.getForProp('prop')['intc'],
                                             tke_diss,
                                             params.getForProp('prop')['prefix'])

        ransSSflx.plot_fss_equation(params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('ssflxeq')['xbl'],
                                    params.getForEqs('ssflxeq')['xbr'],
                                    params.getForEqs('ssflxeq')['ybu'],
                                    params.getForEqs('ssflxeq')['ybd'],
                                    params.getForEqs('ssflxeq')['ilg'])

        ransSSflx.plot_fss_equation2(params.getForProp('prop')['laxis'],
                                     bconv, tconv,
                                     params.getForEqs('ssflxeq')['xbl'],
                                     params.getForEqs('ssflxeq')['xbr'],
                                     params.getForEqs('ssflxeq')['ybu'],
                                     params.getForEqs('ssflxeq')['ybd'],
                                     params.getForEqs('ssflxeq')['ilg'])

    def execSSvar(self, bconv, tconv):
        params = self.params
        tke_diss = 0.
        tauL = 1.

        # instantiate 		
        ransSSvar = EntropyVarianceEquation(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['nsdim'],
                                                    tke_diss, tauL,
                                                    params.getForProp('prop')['prefix'])

        ransSSvar.plot_sigma_ss(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('entrvar')['xbl'],
                                params.getForEqs('entrvar')['xbr'],
                                params.getForEqs('entrvar')['ybu'],
                                params.getForEqs('entrvar')['ybd'],
                                params.getForEqs('entrvar')['ilg'])

    def execSSvarEq(self, tke_diss, tauL, bconv, tconv):
        params = self.params

        # instantiate 		
        ransSSvar = EntropyVarianceEquation(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['nsdim'],
                                                    tke_diss, tauL,
                                                    params.getForProp('prop')['prefix'])

        ransSSvar.plot_sigma_ss_equation(params.getForProp('prop')['laxis'],
                                         bconv, tconv,
                                         params.getForEqs('ssvareq')['xbl'],
                                         params.getForEqs('ssvareq')['xbr'],
                                         params.getForEqs('ssvareq')['ybu'],
                                         params.getForEqs('ssvareq')['ybd'],
                                         params.getForEqs('ssvareq')['ilg'])

    def execDDvar(self, bconv, tconv):
        params = self.params
        tauL = 1.

        # instantiate 		
        ransDDvar = DensityVarianceEquation(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    tauL,
                                                    params.getForProp('prop')['prefix'])

        ransDDvar.plot_sigma_dd(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('densvar')['xbl'],
                                params.getForEqs('densvar')['xbr'],
                                params.getForEqs('densvar')['ybu'],
                                params.getForEqs('densvar')['ybd'],
                                params.getForEqs('densvar')['ilg'])

    def execDDvarEq(self, tauL, bconv, tconv):
        params = self.params

        # instantiate 		
        ransSSvar = DensityVarianceEquation(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    tauL,
                                                    params.getForProp('prop')['prefix'])

        ransSSvar.plot_sigma_dd_equation(params.getForProp('prop')['laxis'],
                                         bconv, tconv,
                                         params.getForEqs('ddvareq')['xbl'],
                                         params.getForEqs('ddvareq')['xbr'],
                                         params.getForEqs('ddvareq')['ybu'],
                                         params.getForEqs('ddvareq')['ybd'],
                                         params.getForEqs('ddvareq')['ilg'])

    def execTMSflx(self, bconv, tconv, lc):
        params = self.params

        # instantiate 		
        ransTMSflx = TurbulentMassFluxEquation(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['intc'],
                                                 params.getForProp('prop')['prefix'],
                                                 lc)

        ransTMSflx.plot_a(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('tmsflx')['xbl'],
                          params.getForEqs('tmsflx')['xbr'],
                          params.getForEqs('tmsflx')['ybu'],
                          params.getForEqs('tmsflx')['ybd'],
                          params.getForEqs('tmsflx')['ilg'])

    def execAeq(self, bconv, tconv, lc):
        params = self.params

        # instantiate 		
        ransTMSflx = TurbulentMassFluxEquation(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['intc'],
                                                 params.getForProp('prop')['prefix'],
                                                 lc)

        ransTMSflx.plot_a_equation(params.getForProp('prop')['laxis'],
                                   bconv, tconv,
                                   params.getForEqs('aeq')['xbl'],
                                   params.getForEqs('aeq')['xbr'],
                                   params.getForEqs('aeq')['ybu'],
                                   params.getForEqs('aeq')['ybd'],
                                   params.getForEqs('aeq')['ilg'])

    def execDSVC(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransDSVC = DensitySpecificVolumeCovarianceEquation(params.getForProp('prop')['eht_data'],
                                                             params.getForProp('prop')['ig'],
                                                             params.getForProp('prop')['intc'],
                                                             params.getForProp('prop')['prefix'])

        ransDSVC.plot_b(params.getForProp('prop')['laxis'],
                        bconv, tconv,
                        params.getForEqs('dsvc')['xbl'],
                        params.getForEqs('dsvc')['xbr'],
                        params.getForEqs('dsvc')['ybu'],
                        params.getForEqs('dsvc')['ybd'],
                        params.getForEqs('dsvc')['ilg'])

    def execBeq(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransDSVC = DensitySpecificVolumeCovarianceEquation(params.getForProp('prop')['eht_data'],
                                                             params.getForProp('prop')['ig'],
                                                             params.getForProp('prop')['intc'],
                                                             params.getForProp('prop')['prefix'])

        ransDSVC.plot_b_equation(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('beq')['xbl'],
                                 params.getForEqs('beq')['xbr'],
                                 params.getForEqs('beq')['ybu'],
                                 params.getForEqs('beq')['ybd'],
                                 params.getForEqs('beq')['ilg'])

    def execRhoTemp(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransTempRho = TemperatureDensity(params.getForProp('prop')['eht_data'],
                                              params.getForProp('prop')['ig'],
                                              params.getForProp('prop')['intc'],
                                              params.getForProp('prop')['prefix'])

        ransTempRho.plot_ttdd(params.getForProp('prop')['laxis'],
                              bconv, tconv,
                              params.getForEqs('ttdd')['xbl'],
                              params.getForEqs('ttdd')['xbr'],
                              params.getForEqs('ttdd')['ybu'],
                              params.getForEqs('ttdd')['ybd'],
                              params.getForEqs('ttdd')['ilg'])

    def execPressEi(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransPressEi = PressureInternalEnergy(params.getForProp('prop')['eht_data'],
                                                  params.getForProp('prop')['ig'],
                                                  params.getForProp('prop')['intc'],
                                                  params.getForProp('prop')['prefix'])

        ransPressEi.plot_ppei(params.getForProp('prop')['laxis'],
                              bconv, tconv,
                              params.getForEqs('ppei')['xbl'],
                              params.getForEqs('ppei')['xbr'],
                              params.getForEqs('ppei')['ybu'],
                              params.getForEqs('ppei')['ybd'],
                              params.getForEqs('ppei')['ilg'])

    def execEnuc(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransEnuc = NuclearEnergyProduction(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                params.getForProp('prop')['intc'],
                                                params.getForProp('prop')['prefix'])

        # ransEnuc.plot_enuc(params.getForProp('prop')['laxis'],
        #                   bconv, tconv,
        #                   params.getForEqs('enuc')['xbl'],
        #                   params.getForEqs('enuc')['xbr'],
        #                   params.getForEqs('enuc')['ybu'],
        #                   params.getForEqs('enuc')['ybd'],
        #                   params.getForEqs('enuc')['ilg'])

        ransEnuc.plot_enuc2(params.getForProp('prop')['laxis'],
                            bconv, tconv,
                            params.getForEqs('enuc')['xbl'],
                            params.getForEqs('enuc')['xbr'],
                            params.getForEqs('enuc')['ybu'],
                            params.getForEqs('enuc')['ybd'],
                            params.getForEqs('enuc')['ilg'])

        # ransEnuc.plot_enuc_per_volume(params.getForProp('prop')['laxis'], \
        #                              params.getForEqs('enuc')['xbl'], \
        #                              params.getForEqs('enuc')['xbr'], \
        #                              params.getForEqs('enuc')['ybu'], \
        #                              params.getForEqs('enuc')['ybd'], \
        #                              params.getForEqs('enuc')['ilg'])

    def execGrav(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransGrav = Gravity(params.getForProp('prop')['eht_data'],
                                params.getForProp('prop')['ig'],
                                params.getForProp('prop')['intc'],
                                params.getForProp('prop')['prefix'])

        ransGrav.plot_grav(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('grav')['xbl'],
                           params.getForEqs('grav')['xbr'],
                           params.getForEqs('grav')['ybu'],
                           params.getForEqs('grav')['ybd'],
                           params.getForEqs('grav')['ilg'])

    def execNablas(self, bconv, tconv, super_ad_i, super_ad_o):
        params = self.params

        # instantiate 		
        ransNablas = TemperatureGradients(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['fext'],
                                                 params.getForProp('prop')['ieos'],
                                                 params.getForProp('prop')['intc'],
                                                 params.getForProp('prop')['prefix'])

        ransNablas.plot_nablas(params.getForProp('prop')['laxis'],
                               bconv, tconv, super_ad_i, super_ad_o,
                               params.getForEqs('nablas')['xbl'],
                               params.getForEqs('nablas')['xbr'],
                               params.getForEqs('nablas')['ybu'],
                               params.getForEqs('nablas')['ybd'],
                               params.getForEqs('nablas')['ilg'])

        #ransNablas.plot_nablas2(params.getForProp('prop')['laxis'],
        #                        bconv, tconv, super_ad_i, super_ad_o,
        #                        params.getForEqs('nablas')['xbl'],
        #                        params.getForEqs('nablas')['xbr'],
        #                        params.getForEqs('nablas')['ybu'],
        #                        params.getForEqs('nablas')['ybd'],
        #                        params.getForEqs('nablas')['ilg'])

    def execDegeneracy(self):
        params = self.params

        # instantiate 		
        ransDeg = Degeneracy(params.getForProp('prop')['eht_data'],
                                 params.getForProp('prop')['ig'],
                                 params.getForProp('prop')['intc'],
                                 params.getForProp('prop')['prefix'])

        ransDeg.plot_degeneracy(params.getForProp('prop')['laxis'],
                                params.getForEqs('psi')['xbl'],
                                params.getForEqs('psi')['xbr'],
                                params.getForEqs('psi')['ybu'],
                                params.getForEqs('psi')['ybd'],
                                params.getForEqs('psi')['ilg'])

    def execVelocitiesMeanExp(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransVelmeanExp = VelocitiesMeanExp(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['fext'],
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['nsdim'],
                                                      params.getForProp('prop')['prefix'])

        ransVelmeanExp.plot_velocities(params.getForProp('prop')['laxis'],
                                       bconv, tconv,
                                       params.getForEqs('velbgr')['xbl'],
                                       params.getForEqs('velbgr')['xbr'],
                                       params.getForEqs('velbgr')['ybu'],
                                       params.getForEqs('velbgr')['ybd'],
                                       params.getForEqs('velbgr')['ilg'])

    def execVelocitiesMLTturb(self, bconv, tconv, uconv, super_ad_i, super_ad_o, ):
        params = self.params

        # instantiate 		
        ransVelMLTturb = VelocitiesMLTturb(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['fext'],
                                                      params.getForProp('prop')['ieos'],
                                                      bconv, tconv, uconv, super_ad_i, super_ad_o,
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['nsdim'],
                                                      params.getForProp('prop')['prefix'])

        ransVelMLTturb.plot_velocities(params.getForProp('prop')['laxis'],
                                       params.getForEqs('velmlt')['xbl'],
                                       params.getForEqs('velmlt')['xbr'],
                                       params.getForEqs('velmlt')['ybu'],
                                       params.getForEqs('velmlt')['ybd'],
                                       params.getForEqs('velmlt')['ilg'])

    def execBruntV(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransBruntV = BruntVaisalla(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['ieos'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransBruntV.plot_bruntvaisalla(params.getForProp('prop')['laxis'],
                                      bconv, tconv,
                                      params.getForEqs('nsq')['xbl'],
                                      params.getForEqs('nsq')['xbr'],
                                      params.getForEqs('nsq')['ybu'],
                                      params.getForEqs('nsq')['ybd'],
                                      params.getForEqs('nsq')['ilg'])

        # ransBruntV.plot_ri(params.getForProp('prop')['laxis'],
        #                              bconv, tconv,
        #                              params.getForEqs('nsq')['xbl'],
        #                              params.getForEqs('nsq')['xbr'],
        #                              params.getForEqs('nsq')['ybu'],
        #                              params.getForEqs('nsq')['ybd'],
        #                              params.getForEqs('nsq')['ilg'])

    def execBuoyancy(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransBuo = Buoyancy(params.getForProp('prop')['eht_data'],
                               params.getForProp('prop')['ig'],
                               params.getForProp('prop')['ieos'],
                               params.getForProp('prop')['intc'],
                               params.getForProp('prop')['prefix'])

        ransBuo.plot_buoyancy(params.getForProp('prop')['laxis'],
                              bconv, tconv,
                              params.getForEqs('buo')['xbl'],
                              params.getForEqs('buo')['xbr'],
                              params.getForEqs('buo')['ybu'],
                              params.getForEqs('buo')['ybd'],
                              params.getForEqs('buo')['ilg'])

    def execRelativeRmsFlct(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransRms = RelativeRMSflct(params.getForProp('prop')['eht_data'],
                                      params.getForProp('prop')['ig'],
                                      params.getForProp('prop')['ieos'],
                                      params.getForProp('prop')['intc'],
                                      params.getForProp('prop')['nsdim'],
                                      params.getForProp('prop')['prefix'])

        ransRms.plot_relative_rms_flct(params.getForProp('prop')['laxis'],
                                       bconv, tconv,
                                       params.getForEqs('relrmsflct')['xbl'],
                                       params.getForEqs('relrmsflct')['xbr'],
                                       params.getForEqs('relrmsflct')['ybu'],
                                       params.getForEqs('relrmsflct')['ybd'],
                                       params.getForEqs('relrmsflct')['ilg'])

        # ransRms.plot_relative_rms_flct2(params.getForProp('prop')['laxis'],
        #                               bconv, tconv,
        #                               params.getForEqs('relrmsflct')['xbl'],
        #                               params.getForEqs('relrmsflct')['xbr'],
        #                               params.getForEqs('relrmsflct')['ybu'],
        #                               params.getForEqs('relrmsflct')['ybd'],
        #                               params.getForEqs('relrmsflct')['ilg'])

    def execAbarZbar(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransAZ = AbarZbar(params.getForProp('prop')['eht_data'],
                                   params.getForProp('prop')['ig'],
                                   params.getForProp('prop')['intc'],
                                   params.getForProp('prop')['prefix'])

        ransAZ.plot_abarzbar(params.getForProp('prop')['laxis'],
                             bconv, tconv,
                             params.getForEqs('abzb')['xbl'],
                             params.getForEqs('abzb')['xbr'],
                             params.getForEqs('abzb')['ybu'],
                             params.getForEqs('abzb')['ybd'],
                             params.getForEqs('abzb')['ilg'])

    def execKe(self, bconv, tconv):
        params = self.params
        kolmrate = 0.

        # instantiate 		
        ransKe = KineticEnergyEquation(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          -kolmrate,
                                          params.getForProp('prop')['prefix'])

        # plot kinetic energy			   
        ransKe.plot_ke(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('kine')['xbl'],
                       params.getForEqs('kine')['xbr'],
                       params.getForEqs('kine')['ybu'],
                       params.getForEqs('kine')['ybd'],
                       params.getForEqs('kine')['ilg'])

    def execKeEq(self, kolmrate, bconv, tconv):
        params = self.params

        # instantiate 		
        ransKe = KineticEnergyEquation(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          -kolmrate,
                                          params.getForProp('prop')['prefix'])

        # plot kinetic energy equation			     
        ransKe.plot_ke_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('kieq')['xbl'],
                                params.getForEqs('kieq')['xbr'],
                                params.getForEqs('kieq')['ybu'],
                                params.getForEqs('kieq')['ybd'],
                                params.getForEqs('kieq')['ilg'])

    def execTe(self, bconv, tconv):
        params = self.params
        kolmrate = 0.

        # instantiate 		
        ransTe = TotalEnergyEquation(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['nsdim'],
                                        -kolmrate,
                                        params.getForProp('prop')['prefix'])

        # plot total energy			   
        ransTe.plot_et(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('toe')['xbl'],
                       params.getForEqs('toe')['xbr'],
                       params.getForEqs('toe')['ybu'],
                       params.getForEqs('toe')['ybd'],
                       params.getForEqs('toe')['ilg'])

    def execTeEq(self, kolmrate, bconv, tconv):
        params = self.params

        # instantiate 		
        ransTe = TotalEnergyEquation(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['nsdim'],
                                        -kolmrate,
                                        params.getForProp('prop')['prefix'])

        # plot total energy equation			     
        ransTe.plot_et_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('teeq')['xbl'],
                                params.getForEqs('teeq')['xbr'],
                                params.getForEqs('teeq')['ybu'],
                                params.getForEqs('teeq')['ybd'],
                                params.getForEqs('teeq')['ilg'])

    def execRxx(self, bconv, tconv):
        params = self.params
        kolmrate = 0.

        # instantiate 		
        ransRxx = ReynoldsStressXXequation(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['fext'],
                                               params.getForProp('prop')['intc'],
                                               -kolmrate,
                                               params.getForProp('prop')['prefix'])

        # plot reynolds stress rxx			   
        ransRxx.plot_rxx(params.getForProp('prop')['laxis'],
                         bconv, tconv,
                         params.getForEqs('rxx')['xbl'],
                         params.getForEqs('rxx')['xbr'],
                         params.getForEqs('rxx')['ybu'],
                         params.getForEqs('rxx')['ybd'],
                         params.getForEqs('rxx')['ilg'])

    def execRxxEq(self, kolmrate, bconv, tconv):
        params = self.params

        # instantiate 		
        ransRxx = ReynoldsStressXXequation(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['fext'],
                                               params.getForProp('prop')['intc'],
                                               -kolmrate,
                                               params.getForProp('prop')['prefix'])

        # plot reynolds stress rxx			     
        ransRxx.plot_rxx_equation(params.getForProp('prop')['laxis'],
                                  bconv, tconv,
                                  params.getForEqs('rexxeq')['xbl'],
                                  params.getForEqs('rexxeq')['xbr'],
                                  params.getForEqs('rexxeq')['ybu'],
                                  params.getForEqs('rexxeq')['ybd'],
                                  params.getForEqs('rexxeq')['ilg'])

    def execRyy(self, bconv, tconv):
        params = self.params
        kolmrate = 0.

        # instantiate 		
        ransRyy = ReynoldsStressYYequation(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               -kolmrate,
                                               params.getForProp('prop')['prefix'])

        # plot reynolds stress ryy			   
        ransRyy.plot_ryy(params.getForProp('prop')['laxis'],
                         bconv, tconv,
                         params.getForEqs('ryy')['xbl'],
                         params.getForEqs('ryy')['xbr'],
                         params.getForEqs('ryy')['ybu'],
                         params.getForEqs('ryy')['ybd'],
                         params.getForEqs('ryy')['ilg'])

    def execRyyEq(self, kolmrate, bconv, tconv):
        params = self.params

        # instantiate 		
        ransRyy = ReynoldsStressYYequation(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               -kolmrate,
                                               params.getForProp('prop')['prefix'])

        # plot reynolds stress ryy			     
        ransRyy.plot_ryy_equation(params.getForProp('prop')['laxis'],
                                  bconv, tconv,
                                  params.getForEqs('reyyeq')['xbl'],
                                  params.getForEqs('reyyeq')['xbr'],
                                  params.getForEqs('reyyeq')['ybu'],
                                  params.getForEqs('reyyeq')['ybd'],
                                  params.getForEqs('reyyeq')['ilg'])

    def execRzz(self, bconv, tconv):
        params = self.params
        kolmrate = 0.

        # instantiate 		
        ransRzz = ReynoldsStressZZequation(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               -kolmrate,
                                               params.getForProp('prop')['prefix'])

        # plot reynolds stress rzz			   
        ransRzz.plot_rzz(params.getForProp('prop')['laxis'],
                         bconv, tconv,
                         params.getForEqs('rzz')['xbl'],
                         params.getForEqs('rzz')['xbr'],
                         params.getForEqs('rzz')['ybu'],
                         params.getForEqs('rzz')['ybd'],
                         params.getForEqs('rzz')['ilg'])

    def execRzzEq(self, kolmrate, bconv, tconv):
        params = self.params

        # instantiate 		
        ransRzz = ReynoldsStressZZequation(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               -kolmrate,
                                               params.getForProp('prop')['prefix'])

        # plot reynolds stress rzz			     
        ransRzz.plot_rzz_equation(params.getForProp('prop')['laxis'],
                                  bconv, tconv,
                                  params.getForEqs('rezzeq')['xbl'],
                                  params.getForEqs('rezzeq')['xbr'],
                                  params.getForEqs('rezzeq')['ybu'],
                                  params.getForEqs('rezzeq')['ybd'],
                                  params.getForEqs('rezzeq')['ilg'])

    def execAbar(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransAbar = AbarTransportEquation(params.getForProp('prop')['eht_data'],
                                              params.getForProp('prop')['ig'],
                                              params.getForProp('prop')['intc'],
                                              params.getForProp('prop')['nsdim'],
                                              params.getForProp('prop')['prefix'])

        # plot abar
        ransAbar.plot_abar(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('abar')['xbl'],
                           params.getForEqs('abar')['xbr'],
                           params.getForEqs('abar')['ybu'],
                           params.getForEqs('abar')['ybd'],
                           params.getForEqs('abar')['ilg'])

    def execAbarEq(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransAbar = AbarTransportEquation(params.getForProp('prop')['eht_data'],
                                              params.getForProp('prop')['ig'],
                                              params.getForProp('prop')['intc'],
                                              params.getForProp('prop')['nsdim'],
                                              params.getForProp('prop')['prefix'])

        # plot abar equation						       
        ransAbar.plot_abar_equation(params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('abreq')['xbl'],
                                    params.getForEqs('abreq')['xbr'],
                                    params.getForEqs('abreq')['ybu'],
                                    params.getForEqs('abreq')['ybd'],
                                    params.getForEqs('abreq')['ilg'])

    def execFabarx(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransFabarx = AbarFluxTransportEquation(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['nsdim'],
                                                      params.getForProp('prop')['prefix'])

        # plot fabarx
        ransFabarx.plot_abarflux(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('abflx')['xbl'],
                                 params.getForEqs('abflx')['xbr'],
                                 params.getForEqs('abflx')['ybu'],
                                 params.getForEqs('abflx')['ybd'],
                                 params.getForEqs('abflx')['ilg'])

    def execFabarxEq(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransFabarx = AbarFluxTransportEquation(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['nsdim'],
                                                      params.getForProp('prop')['prefix'])

        # plot fabarx equation						       
        ransFabarx.plot_abarflux_equation(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('fabxeq')['xbl'],
                                          params.getForEqs('fabxeq')['xbr'],
                                          params.getForEqs('fabxeq')['ybu'],
                                          params.getForEqs('fabxeq')['ybd'],
                                          params.getForEqs('fabxeq')['ilg'])

    def execZbar(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransZbar = ZbarTransportEquation(params.getForProp('prop')['eht_data'],
                                              params.getForProp('prop')['ig'],
                                              params.getForProp('prop')['intc'],
                                              params.getForProp('prop')['prefix'])

        # plot zbar
        ransZbar.plot_zbar(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('zbar')['xbl'],
                           params.getForEqs('zbar')['xbr'],
                           params.getForEqs('zbar')['ybu'],
                           params.getForEqs('zbar')['ybd'],
                           params.getForEqs('zbar')['ilg'])

    def execZbarEq(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransZbar = ZbarTransportEquation(params.getForProp('prop')['eht_data'],
                                              params.getForProp('prop')['ig'],
                                              params.getForProp('prop')['intc'],
                                              params.getForProp('prop')['prefix'])

        # plot zbar equation						       
        ransZbar.plot_zbar_equation(params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('zbreq')['xbl'],
                                    params.getForEqs('zbreq')['xbr'],
                                    params.getForEqs('zbreq')['ybu'],
                                    params.getForEqs('zbreq')['ybd'],
                                    params.getForEqs('zbreq')['ilg'])

    def execFzbarx(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransFzbarx = ZbarFluxTransportEquation(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['prefix'])

        # plot fzbarx
        ransFzbarx.plot_zbarflux(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('zbflx')['xbl'],
                                 params.getForEqs('zbflx')['xbr'],
                                 params.getForEqs('zbflx')['ybu'],
                                 params.getForEqs('zbflx')['ybd'],
                                 params.getForEqs('zbflx')['ilg'])

    def execFzbarxEq(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransFzbarx = ZbarFluxTransportEquation(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['prefix'])

        # plot fzbarx equation						       
        ransFzbarx.plot_zbarflux_equation(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('fzbxeq')['xbl'],
                                          params.getForEqs('fzbxeq')['xbr'],
                                          params.getForEqs('fzbxeq')['ybu'],
                                          params.getForEqs('fzbxeq')['ybd'],
                                          params.getForEqs('fzbxeq')['ilg'])

    def execPP(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransPP = PressureEquation(params.getForProp('prop')['eht_data'],
                                     params.getForProp('prop')['ig'],
                                     params.getForProp('prop')['fext'],
                                     params.getForProp('prop')['ieos'],
                                     params.getForProp('prop')['intc'],
                                     params.getForProp('prop')['nsdim'],
                                     tke_diss,
                                     params.getForProp('prop')['prefix'])

        ransPP.plot_pp(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('press')['xbl'],
                       params.getForEqs('press')['xbr'],
                       params.getForEqs('press')['ybu'],
                       params.getForEqs('press')['ybd'],
                       params.getForEqs('press')['ilg'])

        # ransPP.plot_dAdt(params.getForProp('prop')['laxis'], \
        #                 params.getForEqs('press')['xbl'], \
        #                 params.getForEqs('press')['xbr'], \
        #                 params.getForEqs('press')['ybu'], \
        #                 params.getForEqs('press')['ybd'], \
        #                 params.getForEqs('press')['ilg'])

    def execPPeq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransPP = PressureEquation(params.getForProp('prop')['eht_data'],
                                     params.getForProp('prop')['ig'],
                                     params.getForProp('prop')['fext'],
                                     params.getForProp('prop')['ieos'],
                                     params.getForProp('prop')['intc'],
                                     params.getForProp('prop')['nsdim'],
                                     tke_diss,
                                     params.getForProp('prop')['prefix'])

        ransPP.plot_pp_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('ppeq')['xbl'],
                                params.getForEqs('ppeq')['xbr'],
                                params.getForEqs('ppeq')['ybu'],
                                params.getForEqs('ppeq')['ybd'],
                                params.getForEqs('ppeq')['ilg'])

    def execPPxflx(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransPPxflx = PressureFluxXequation(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                params.getForProp('prop')['ieos'],
                                                params.getForProp('prop')['intc'],
                                                tke_diss,
                                                params.getForProp('prop')['prefix'])

        ransPPxflx.plot_fppx(params.getForProp('prop')['laxis'],
                             bconv, tconv,
                             params.getForEqs('pressxflx')['xbl'],
                             params.getForEqs('pressxflx')['xbr'],
                             params.getForEqs('pressxflx')['ybu'],
                             params.getForEqs('pressxflx')['ybd'],
                             params.getForEqs('pressxflx')['ilg'])

    def execPPxflxEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransPPxflx = PressureFluxXequation(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                params.getForProp('prop')['ieos'],
                                                params.getForProp('prop')['intc'],
                                                tke_diss,
                                                params.getForProp('prop')['prefix'])

        ransPPxflx.plot_fppx_equation(params.getForProp('prop')['laxis'],
                                      bconv, tconv,
                                      params.getForEqs('ppxflxeq')['xbl'],
                                      params.getForEqs('ppxflxeq')['xbr'],
                                      params.getForEqs('ppxflxeq')['ybu'],
                                      params.getForEqs('ppxflxeq')['ybd'],
                                      params.getForEqs('ppxflxeq')['ilg'])

    def execPPyflx(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransPPyflx = PressureFluxYequation(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                params.getForProp('prop')['ieos'],
                                                params.getForProp('prop')['intc'],
                                                tke_diss,
                                                params.getForProp('prop')['prefix'])

        ransPPyflx.plot_fppy(params.getForProp('prop')['laxis'],
                             bconv, tconv,
                             params.getForEqs('pressyflx')['xbl'],
                             params.getForEqs('pressyflx')['xbr'],
                             params.getForEqs('pressyflx')['ybu'],
                             params.getForEqs('pressyflx')['ybd'],
                             params.getForEqs('pressyflx')['ilg'])

    def execPPyflxEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransPPyflx = PressureFluxYequation(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                params.getForProp('prop')['ieos'],
                                                params.getForProp('prop')['intc'],
                                                tke_diss,
                                                params.getForProp('prop')['prefix'])

        ransPPyflx.plot_fppy_equation(params.getForProp('prop')['laxis'],
                                      bconv, tconv,
                                      params.getForEqs('ppyflxeq')['xbl'],
                                      params.getForEqs('ppyflxeq')['xbr'],
                                      params.getForEqs('ppyflxeq')['ybu'],
                                      params.getForEqs('ppyflxeq')['ybd'],
                                      params.getForEqs('ppyflxeq')['ilg'])

    def execPPzflx(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransPPzflx = PressureFluxZequation(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                params.getForProp('prop')['ieos'],
                                                params.getForProp('prop')['intc'],
                                                tke_diss,
                                                params.getForProp('prop')['prefix'])

        ransPPzflx.plot_fppz(params.getForProp('prop')['laxis'],
                             bconv, tconv,
                             params.getForEqs('presszflx')['xbl'],
                             params.getForEqs('presszflx')['xbr'],
                             params.getForEqs('presszflx')['ybu'],
                             params.getForEqs('presszflx')['ybd'],
                             params.getForEqs('presszflx')['ilg'])

    def execPPzflxEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransPPzflx = PressureFluxZequation(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                params.getForProp('prop')['ieos'],
                                                params.getForProp('prop')['intc'],
                                                tke_diss,
                                                params.getForProp('prop')['prefix'])

        ransPPzflx.plot_fppz_equation(params.getForProp('prop')['laxis'],
                                      bconv, tconv,
                                      params.getForEqs('ppzflxeq')['xbl'],
                                      params.getForEqs('ppzflxeq')['xbr'],
                                      params.getForEqs('ppzflxeq')['ybu'],
                                      params.getForEqs('ppzflxeq')['ybd'],
                                      params.getForEqs('ppzflxeq')['ilg'])

    def execPPvar(self, bconv, tconv):
        params = self.params
        tke_diss = 0.
        tauL = 1.

        # instantiate 		
        ransPPvar = PressureVarianceEquation(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['ieos'],
                                                     params.getForProp('prop')['intc'],
                                                     tke_diss, tauL,
                                                     params.getForProp('prop')['prefix'])

        ransPPvar.plot_sigma_pp(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('pressvar')['xbl'],
                                params.getForEqs('pressvar')['xbr'],
                                params.getForEqs('pressvar')['ybu'],
                                params.getForEqs('pressvar')['ybd'],
                                params.getForEqs('pressvar')['ilg'])

    def execPPvarEq(self, tke_diss, tauL, bconv, tconv):
        params = self.params

        # instantiate 		
        ransPPvar = PressureVarianceEquation(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['ieos'],
                                                     params.getForProp('prop')['intc'],
                                                     tke_diss, tauL,
                                                     params.getForProp('prop')['prefix'])

        ransPPvar.plot_sigma_pp_equation(params.getForProp('prop')['laxis'],
                                         bconv, tconv,
                                         params.getForEqs('ppvareq')['xbl'],
                                         params.getForEqs('ppvareq')['xbr'],
                                         params.getForEqs('ppvareq')['ybu'],
                                         params.getForEqs('ppvareq')['ybd'],
                                         params.getForEqs('ppvareq')['ilg'])

    def execTT(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransTT = TemperatureEquation(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['ieos'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['nsdim'],
                                        tke_diss,
                                        params.getForProp('prop')['prefix'])

        ransTT.plot_tt(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('temp')['xbl'],
                       params.getForEqs('temp')['xbr'],
                       params.getForEqs('temp')['ybu'],
                       params.getForEqs('temp')['ybd'],
                       params.getForEqs('temp')['ilg'])

    def execTTeq(self, wxStudio, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransTT = TemperatureEquation(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['ieos'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['nsdim'],
                                        tke_diss,
                                        params.getForProp('prop')['prefix'])

        if wxStudio[0]:
            ransTT.plot_tt_equation(wxStudio[0], params.getForProp('prop')['laxis'],
                                             bconv, tconv,
                                              wxStudio[1],
                                              wxStudio[2],
                                              wxStudio[3],
                                              wxStudio[4],
                                              params.getForEqs('conteq')['ilg'])
        else:
            ransTT.plot_tt_equation(wxStudio[0], params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('tteq')['xbl'],
                                    params.getForEqs('tteq')['xbr'],
                                    params.getForEqs('tteq')['ybu'],
                                    params.getForEqs('tteq')['ybd'],
                                    params.getForEqs('tteq')['ilg'])


    def execTTvar(self, bconv, tconv):
        params = self.params
        tke_diss = 0.
        tauL = 1.

        # instantiate 		
        ransTTvar = TemperatureVarianceEquation(params.getForProp('prop')['eht_data'],
                                                        params.getForProp('prop')['ig'],
                                                        params.getForProp('prop')['fext'],
                                                        params.getForProp('prop')['ieos'],
                                                        params.getForProp('prop')['intc'],
                                                        tke_diss, tauL,
                                                        params.getForProp('prop')['prefix'])

        ransTTvar.plot_sigma_tt(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('tempvar')['xbl'],
                                params.getForEqs('tempvar')['xbr'],
                                params.getForEqs('tempvar')['ybu'],
                                params.getForEqs('tempvar')['ybd'],
                                params.getForEqs('tempvar')['ilg'])

    def execTTvarEq(self, tke_diss, tauL, bconv, tconv):
        params = self.params

        # instantiate 		
        ransTTvar = TemperatureVarianceEquation(params.getForProp('prop')['eht_data'],
                                                        params.getForProp('prop')['ig'],
                                                        params.getForProp('prop')['fext'],
                                                        params.getForProp('prop')['ieos'],
                                                        params.getForProp('prop')['intc'],
                                                        tke_diss, tauL,
                                                        params.getForProp('prop')['prefix'])

        ransTTvar.plot_sigma_tt_equation(params.getForProp('prop')['laxis'],
                                         bconv, tconv,
                                         params.getForEqs('ttvareq')['xbl'],
                                         params.getForEqs('ttvareq')['xbr'],
                                         params.getForEqs('ttvareq')['ybu'],
                                         params.getForEqs('ttvareq')['ybd'],
                                         params.getForEqs('ttvareq')['ilg'])

    def execTTflx(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransTTflx = TemperatureFluxEquation(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['fext'],
                                                 params.getForProp('prop')['ieos'],
                                                 params.getForProp('prop')['intc'],
                                                 tke_diss,
                                                 params.getForProp('prop')['prefix'])

        ransTTflx.plot_ftt(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('tempflx')['xbl'],
                           params.getForEqs('tempflx')['xbr'],
                           params.getForEqs('tempflx')['ybu'],
                           params.getForEqs('tempflx')['ybd'],
                           params.getForEqs('tempflx')['ilg'])

    def execTTflxEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransTTflx = TemperatureFluxEquation(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['fext'],
                                                 params.getForProp('prop')['ieos'],
                                                 params.getForProp('prop')['intc'],
                                                 tke_diss,
                                                 params.getForProp('prop')['prefix'])

        ransTTflx.plot_ftt_equation(params.getForProp('prop')['laxis'],
                                    bconv, tconv,
                                    params.getForEqs('ttflxeq')['xbl'],
                                    params.getForEqs('ttflxeq')['xbr'],
                                    params.getForEqs('ttflxeq')['ybu'],
                                    params.getForEqs('ttflxeq')['ybd'],
                                    params.getForEqs('ttflxeq')['ilg'])

    def execHH(self, bconv, tconv):
        params = self.params
        tke_diss = 0.

        # instantiate 		
        ransHH = EnthalpyEquation(params.getForProp('prop')['eht_data'],
                                     params.getForProp('prop')['ig'],
                                     params.getForProp('prop')['fext'],
                                     params.getForProp('prop')['ieos'],
                                     params.getForProp('prop')['intc'],
                                     params.getForProp('prop')['nsdim'],
                                     tke_diss,
                                     params.getForProp('prop')['prefix'])

        ransHH.plot_hh(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('enth')['xbl'],
                       params.getForEqs('enth')['xbr'],
                       params.getForEqs('enth')['ybu'],
                       params.getForEqs('enth')['ybd'],
                       params.getForEqs('enth')['ilg'])

    def execHHeq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate 		
        ransHH = EnthalpyEquation(params.getForProp('prop')['eht_data'],
                                     params.getForProp('prop')['ig'],
                                     params.getForProp('prop')['fext'],
                                     params.getForProp('prop')['ieos'],
                                     params.getForProp('prop')['intc'],
                                     params.getForProp('prop')['nsdim'],
                                     tke_diss,
                                     params.getForProp('prop')['prefix'])

        ransHH.plot_hh_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('hheq')['xbl'],
                                params.getForEqs('hheq')['xbr'],
                                params.getForEqs('hheq')['ybu'],
                                params.getForEqs('hheq')['ybd'],
                                params.getForEqs('hheq')['ilg'])

    def execFtvfhX(self, bconv, tconv):
        params = self.params

        # instantiate 		
        ransFtvfhX = FullTurbulenceVelocityFieldHypothesisX(params.getForProp('prop')['eht_data'],
                                                                   params.getForProp('prop')['ig'],
                                                                   params.getForProp('prop')['fext'],
                                                                   params.getForProp('prop')['ieos'],
                                                                   params.getForProp('prop')['intc'],
                                                                   params.getForProp('prop')['prefix'],
                                                                   bconv, tconv)

        ransFtvfhX.plot_ftvfhX_equation(params.getForProp('prop')['laxis'],
                                        params.getForEqs('ftvfh_x')['xbl'],
                                        params.getForEqs('ftvfh_x')['xbr'],
                                        params.getForEqs('ftvfh_x')['ybu'],
                                        params.getForEqs('ftvfh_x')['ybd'],
                                        params.getForEqs('ftvfh_x')['ilg'])

    def execFtvfhY(self, bconv, tconv):
        params = self.params

        # instantiate
        ransFtvfhY = FullTurbulenceVelocityFieldHypothesisY(params.getForProp('prop')['eht_data'],
                                                                   params.getForProp('prop')['ig'],
                                                                   params.getForProp('prop')['fext'],
                                                                   params.getForProp('prop')['ieos'],
                                                                   params.getForProp('prop')['intc'],
                                                                   params.getForProp('prop')['prefix'],
                                                                   bconv, tconv)

        ransFtvfhY.plot_ftvfhY_equation(params.getForProp('prop')['laxis'],
                                        params.getForEqs('ftvfh_y')['xbl'],
                                        params.getForEqs('ftvfh_y')['xbr'],
                                        params.getForEqs('ftvfh_y')['ybu'],
                                        params.getForEqs('ftvfh_y')['ybd'],
                                        params.getForEqs('ftvfh_y')['ilg'])

    def execFtvfhZ(self, bconv, tconv):
        params = self.params

        # instantiate
        ransFtvfhZ = FullTurbulenceVelocityFieldHypothesisZ(params.getForProp('prop')['eht_data'],
                                                                   params.getForProp('prop')['ig'],
                                                                   params.getForProp('prop')['fext'],
                                                                   params.getForProp('prop')['ieos'],
                                                                   params.getForProp('prop')['intc'],
                                                                   params.getForProp('prop')['prefix'],
                                                                   bconv, tconv)

        ransFtvfhZ.plot_ftvfhZ_equation(params.getForProp('prop')['laxis'],
                                        params.getForEqs('ftvfh_z')['xbl'],
                                        params.getForEqs('ftvfh_z')['xbr'],
                                        params.getForEqs('ftvfh_z')['ybu'],
                                        params.getForEqs('ftvfh_z')['ybd'],
                                        params.getForEqs('ftvfh_z')['ilg'])

    def execUxfpd(self, bconv, tconv):
        params = self.params

        # instantiate
        ransUxfpd = UxfpdIdentity(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['ieos'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['prefix'],
                                        bconv, tconv)

        ransUxfpd.plot_uxfpd_identity(params.getForProp('prop')['laxis'],
                                      params.getForEqs('uxfpd')['xbl'],
                                      params.getForEqs('uxfpd')['xbr'],
                                      params.getForEqs('uxfpd')['ybu'],
                                      params.getForEqs('uxfpd')['ybd'],
                                      params.getForEqs('uxfpd')['ilg'])

    def execUyfpd(self, bconv, tconv):
        params = self.params

        # instantiate
        ransUyfpd = UyfpdIdentity(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['ieos'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['prefix'],
                                        bconv, tconv)

        ransUyfpd.plot_uyfpd_identity(params.getForProp('prop')['laxis'],
                                      params.getForEqs('uyfpd')['xbl'],
                                      params.getForEqs('uyfpd')['xbr'],
                                      params.getForEqs('uyfpd')['ybu'],
                                      params.getForEqs('uyfpd')['ybd'],
                                      params.getForEqs('uyfpd')['ilg'])

    def execUzfpd(self, bconv, tconv):
        params = self.params

        # instantiate
        ransUzfpd = UzfpdIdentity(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['ieos'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['prefix'],
                                        bconv, tconv)

        ransUzfpd.plot_uzfpd_identity(params.getForProp('prop')['laxis'],
                                      params.getForEqs('uzfpd')['xbl'],
                                      params.getForEqs('uzfpd')['xbr'],
                                      params.getForEqs('uzfpd')['ybu'],
                                      params.getForEqs('uzfpd')['ybd'],
                                      params.getForEqs('uzfpd')['ilg'])

    def execDivu(self, bconv, tconv):
        params = self.params

        # instantiate
        ransDivu = DivuDilatation(params.getForProp('prop')['eht_data'],
                                       params.getForProp('prop')['ig'],
                                       params.getForProp('prop')['fext'],
                                       params.getForProp('prop')['ieos'],
                                       params.getForProp('prop')['intc'],
                                       params.getForProp('prop')['prefix'],
                                       bconv, tconv)

        ransDivu.plot_divu(params.getForProp('prop')['laxis'],
                           params.getForEqs('divu')['xbl'],
                           params.getForEqs('divu')['xbr'],
                           params.getForEqs('divu')['ybu'],
                           params.getForEqs('divu')['ybd'],
                           params.getForEqs('divu')['ilg'])

        #ransDivu.plot_divu_space_time(params.getForProp('prop')['laxis'],
        #                              bconv, tconv,
        #                              params.getForEqs('conteqfdd')['xbl'],
        #                              params.getForEqs('conteqfdd')['xbr'],
        #                              params.getForEqs('conteqfdd')['ybu'],
        #                              params.getForEqs('conteqfdd')['ybd'],
        #                              params.getForEqs('conteqfdd')['ilg'])

    def SetMatplotlibParams(self):
        """ This routine sets some standard values for matplotlib """
        """ to obtain publication-quality figures """

        # plt.rc('text',usetex=True)
        # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
        plt.rc('font', size=16.)
        plt.rc('lines', linewidth=2, markeredgewidth=2., markersize=12)
        plt.rc('axes', linewidth=1.5)
        plt.rcParams['xtick.major.size'] = 8.
        plt.rcParams['xtick.minor.size'] = 4.
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.17
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams.update({'figure.max_open_warning': 0})
