import EQUATIONS.ContinuityEquationWithMassFlux as contfdd
import EQUATIONS.ContinuityEquationWithFavrianDilatation as contfdil

import EQUATIONS.MomentumEquationX as momx
import EQUATIONS.MomentumEquationY as momy
import EQUATIONS.MomentumEquationZ as momz

import EQUATIONS.ReynoldsStressXXequation as rxx
import EQUATIONS.ReynoldsStressYYequation as ryy
import EQUATIONS.ReynoldsStressZZequation as rzz

import EQUATIONS.TurbulentKineticEnergyEquation as tke
#import EQUATIONS.RadialTurbulentKineticEnergyEquation as rtke N/A YET
#import EQUATIONS.HorizontalTurbulentKineticEnergyEquation as htke N/A YET

import EQUATIONS.InternalEnergyEquation as ei
import EQUATIONS.InternalEnergyFluxEquation as feix
import EQUATIONS.InternalEnergyVarianceEquation as sigmaei

import EQUATIONS.KineticEnergyEquation as ek
import EQUATIONS.TotalEnergyEquation as et

import EQUATIONS.EntropyEquation as ss
import EQUATIONS.EntropyFluxEquation as fssx
import EQUATIONS.EntropyVarianceEquation as sigmass

import EQUATIONS.PressureEquation as pp
import EQUATIONS.PressureFluxEquation as fppx
import EQUATIONS.PressureVarianceEquation as sigmapp

import EQUATIONS.TemperatureEquation as tt
import EQUATIONS.TemperatureFluxEquation as fttx
import EQUATIONS.TemperatureVarianceEquation as sigmatt

import EQUATIONS.EnthalpyEquation as hh
import EQUATIONS.EnthalpyFluxEquation as fhhx 
import EQUATIONS.EnthalpyVarianceEquation as sigmahh 

import EQUATIONS.DensityVarianceEquation as sigmadd
import EQUATIONS.TurbulentMassFluxEquation as a
import EQUATIONS.DensitySpecificVolumeCovarianceEquation as b

import EQUATIONS.XtransportEquation as xtra 
import EQUATIONS.XfluxEquation as xflx
import EQUATIONS.XvarianceEquation as xvar
import EQUATIONS.Xdiffusivity as xdiff

import EQUATIONS.AbarTransportEquation as abar
import EQUATIONS.ZbarTransportEquation as zbar

import EQUATIONS.AbarFluxTransportEquation as fabarx
import EQUATIONS.ZbarFluxTransportEquation as fzbarx

import EQUATIONS.TemperatureDensity as ttdd
import EQUATIONS.PressureInternalEnergy as ppei
import EQUATIONS.NuclearEnergyProduction as enuc
import EQUATIONS.TemperatureGradients as nablas
import EQUATIONS.Degeneracy as psi
import EQUATIONS.Velocities as vel
import EQUATIONS.RelativeRMSflct as rms
import EQUATIONS.AbarZbar as abarzbar
import EQUATIONS.BruntVaisalla as bruntv
import EQUATIONS.Buoyancy as buo

# import classes for hydrodynamic stellar structure equations
import EQUATIONS.HsseContinuityEquation as hssecont
import EQUATIONS.HsseMomentumEquationX as hssemomx
import EQUATIONS.HsseTemperatureEquation as hssetemp
import EQUATIONS.HsseLuminosityEquation as hsselumi
import EQUATIONS.HsseXtransportEquation as hssecomp

import matplotlib.pyplot as plt 

class MasterPlot():

    def __init__(self,params):

        self.params = params

    def execRho(self):
	
        params = self.params

        # instantiate 
        ransCONT = contfdil.ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])

        # plot density
        ransCONT.plot_rho(params.getForProp('prop')['laxis'],\
                          params.getForEqs('rho')['xbl'],\
                          params.getForEqs('rho')['xbr'],\
                          params.getForEqs('rho')['ybu'],\
                          params.getForEqs('rho')['ybd'],\
                          params.getForEqs('rho')['ilg'])

    def execContEq(self):
						  
        params = self.params						  
						  
        # instantiate 
        ransCONT = contfdil.ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])

        # plot continuity equation						       
        ransCONT.plot_continuity_equation(params.getForProp('prop')['laxis'],\
                                          params.getForEqs('conteq')['xbl'],\
                                          params.getForEqs('conteq')['xbr'],\
                                          params.getForEqs('conteq')['ybu'],\
                                          params.getForEqs('conteq')['ybd'],\
                                          params.getForEqs('conteq')['ilg'])
								  
    def execContEqBar(self):

        params = self.params
	
        # instantiate 
        ransCONT = contfdil.ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])

        # plot continuity equation integral budget					       
        ransCONT.plot_continuity_equation_integral_budget(params.getForProp('prop')['laxis'],\
                                                          params.getForEqsBar('conteqBar')['xbl'],\
                                                          params.getForEqsBar('conteqBar')['xbr'],\
                                                          params.getForEqsBar('conteqBar')['ybu'],\
                                                          params.getForEqsBar('conteqBar')['ybd'])


    def execContFddEq(self):
						  
        params = self.params						  
						  
        # instantiate 
        ransCONTfdd = contfdd.ContinuityEquationWithMassFlux(params.getForProp('prop')['eht_data'],\
                                                             params.getForProp('prop')['ig'],\
                                                             params.getForProp('prop')['intc'],\
                                                             params.getForProp('prop')['prefix'])

        # plot continuity equation						       
        ransCONTfdd.plot_continuity_equation(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('conteqfdd')['xbl'],\
                                             params.getForEqs('conteqfdd')['xbr'],\
                                             params.getForEqs('conteqfdd')['ybu'],\
                                             params.getForEqs('conteqfdd')['ybd'],\
                                             params.getForEqs('conteqfdd')['ilg'])
											 
    def execContFddEqBar(self):
						  
        params = self.params						  
						  
        # instantiate 
        ransCONTfdd = contfdd.ContinuityEquationWithMassFlux(params.getForProp('prop')['eht_data'],\
                                                             params.getForProp('prop')['ig'],\
                                                             params.getForProp('prop')['intc'],\
                                                             params.getForProp('prop')['prefix'])

        # plot continuity equation integral budget					       
        ransCONTfdd.plot_continuity_equation_integral_budget(params.getForProp('prop')['laxis'],\
                                                          params.getForEqsBar('conteqfddBar')['xbl'],\
                                                          params.getForEqsBar('conteqfddBar')['xbr'],\
                                                          params.getForEqsBar('conteqfddBar')['ybu'],\
                                                          params.getForEqsBar('conteqfddBar')['ybd'])		
														  
    def execHssContEq(self):
						  
        params = self.params						  
						  
        # instantiate 
        ranshssecont = hssecont.HsseContinuityEquation(params.getForProp('prop')['eht_data'],\
                                                             params.getForProp('prop')['ig'],\
                                                             params.getForProp('prop')['intc'],\
                                                             params.getForProp('prop')['prefix'])

        # plot continuity equation						       
        ranshssecont.plot_continuity_equation(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('cteqhsse')['xbl'],\
                                             params.getForEqs('cteqhsse')['xbr'],\
                                             params.getForEqs('cteqhsse')['ybu'],\
                                             params.getForEqs('cteqhsse')['ybd'],\
                                             params.getForEqs('cteqhsse')['ilg'])

        # plot continuity equation alternative						       
        ranshssecont.plot_continuity_equation_2(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('cteqhsse')['xbl'],\
                                             params.getForEqs('cteqhsse')['xbr'],\
                                             params.getForEqs('cteqhsse')['ybu'],\
                                             params.getForEqs('cteqhsse')['ybd'],\
                                             params.getForEqs('cteqhsse')['ilg'])											 

        # plot continuity equation alternative simplified						       
        ranshssecont.plot_continuity_equation_3(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('cteqhsse')['xbl'],\
                                             params.getForEqs('cteqhsse')['xbr'],\
                                             params.getForEqs('cteqhsse')['ybu'],\
                                             params.getForEqs('cteqhsse')['ybd'],\
                                             params.getForEqs('cteqhsse')['ilg'])

											 
    def execHssMomxEq(self):
						  
        params = self.params						  
						  
        # instantiate 
        ranshssemomx = hssemomx.HsseMomentumEquationX(params.getForProp('prop')['eht_data'],\
                                                             params.getForProp('prop')['ig'],\
                                                             params.getForProp('prop')['intc'],\
                                                             params.getForProp('prop')['prefix'])

        # plot hsse momentm equation						       
        ranshssemomx.plot_momentum_equation_x(params.getForProp('prop')['laxis'],\
                                              params.getForEqs('mxeqhsse')['xbl'],\
                                              params.getForEqs('mxeqhsse')['xbr'],\
                                              params.getForEqs('mxeqhsse')['ybu'],\
                                              params.getForEqs('mxeqhsse')['ybd'],\
                                              params.getForEqs('mxeqhsse')['ilg'])
											 
        # plot hsse momentm equation alternative						       
        ranshssemomx.plot_momentum_equation_x_2(params.getForProp('prop')['laxis'],\
                                              params.getForEqs('mxeqhsse')['xbl'],\
                                              params.getForEqs('mxeqhsse')['xbr'],\
                                              params.getForEqs('mxeqhsse')['ybu'],\
                                              params.getForEqs('mxeqhsse')['ybd'],\
                                              params.getForEqs('mxeqhsse')['ilg'])

        # plot hsse momentm equation alternative simplified						       
        ranshssemomx.plot_momentum_equation_x_3(params.getForProp('prop')['laxis'],\
                                              params.getForEqs('mxeqhsse')['xbl'],\
                                              params.getForEqs('mxeqhsse')['xbr'],\
                                              params.getForEqs('mxeqhsse')['ybu'],\
                                              params.getForEqs('mxeqhsse')['ybd'],\
                                              params.getForEqs('mxeqhsse')['ilg'])											  
											  
											 
    def execHssTempEq(self,tke_diss):
						  
        params = self.params						  
						
        # instantiate 
        ranshssetemp = hssetemp.HsseTemperatureEquation(params.getForProp('prop')['eht_data'],\
                                                             params.getForProp('prop')['ig'],\
                                                             params.getForProp('prop')['intc'],\
                                                             tke_diss,\
                                                             params.getForProp('prop')['prefix'])

        # plot hsse temperature equation						       
        ranshssetemp.plot_tt_equation(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('tpeqhsse')['xbl'],\
                                             params.getForEqs('tpeqhsse')['xbr'],\
                                             params.getForEqs('tpeqhsse')['ybu'],\
                                             params.getForEqs('tpeqhsse')['ybd'],\
                                             params.getForEqs('tpeqhsse')['ilg'])

        # plot hsse temperature equation alternative						       
        ranshssetemp.plot_tt_equation_2(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('tpeqhsse')['xbl'],\
                                             params.getForEqs('tpeqhsse')['xbr'],\
                                             params.getForEqs('tpeqhsse')['ybu'],\
                                             params.getForEqs('tpeqhsse')['ybd'],\
                                             params.getForEqs('tpeqhsse')['ilg'])

        # plot hsse temperature equation alternative simplified						       
        ranshssetemp.plot_tt_equation_3(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('tpeqhsse')['xbl'],\
                                             params.getForEqs('tpeqhsse')['xbr'],\
                                             params.getForEqs('tpeqhsse')['ybu'],\
                                             params.getForEqs('tpeqhsse')['ybd'],\
                                             params.getForEqs('tpeqhsse')['ilg'])
											 
    def execHssLumiEq(self,tke_diss):
						  
        params = self.params						  
						
        # instantiate 
        ranshsselumi = hsselumi.HsseLuminosityEquation(params.getForProp('prop')['eht_data'],\
                                                             params.getForProp('prop')['ig'],\
                                                             params.getForProp('prop')['intc'],\
                                                             tke_diss,\
                                                             params.getForProp('prop')['prefix'])

        # plot hsse temperature equation						       
        ranshsselumi.plot_luminosity_equation(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('lueqhsse')['xbl'],\
                                             params.getForEqs('lueqhsse')['xbr'],\
                                             params.getForEqs('lueqhsse')['ybu'],\
                                             params.getForEqs('lueqhsse')['ybd'],\
                                             params.getForEqs('lueqhsse')['ilg'])											 

        # plot hsse temperature equation exact						       
        ranshsselumi.plot_luminosity_equation_exact(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('lueqhsse')['xbl'],\
                                             params.getForEqs('lueqhsse')['xbr'],\
                                             params.getForEqs('lueqhsse')['ybu'],\
                                             params.getForEqs('lueqhsse')['ybd'],\
                                             params.getForEqs('lueqhsse')['ilg'])

        # plot hsse temperature equation alternative						       
        ranshsselumi.plot_luminosity_equation_2(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('lueqhsse')['xbl'],\
                                             params.getForEqs('lueqhsse')['xbr'],\
                                             params.getForEqs('lueqhsse')['ybu'],\
                                             params.getForEqs('lueqhsse')['ybd'],\
                                             params.getForEqs('lueqhsse')['ilg'])	

        # plot hsse temperature equation alternative simplified						       
        ranshsselumi.plot_luminosity_equation_3(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('lueqhsse')['xbl'],\
                                             params.getForEqs('lueqhsse')['xbr'],\
                                             params.getForEqs('lueqhsse')['ybu'],\
                                             params.getForEqs('lueqhsse')['ybd'],\
                                             params.getForEqs('lueqhsse')['ilg'])

											 
    def execHssCompEq(self):
						  
        params = self.params						  
						
        # instantiate 
        ranshssecomp = hssecomp.HsseXtransportEquation(params.getForProp('prop')['eht_data'],\
                                                             params.getForProp('prop')['ig'],\
                                                             params.getForProp('prop')['intc'],\
                                                             params.getForProp('prop')['prefix'])

        # plot hsse X transport equation						       
        ranshssecomp.plot_Xtransport_equation(params.getForProp('prop')['laxis'],\
                                             params.getForEqs('coeqhsse')['xbl'],\
                                             params.getForEqs('coeqhsse')['xbr'],\
                                             params.getForEqs('coeqhsse')['ybu'],\
                                             params.getForEqs('coeqhsse')['ybd'],\
                                             params.getForEqs('coeqhsse')['ilg'])											 

    def execHssCompEq(self,inuc,element,x):

        params = self.params
	
        # instantiate 
        ranshssecomp = hssecomp.HsseXtransportEquation(params.getForProp('prop')['eht_data'],\
                                                       params.getForProp('prop')['ig'],\
                                                       inuc,element,\
                                                       params.getForProp('prop')['intc'],\
                                                       params.getForProp('prop')['prefix'])
							
        ranshssecomp.plot_Xtransport_equation(params.getForProp('prop')['laxis'],\
                                              params.getForEqs(x)['xbl'],\
                                              params.getForEqs(x)['xbr'],\
                                              params.getForEqs(x)['ybu'],\
                                              params.getForEqs(x)['ybd'],\
                                              params.getForEqs(x)['ilg'])
											 
    def execXrho(self,inuc,element,x):
	
    	params = self.params	

        # instantiate 		
        ransXtra = xtra.XtransportEquation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           inuc,element,\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])

        ransXtra.plot_Xrho(params.getForProp('prop')['laxis'],\
                           params.getForEqs(x)['xbl'],\
                           params.getForEqs(x)['xbr'],\
                           params.getForEqs(x)['ybu'],\
                           params.getForEqs(x)['ybd'],\
                           params.getForEqs(x)['ilg'])

    def execXtrsEq(self,inuc,element,x):

        params = self.params
	
        # instantiate 
        ransXtra = xtra.XtransportEquation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           inuc,element,\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
							
        ransXtra.plot_Xtransport_equation(params.getForProp('prop')['laxis'],\
                                          params.getForEqs(x)['xbl'],\
                                          params.getForEqs(x)['xbr'],\
                                          params.getForEqs(x)['ybu'],\
                                          params.getForEqs(x)['ybd'],\
                                          params.getForEqs(x)['ilg'])	
										  										  

    def execXtrsEqBar(self,inuc,element,x):

        params = self.params
	
        # instantiate 
        ransXtra = xtra.XtransportEquation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           inuc,element,\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
																	  
        # plot X transport equation integral budget					       
        ransXtra.plot_Xtransport_equation_integral_budget(params.getForProp('prop')['laxis'],\
                                                          params.getForEqsBar(x)['xbl'],\
                                                          params.getForEqsBar(x)['xbr'],\
                                                          params.getForEqsBar(x)['ybu'],\
                                                          params.getForEqsBar(x)['ybd'])
										  
										  
    def execXflx(self,inuc,element,x):
	
    	params = self.params	

        # instantiate 		
        ransXflx = xflx.XfluxEquation(params.getForProp('prop')['eht_data'],\
                                      params.getForProp('prop')['ig'],\
                                      inuc,element,\
                                      params.getForProp('prop')['intc'],\
                                      params.getForProp('prop')['prefix'])

        ransXflx.plot_Xflux(params.getForProp('prop')['laxis'],\
                            params.getForEqs(x)['xbl'],\
                            params.getForEqs(x)['xbr'],\
                            params.getForEqs(x)['ybu'],\
                            params.getForEqs(x)['ybd'],\
                            params.getForEqs(x)['ilg'])

    def execXflxEq(self,inuc,element,x):

        params = self.params
	 				    					  
        # instantiate 
        ransXflx = xflx.XfluxEquation(params.getForProp('prop')['eht_data'],\
                                      params.getForProp('prop')['ig'],\
                                      inuc,element,\
                                      params.getForProp('prop')['intc'],\
                                      params.getForProp('prop')['prefix'])
							
        ransXflx.plot_Xflux_equation(params.getForProp('prop')['laxis'],\
                                     params.getForEqs(x)['xbl'],\
                                     params.getForEqs(x)['xbr'],\
                                     params.getForEqs(x)['ybu'],\
                                     params.getForEqs(x)['ybd'],\
                                     params.getForEqs(x)['ilg'])	
										  

    def execXvar(self,inuc,element,x):
	
    	params = self.params	
        tauL = 1.
		
        # instantiate 		
        ransXvar = xvar.XvarianceEquation(params.getForProp('prop')['eht_data'],\
                                          params.getForProp('prop')['ig'],\
                                          inuc,element,tauL,\
                                          params.getForProp('prop')['intc'],\
                                          params.getForProp('prop')['prefix'])

        ransXvar.plot_Xvariance(params.getForProp('prop')['laxis'],\
                                params.getForEqs(x)['xbl'],\
                                params.getForEqs(x)['xbr'],\
                                params.getForEqs(x)['ybu'],\
                                params.getForEqs(x)['ybd'],\
                                params.getForEqs(x)['ilg'])

    def execXvarEq(self,inuc,element,x,tauL):

        params = self.params
			  
        # instantiate 
        ransXvar = xvar.XvarianceEquation(params.getForProp('prop')['eht_data'],\
                                          params.getForProp('prop')['ig'],\
                                          inuc,element,tauL, \
                                          params.getForProp('prop')['intc'],\
                                          params.getForProp('prop')['prefix'])
							
        ransXvar.plot_Xvariance_equation(params.getForProp('prop')['laxis'],\
                                         params.getForEqs(x)['xbl'],\
                                         params.getForEqs(x)['xbr'],\
                                         params.getForEqs(x)['ybu'],\
                                         params.getForEqs(x)['ybd'],\
                                         params.getForEqs(x)['ilg'])	
							

    def execDiff(self,inuc,element,x,lc,uconv):

        params = self.params
						    			  
        # instantiate 
        ransXdiff = xdiff.Xdiffusivity(params.getForProp('prop')['eht_data'],\
                                       params.getForProp('prop')['ig'],\
                                       inuc,element,lc,uconv,\
                                       params.getForProp('prop')['intc'],\
                                       params.getForProp('prop')['prefix'])
							
        ransXdiff.plot_X_Ediffusivity(params.getForProp('prop')['laxis'],\
                                      params.getForEqs(x)['xbl'],\
                                      params.getForEqs(x)['xbr'],\
                                      params.getForEqs(x)['ybu'],\
                                      params.getForEqs(x)['ybd'],\
                                      params.getForEqs(x)['ilg'])	
							
    def execTke(self):
						  
        params = self.params			
        kolmrate = 0.		
						  
        # instantiate 		
        ransTke =  tke.TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      -kolmrate,\
                                                      params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy			   
        ransTke.plot_tke(params.getForProp('prop')['laxis'],\
                         params.getForEqs('tkie')['xbl'],\
                         params.getForEqs('tkie')['xbr'],\
                         params.getForEqs('tkie')['ybu'],\
                         params.getForEqs('tkie')['ybd'],\
                         params.getForEqs('tkie')['ilg'])
										  
        # plot turbulent kinetic energy evolution	   
        ransTke.plot_tke_evolution()

										  
    def execTkeEq(self,kolmrate):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTke =  tke.TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      -kolmrate,\
                                                      params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy equation			     
        ransTke.plot_tke_equation(params.getForProp('prop')['laxis'],\
                                  params.getForEqs('tkeeq')['xbl'],\
                                  params.getForEqs('tkeeq')['xbr'],\
                                  params.getForEqs('tkeeq')['ybu'],\
                                  params.getForEqs('tkeeq')['ybd'],\
                                  params.getForEqs('tkeeq')['ilg'])
				  
    def execMomx(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomx =  momx.MomentumEquationX(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
								   
        ransMomx.plot_momentum_x(params.getForProp('prop')['laxis'],\
                                 params.getForEqs('momex')['xbl'],\
                                 params.getForEqs('momex')['xbr'],\
                                 params.getForEqs('momex')['ybu'],\
                                 params.getForEqs('momex')['ybd'],\
                                 params.getForEqs('momex')['ilg'])
										  
    def execMomxEq(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomx =  momx.MomentumEquationX(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
								   
        ransMomx.plot_momentum_equation_x(params.getForProp('prop')['laxis'],\
                                          params.getForEqs('momxeq')['xbl'],\
                                          params.getForEqs('momxeq')['xbr'],\
                                          params.getForEqs('momxeq')['ybu'],\
                                          params.getForEqs('momxeq')['ybd'],\
                                          params.getForEqs('momxeq')['ilg'])		  
									  	  
    def execMomy(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomy =  momy.MomentumEquationY(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
								   
        ransMomy.plot_momentum_y(params.getForProp('prop')['laxis'],\
                                 params.getForEqs('momey')['xbl'],\
                                 params.getForEqs('momey')['xbr'],\
                                 params.getForEqs('momey')['ybu'],\
                                 params.getForEqs('momey')['ybd'],\
                                 params.getForEqs('momey')['ilg'])
										  
    def execMomyEq(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomy =  momy.MomentumEquationY(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
								   
        ransMomy.plot_momentum_equation_y(params.getForProp('prop')['laxis'],\
                                          params.getForEqs('momyeq')['xbl'],\
                                          params.getForEqs('momyeq')['xbr'],\
                                          params.getForEqs('momyeq')['ybu'],\
                                          params.getForEqs('momyeq')['ybd'],\
                                          params.getForEqs('momyeq')['ilg'])
	
    def execMomz(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomz =  momz.MomentumEquationZ(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
								   
        ransMomz.plot_momentum_z(params.getForProp('prop')['laxis'],\
                                 params.getForEqs('momez')['xbl'],\
                                 params.getForEqs('momez')['xbr'],\
                                 params.getForEqs('momez')['ybu'],\
                                 params.getForEqs('momez')['ybd'],\
                                 params.getForEqs('momez')['ilg'])
										  
    def execMomzEq(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomz =  momz.MomentumEquationZ(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])
								   
        ransMomz.plot_momentum_equation_z(params.getForProp('prop')['laxis'],\
                                          params.getForEqs('momzeq')['xbl'],\
                                          params.getForEqs('momzeq')['xbr'],\
                                          params.getForEqs('momzeq')['ybu'],\
                                          params.getForEqs('momzeq')['ybd'],\
                                          params.getForEqs('momzeq')['ilg'])
				  
    def execEi(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransEi =  ei.InternalEnergyEquation(params.getForProp('prop')['eht_data'],\
                                            params.getForProp('prop')['ig'],\
                                            params.getForProp('prop')['intc'],\
                                            tke_diss,\
                                            params.getForProp('prop')['prefix'])
								   
        ransEi.plot_ei(params.getForProp('prop')['laxis'],\
                       params.getForEqs('eint')['xbl'],\
                       params.getForEqs('eint')['xbr'],\
                       params.getForEqs('eint')['ybu'],\
                       params.getForEqs('eint')['ybd'],\
                       params.getForEqs('eint')['ilg'])
										  
										  
    def execEiEq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransEi =  ei.InternalEnergyEquation(params.getForProp('prop')['eht_data'],\
                                            params.getForProp('prop')['ig'],\
                                            params.getForProp('prop')['intc'],\
                                            tke_diss,\
                                            params.getForProp('prop')['prefix'])

        ransEi.plot_ei_equation(params.getForProp('prop')['laxis'],\
                                params.getForEqs('eieq')['xbl'],\
                                params.getForEqs('eieq')['xbr'],\
                                params.getForEqs('eieq')['ybu'],\
                                params.getForEqs('eieq')['ybd'],\
                                params.getForEqs('eieq')['ilg'])
	
    def execEiFlx(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransEiFlx =  feix.InternalEnergyFluxEquation(params.getForProp('prop')['eht_data'],\
                                                     params.getForProp('prop')['ig'],\
                                                     params.getForProp('prop')['intc'],\
                                                     tke_diss,\
                                                     params.getForProp('prop')['prefix'])
								   
        ransEiFlx.plot_fei(params.getForProp('prop')['laxis'],\
                           params.getForEqs('eintflx')['xbl'],\
                           params.getForEqs('eintflx')['xbr'],\
                           params.getForEqs('eintflx')['ybu'],\
                           params.getForEqs('eintflx')['ybd'],\
                           params.getForEqs('eintflx')['ilg'])
										  			  
    def execEiFlxEq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransEiFlx =  feix.InternalEnergyFluxEquation(params.getForProp('prop')['eht_data'],\
                                                     params.getForProp('prop')['ig'],\
                                                     params.getForProp('prop')['intc'],\
                                                     tke_diss,\
                                                     params.getForProp('prop')['prefix'])

        ransEiFlx.plot_fei_equation(params.getForProp('prop')['laxis'],\
                                    params.getForEqs('eiflxeq')['xbl'],\
                                    params.getForEqs('eiflxeq')['xbr'],\
                                    params.getForEqs('eiflxeq')['ybu'],\
                                    params.getForEqs('eiflxeq')['ybd'],\
                                    params.getForEqs('eiflxeq')['ilg'])	
									  
    def execHHflx(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransHHflx =  fhhx.EnthalpyFluxEquation(params.getForProp('prop')['eht_data'],\
                                               params.getForProp('prop')['ig'],\
                                               params.getForProp('prop')['intc'],\
                                               tke_diss,\
                                               params.getForProp('prop')['prefix'])
								   
        ransHHflx.plot_fhh(params.getForProp('prop')['laxis'],\
                           params.getForEqs('enthflx')['xbl'],\
                           params.getForEqs('enthflx')['xbr'],\
                           params.getForEqs('enthflx')['ybu'],\
                           params.getForEqs('enthflx')['ybd'],\
                           params.getForEqs('enthflx')['ilg'])
										  			  
    def execHHflxEq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransHHflx =  fhhx.EnthalpyFluxEquation(params.getForProp('prop')['eht_data'],\
                                               params.getForProp('prop')['ig'],\
                                               params.getForProp('prop')['intc'],\
                                               tke_diss,\
                                               params.getForProp('prop')['prefix'])
									   
        ransHHflx.plot_fhh_equation(params.getForProp('prop')['laxis'],\
                                    params.getForEqs('hhflxeq')['xbl'],\
                                    params.getForEqs('hhflxeq')['xbr'],\
                                    params.getForEqs('hhflxeq')['ybu'],\
                                    params.getForEqs('hhflxeq')['ybd'],\
                                    params.getForEqs('hhflxeq')['ilg'])	
									 
    def execHHvar(self):
						  
        params = self.params			
        tke_diss = 0.
        tauL = 1.
						  
        # instantiate 		
        ransHHvar =  sigmahh.EnthalpyVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
					                                  params.getForProp('prop')['intc'],\
                                                      tke_diss,tauL,\
                                                      params.getForProp('prop')['prefix'])
								   
        ransHHvar.plot_sigma_hh(params.getForProp('prop')['laxis'],\
                                params.getForEqs('enthvar')['xbl'],\
                                params.getForEqs('enthvar')['xbr'],\
                                params.getForEqs('enthvar')['ybu'],\
                                params.getForEqs('enthvar')['ybd'],\
                                params.getForEqs('enthvar')['ilg'])
										  		  
    def execHHvarEq(self,tke_diss,tauL):
						  
        params = self.params						  
						  
        # instantiate 		
        ransHHvar =  sigmahh.EnthalpyVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      tke_diss,tauL, \
                                                      params.getForProp('prop')['prefix'])

									   
        ransHHvar.plot_sigma_hh_equation(params.getForProp('prop')['laxis'],\
                     params.getForEqs('hhvareq')['xbl'],\
					 params.getForEqs('hhvareq')['xbr'],\
					 params.getForEqs('hhvareq')['ybu'],\
					 params.getForEqs('hhvareq')['ybd'],\
					 params.getForEqs('hhvareq')['ilg'])			  
									  
									  
									  
    def execEiVar(self):
						  
        params = self.params			
        tke_diss = 0.
        tauL = 1.
						  
        # instantiate 		
        ransEiVar =  sigmaei.InternalEnergyVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                            params.getForProp('prop')['ig'],\
                                                            params.getForProp('prop')['intc'],\
                                                            tke_diss,tauL,\
                                                            params.getForProp('prop')['prefix'])
								   
        ransEiVar.plot_sigma_ei(params.getForProp('prop')['laxis'],\
                                params.getForEqs('eintvar')['xbl'],\
                                params.getForEqs('eintvar')['xbr'],\
                                params.getForEqs('eintvar')['ybu'],\
                                params.getForEqs('eintvar')['ybd'],\
                                params.getForEqs('eintvar')['ilg'])
										  		  
    def execEiVarEq(self,tke_diss,tauL):
						  
        params = self.params						  
						  
        # instantiate 		
        ransEiVar =  sigmaei.InternalEnergyVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                            params.getForProp('prop')['ig'],\
                                                            params.getForProp('prop')['intc'],\
                                                            tke_diss,tauL, \
                                                            params.getForProp('prop')['prefix'])

        ransEiVar.plot_sigma_ei_equation(params.getForProp('prop')['laxis'],\
                                         params.getForEqs('eivareq')['xbl'],\
                                         params.getForEqs('eivareq')['xbr'],\
                                         params.getForEqs('eivareq')['ybu'],\
                                         params.getForEqs('eivareq')['ybd'],\
                                         params.getForEqs('eivareq')['ilg'])	
									  
									  
    def execSS(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransSS =  ss.EntropyEquation(params.getForProp('prop')['eht_data'],\
                                     params.getForProp('prop')['ig'],\
                                     params.getForProp('prop')['intc'],\
                                     tke_diss,\
                                     params.getForProp('prop')['prefix'])
						       	   
        ransSS.plot_ss(params.getForProp('prop')['laxis'],\
                       params.getForEqs('entr')['xbl'],\
                       params.getForEqs('entr')['xbr'],\
                       params.getForEqs('entr')['ybu'],\
                       params.getForEqs('entr')['ybd'],\
                       params.getForEqs('entr')['ilg'])
										  
										  
    def execSSeq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransSS =  ss.EntropyEquation(params.getForProp('prop')['eht_data'],\
                                     params.getForProp('prop')['ig'],\
                                     params.getForProp('prop')['intc'],\
                                     tke_diss,\
                                     params.getForProp('prop')['prefix'])
  
        ransSS.plot_ss_equation(params.getForProp('prop')['laxis'],\
                                params.getForEqs('sseq')['xbl'],\
                                params.getForEqs('sseq')['xbr'],\
                                params.getForEqs('sseq')['ybu'],\
                                params.getForEqs('sseq')['ybd'],\
                                params.getForEqs('sseq')['ilg'])
	
    def execSSflx(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransSSflx =  fssx.EntropyFluxEquation(params.getForProp('prop')['eht_data'],\
                                              params.getForProp('prop')['ig'],\
                                              params.getForProp('prop')['intc'],\
                                              tke_diss,\
                                              params.getForProp('prop')['prefix'])
								   
        ransSSflx.plot_fss(params.getForProp('prop')['laxis'],\
                           params.getForEqs('entrflx')['xbl'],\
                           params.getForEqs('entrflx')['xbr'],\
                           params.getForEqs('entrflx')['ybu'],\
                           params.getForEqs('entrflx')['ybd'],\
                           params.getForEqs('entrflx')['ilg'])
										  			  
    def execSSflxEq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransSSflx =  fssx.EntropyFluxEquation(params.getForProp('prop')['eht_data'],\
                                              params.getForProp('prop')['ig'],\
                                              params.getForProp('prop')['intc'],\
                                              tke_diss,\
                                              params.getForProp('prop')['prefix'])
									   
        ransSSflx.plot_fss_equation(params.getForProp('prop')['laxis'],\
                                    params.getForEqs('ssflxeq')['xbl'],\
                                    params.getForEqs('ssflxeq')['xbr'],\
                                    params.getForEqs('ssflxeq')['ybu'],\
                                    params.getForEqs('ssflxeq')['ybd'],\
                                    params.getForEqs('ssflxeq')['ilg'])	
									    
    def execSSvar(self):
						  
        params = self.params			
        tke_diss = 0.
        tauL = 1.
						  
        # instantiate 		
        ransSSvar =  sigmass.EntropyVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                     params.getForProp('prop')['ig'],\
                                                     params.getForProp('prop')['intc'],\
                                                     tke_diss,tauL,\
                                                     params.getForProp('prop')['prefix'])
								   
        ransSSvar.plot_sigma_ss(params.getForProp('prop')['laxis'],\
                                params.getForEqs('entrvar')['xbl'],\
                                params.getForEqs('entrvar')['xbr'],\
                                params.getForEqs('entrvar')['ybu'],\
                                params.getForEqs('entrvar')['ybd'],\
                                params.getForEqs('entrvar')['ilg'])
										  		  
    def execSSvarEq(self,tke_diss,tauL):
						  
        params = self.params						  
						  
        # instantiate 		
        ransSSvar =  sigmass.EntropyVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                     params.getForProp('prop')['ig'],\
                                                     params.getForProp('prop')['intc'],\
                                                     tke_diss,tauL, \
                                                     params.getForProp('prop')['prefix'])

									   
        ransSSvar.plot_sigma_ss_equation(params.getForProp('prop')['laxis'],\
                                         params.getForEqs('ssvareq')['xbl'],\
                                         params.getForEqs('ssvareq')['xbr'],\
                                         params.getForEqs('ssvareq')['ybu'],\
                                         params.getForEqs('ssvareq')['ybd'],\
                                         params.getForEqs('ssvareq')['ilg'])
	
    def execDDvar(self):
						  
        params = self.params			
        tke_diss = 0.
        tauL = 1.
						  
        # instantiate 		
        ransDDvar =  sigmadd.DensityVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                     params.getForProp('prop')['ig'],\
                                                     params.getForProp('prop')['intc'],\
                                                     tauL,\
                                                     params.getForProp('prop')['prefix'])
								   
        ransDDvar.plot_sigma_dd(params.getForProp('prop')['laxis'],\
                                params.getForEqs('densvar')['xbl'],\
                                params.getForEqs('densvar')['xbr'],\
                                params.getForEqs('densvar')['ybu'],\
                                params.getForEqs('densvar')['ybd'],\
                                params.getForEqs('densvar')['ilg'])
										  
										  
    def execDDvarEq(self,tauL):
						  
        params = self.params						  
						  
        # instantiate 		
        ransSSvar =  sigmadd.DensityVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                     params.getForProp('prop')['ig'],\
                                                     params.getForProp('prop')['intc'],\
                                                     tauL, \
                                                     params.getForProp('prop')['prefix'])

									   
        ransSSvar.plot_sigma_dd_equation(params.getForProp('prop')['laxis'],\
                                         params.getForEqs('ddvareq')['xbl'],\
                                         params.getForEqs('ddvareq')['xbr'],\
                                         params.getForEqs('ddvareq')['ybu'],\
                                         params.getForEqs('ddvareq')['ybd'],\
                                         params.getForEqs('ddvareq')['ilg'])			

    def execTMSflx(self):

        params = self.params	
	
        # instantiate 		
        ransTMSflx =  a.TurbulentMassFluxEquation(params.getForProp('prop')['eht_data'],\
                                                  params.getForProp('prop')['ig'],\
                                                  params.getForProp('prop')['intc'],\
                                                  params.getForProp('prop')['prefix'])
								   
        ransTMSflx.plot_a(params.getForProp('prop')['laxis'],\
                          params.getForEqs('tmsflx')['xbl'],\
                          params.getForEqs('tmsflx')['xbr'],\
                          params.getForEqs('tmsflx')['ybu'],\
                          params.getForEqs('tmsflx')['ybd'],\
                          params.getForEqs('tmsflx')['ilg'])
										  
										  
    def execAeq(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTMSflx =  a.TurbulentMassFluxEquation(params.getForProp('prop')['eht_data'],\
                                                  params.getForProp('prop')['ig'],\
                                                  params.getForProp('prop')['intc'],\
                                                  params.getForProp('prop')['prefix'])

									   
        ransTMSflx.plot_a_equation(params.getForProp('prop')['laxis'],\
                                   params.getForEqs('aeq')['xbl'],\
                                   params.getForEqs('aeq')['xbr'],\
                                   params.getForEqs('aeq')['ybu'],\
                                   params.getForEqs('aeq')['ybd'],\
                                   params.getForEqs('aeq')['ilg'])	
									  
									  
    def execDSVC(self):

        params = self.params	
	
        # instantiate 		
        ransDSVC =  b.DensitySpecificVolumeCovarianceEquation(params.getForProp('prop')['eht_data'],\
                                                              params.getForProp('prop')['ig'],\
                                                              params.getForProp('prop')['intc'],\
                                                              params.getForProp('prop')['prefix'])
								   
        ransDSVC.plot_b(params.getForProp('prop')['laxis'],\
                        params.getForEqs('dsvc')['xbl'],\
                        params.getForEqs('dsvc')['xbr'],\
                        params.getForEqs('dsvc')['ybu'],\
                        params.getForEqs('dsvc')['ybd'],\
                        params.getForEqs('dsvc')['ilg'])
										  
										  
    def execBeq(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransDSVC =  b.DensitySpecificVolumeCovarianceEquation(params.getForProp('prop')['eht_data'],\
                                                              params.getForProp('prop')['ig'],\
                                                              params.getForProp('prop')['intc'],\
                                                              params.getForProp('prop')['prefix'])

									   
        ransDSVC.plot_b_equation(params.getForProp('prop')['laxis'],\
                                 params.getForEqs('beq')['xbl'],\
                                 params.getForEqs('beq')['xbr'],\
                                 params.getForEqs('beq')['ybu'],\
                                 params.getForEqs('beq')['ybd'],\
                                 params.getForEqs('beq')['ilg'])
				 
    def execRhoTemp(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTempRho =  ttdd.TemperatureDensity(params.getForProp('prop')['eht_data'],\
                                               params.getForProp('prop')['ig'],\
                                               params.getForProp('prop')['intc'],\
                                               params.getForProp('prop')['prefix'])

									   
        ransTempRho.plot_ttdd(params.getForProp('prop')['laxis'],\
                              params.getForEqs('ttdd')['xbl'],\
                              params.getForEqs('ttdd')['xbr'],\
                              params.getForEqs('ttdd')['ybu'],\
                              params.getForEqs('ttdd')['ybd'],\
                              params.getForEqs('ttdd')['ilg'])				 
				 
    def execPressEi(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransPressEi =  ppei.PressureInternalEnergy(params.getForProp('prop')['eht_data'],\
                                                   params.getForProp('prop')['ig'],\
                                                   params.getForProp('prop')['intc'],\
                                                   params.getForProp('prop')['prefix'])

									   
        ransPressEi.plot_ppei(params.getForProp('prop')['laxis'],\
                              params.getForEqs('ppei')['xbl'],\
                              params.getForEqs('ppei')['xbr'],\
                              params.getForEqs('ppei')['ybu'],\
                              params.getForEqs('ppei')['ybd'],\
                              params.getForEqs('ppei')['ilg'])				 
				 				 
    def execEnuc(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransEnuc =  enuc.NuclearEnergyProduction(params.getForProp('prop')['eht_data'],\
                                                 params.getForProp('prop')['ig'],\
                                                 params.getForProp('prop')['intc'],\
                                                 params.getForProp('prop')['prefix'])

									   
        ransEnuc.plot_enuc(params.getForProp('prop')['laxis'],\
                           params.getForEqs('enuc')['xbl'],\
                           params.getForEqs('enuc')['xbr'],\
                           params.getForEqs('enuc')['ybu'],\
                           params.getForEqs('enuc')['ybd'],\
                           params.getForEqs('enuc')['ilg'])
								 
    def execNablas(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransNablas =  nablas.TemperatureGradients(params.getForProp('prop')['eht_data'],\
                                                  params.getForProp('prop')['ig'],\
                                                  params.getForProp('prop')['intc'],\
                                                  params.getForProp('prop')['prefix'])

									   
        ransNablas.plot_nablas(params.getForProp('prop')['laxis'],\
                               params.getForEqs('nablas')['xbl'],\
                               params.getForEqs('nablas')['xbr'],\
                               params.getForEqs('nablas')['ybu'],\
                               params.getForEqs('nablas')['ybd'],\
                               params.getForEqs('nablas')['ilg'])
								 
    def execDegeneracy(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransDeg =  psi.Degeneracy(params.getForProp('prop')['eht_data'],\
                                  params.getForProp('prop')['ig'],\
                                  params.getForProp('prop')['intc'],\
                                  params.getForProp('prop')['prefix'])

									   
        ransDeg.plot_degeneracy(params.getForProp('prop')['laxis'],\
                                params.getForEqs('psi')['xbl'],\
                                params.getForEqs('psi')['xbr'],\
                                params.getForEqs('psi')['ybu'],\
                                params.getForEqs('psi')['ybd'],\
                                params.getForEqs('psi')['ilg'])								 

    def execVelocities(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransVel =  vel.Velocities(params.getForProp('prop')['eht_data'],\
                                  params.getForProp('prop')['ig'],\
                                  params.getForProp('prop')['intc'],\
                                  params.getForProp('prop')['prefix'])

									   
        ransVel.plot_velocities(params.getForProp('prop')['laxis'],\
                                params.getForEqs('vel')['xbl'],\
                                params.getForEqs('vel')['xbr'],\
                                params.getForEqs('vel')['ybu'],\
                                params.getForEqs('vel')['ybd'],\
                                params.getForEqs('vel')['ilg'])				 
				 
    def execBruntV(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransBruntV =  bruntv.BruntVaisalla(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])

									   
        ransBruntV.plot_bruntvaisalla(params.getForProp('prop')['laxis'],\
                                      params.getForEqs('nsq')['xbl'],\
                                      params.getForEqs('nsq')['xbr'],\
                                      params.getForEqs('nsq')['ybu'],\
                                      params.getForEqs('nsq')['ybd'],\
                                      params.getForEqs('nsq')['ilg'])	

    def execBuoyancy(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransBuo =  buo.Buoyancy(params.getForProp('prop')['eht_data'],\
                                params.getForProp('prop')['ig'],\
                                params.getForProp('prop')['intc'],\
                                params.getForProp('prop')['prefix'])

									   
        ransBuo.plot_buoyancy(params.getForProp('prop')['laxis'],\
                              params.getForEqs('buo')['xbl'],\
                              params.getForEqs('buo')['xbr'],\
                              params.getForEqs('buo')['ybu'],\
                              params.getForEqs('buo')['ybd'],\
                              params.getForEqs('buo')['ilg'])

    def execRelativeRmsFlct(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransRms =  rms.RelativeRMSflct(params.getForProp('prop')['eht_data'],\
                                       params.getForProp('prop')['ig'],\
                                       params.getForProp('prop')['intc'],\
                                       params.getForProp('prop')['prefix'])

									   
        ransRms.plot_relative_rms_flct(params.getForProp('prop')['laxis'],\
                                       params.getForEqs('relrmsflct')['xbl'],\
                                       params.getForEqs('relrmsflct')['xbr'],\
                                       params.getForEqs('relrmsflct')['ybu'],\
                                       params.getForEqs('relrmsflct')['ybd'],\
                                       params.getForEqs('relrmsflct')['ilg'])				 
			
    def execAbarZbar(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransAZ =  abarzbar.AbarZbar(params.getForProp('prop')['eht_data'],\
                                    params.getForProp('prop')['ig'],\
                                    params.getForProp('prop')['intc'],\
                                    params.getForProp('prop')['prefix'])

									   
        ransAZ.plot_abarzbar(params.getForProp('prop')['laxis'],\
                             params.getForEqs('abzb')['xbl'],\
                             params.getForEqs('abzb')['xbr'],\
                             params.getForEqs('abzb')['ybu'],\
                             params.getForEqs('abzb')['ybd'],\
                             params.getForEqs('abzb')['ilg'])
			
    def execKe(self):
						  
        params = self.params			
        kolmrate = 0.		
						  
        # instantiate 		
        ransKe =  ek.KineticEnergyEquation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           -kolmrate,\
                                           params.getForProp('prop')['prefix'])

        # plot kinetic energy			   
        ransKe.plot_ke(params.getForProp('prop')['laxis'],\
                       params.getForEqs('kine')['xbl'],\
                       params.getForEqs('kine')['xbr'],\
                       params.getForEqs('kine')['ybu'],\
                       params.getForEqs('kine')['ybd'],\
                       params.getForEqs('kine')['ilg'])
										  
										  
    def execKeEq(self,kolmrate):
						  
        params = self.params						  
						  
        # instantiate 		
        ransKe =  ek.KineticEnergyEquation(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           -kolmrate,\
                                           params.getForProp('prop')['prefix'])

        # plot kinetic energy equation			     
        ransKe.plot_ke_equation(params.getForProp('prop')['laxis'],\
                                params.getForEqs('kieq')['xbl'],\
                                params.getForEqs('kieq')['xbr'],\
                                params.getForEqs('kieq')['ybu'],\
                                params.getForEqs('kieq')['ybd'],\
                                params.getForEqs('kieq')['ilg'])		

    def execTe(self):
						  
        params = self.params			
        kolmrate = 0.		
						  
        # instantiate 		
        ransTe =  et.TotalEnergyEquation(params.getForProp('prop')['eht_data'],\
                                         params.getForProp('prop')['ig'],\
                                         params.getForProp('prop')['intc'],\
                                         -kolmrate,\
                                         params.getForProp('prop')['prefix'])

        # plot total energy			   
        ransTe.plot_et(params.getForProp('prop')['laxis'],\
                       params.getForEqs('toe')['xbl'],\
                       params.getForEqs('toe')['xbr'],\
                       params.getForEqs('toe')['ybu'],\
                       params.getForEqs('toe')['ybd'],\
                       params.getForEqs('toe')['ilg'])
										  
										  
    def execTeEq(self,kolmrate):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTe =  et.TotalEnergyEquation(params.getForProp('prop')['eht_data'],\
                                         params.getForProp('prop')['ig'],\
                                         params.getForProp('prop')['intc'],\
                                         -kolmrate,\
                                         params.getForProp('prop')['prefix'])

        # plot total energy equation			     
        ransTe.plot_et_equation(params.getForProp('prop')['laxis'],\
                                params.getForEqs('teeq')['xbl'],\
                                params.getForEqs('teeq')['xbr'],\
                                params.getForEqs('teeq')['ybu'],\
                                params.getForEqs('teeq')['ybd'],\
                                params.getForEqs('teeq')['ilg'])

    def execRxx(self):
						  
        params = self.params			
        kolmrate = 0.		
						  
        # instantiate 		
        ransRxx =  rxx.ReynoldsStressXXequation(params.getForProp('prop')['eht_data'],\
                                                params.getForProp('prop')['ig'],\
                                                params.getForProp('prop')['intc'],\
                                                -kolmrate,\
                                                params.getForProp('prop')['prefix'])

        # plot reynolds stress rxx			   
        ransRxx.plot_rxx(params.getForProp('prop')['laxis'],\
                         params.getForEqs('rxx')['xbl'],\
                         params.getForEqs('rxx')['xbr'],\
                         params.getForEqs('rxx')['ybu'],\
                         params.getForEqs('rxx')['ybd'],\
                         params.getForEqs('rxx')['ilg'])
										  
										  
    def execRxxEq(self,kolmrate):
						  
        params = self.params						  
						  
        # instantiate 		
        ransRxx =  rxx.ReynoldsStressXXequation(params.getForProp('prop')['eht_data'],\
                                                params.getForProp('prop')['ig'],\
                                                params.getForProp('prop')['intc'],\
                                                -kolmrate,\
                                                params.getForProp('prop')['prefix'])

        # plot reynolds stress rxx			     
        ransRxx.plot_rxx_equation(params.getForProp('prop')['laxis'],\
                                  params.getForEqs('rexxeq')['xbl'],\
                                  params.getForEqs('rexxeq')['xbr'],\
                                  params.getForEqs('rexxeq')['ybu'],\
                                  params.getForEqs('rexxeq')['ybd'],\
                                  params.getForEqs('rexxeq')['ilg'])
				  
				  
    def execRyy(self):
						  
        params = self.params			
        kolmrate = 0.		
						  
        # instantiate 		
        ransRyy =  ryy.ReynoldsStressYYequation(params.getForProp('prop')['eht_data'],\
                                                params.getForProp('prop')['ig'],\
                                                params.getForProp('prop')['intc'],\
                                                -kolmrate,\
                                                params.getForProp('prop')['prefix'])

        # plot reynolds stress ryy			   
        ransRyy.plot_ryy(params.getForProp('prop')['laxis'],\
                         params.getForEqs('ryy')['xbl'],\
                         params.getForEqs('ryy')['xbr'],\
                         params.getForEqs('ryy')['ybu'],\
                         params.getForEqs('ryy')['ybd'],\
                         params.getForEqs('ryy')['ilg'])
										  
										  
    def execRyyEq(self,kolmrate):
						  
        params = self.params						  
						  
        # instantiate 		
        ransRyy =  ryy.ReynoldsStressYYequation(params.getForProp('prop')['eht_data'],\
                                                params.getForProp('prop')['ig'],\
                                                params.getForProp('prop')['intc'],\
                                                -kolmrate,\
                                                params.getForProp('prop')['prefix'])

        # plot reynolds stress ryy			     
        ransRyy.plot_ryy_equation(params.getForProp('prop')['laxis'],\
                                  params.getForEqs('reyyeq')['xbl'],\
                                  params.getForEqs('reyyeq')['xbr'],\
                                  params.getForEqs('reyyeq')['ybu'],\
                                  params.getForEqs('reyyeq')['ybd'],\
                                  params.getForEqs('reyyeq')['ilg'])
				  				  
				  
    def execRzz(self):
						  
        params = self.params			
        kolmrate = 0.		
						  
        # instantiate 		
        ransRzz =  rzz.ReynoldsStressZZequation(params.getForProp('prop')['eht_data'],\
                                                params.getForProp('prop')['ig'],\
                                                params.getForProp('prop')['intc'],\
                                                -kolmrate,\
                                                params.getForProp('prop')['prefix'])

        # plot reynolds stress rzz			   
        ransRzz.plot_rzz(params.getForProp('prop')['laxis'],\
                         params.getForEqs('rzz')['xbl'],\
                         params.getForEqs('rzz')['xbr'],\
                         params.getForEqs('rzz')['ybu'],\
                         params.getForEqs('rzz')['ybd'],\
                         params.getForEqs('rzz')['ilg'])
										  
										  
    def execRzzEq(self,kolmrate):
						  
        params = self.params						  
						  
        # instantiate 		
        ransRzz =  rzz.ReynoldsStressZZequation(params.getForProp('prop')['eht_data'],\
                                                params.getForProp('prop')['ig'],\
                                                params.getForProp('prop')['intc'],\
                                                -kolmrate,\
                                                params.getForProp('prop')['prefix'])

        # plot reynolds stress rzz			     
        ransRzz.plot_rzz_equation(params.getForProp('prop')['laxis'],\
                                  params.getForEqs('rezzeq')['xbl'],\
                                  params.getForEqs('rezzeq')['xbr'],\
                                  params.getForEqs('rezzeq')['ybu'],\
                                  params.getForEqs('rezzeq')['ybd'],\
                                  params.getForEqs('rezzeq')['ilg'])
				  				  
    def execAbar(self):
	
        params = self.params

        # instantiate 
        ransAbar = abar.AbarTransportEquation(params.getForProp('prop')['eht_data'],\
                                              params.getForProp('prop')['ig'],\
                                              params.getForProp('prop')['intc'],\
                                              params.getForProp('prop')['prefix'])

	# plot abar
        ransAbar.plot_abar(params.getForProp('prop')['laxis'],\
                           params.getForEqs('abar')['xbl'],\
                           params.getForEqs('abar')['xbr'],\
                           params.getForEqs('abar')['ybu'],\
                           params.getForEqs('abar')['ybd'],\
                           params.getForEqs('abar')['ilg'])

    def execAbarEq(self):
						  
        params = self.params						  
						  
        # instantiate 
        ransAbar = abar.AbarTransportEquation(params.getForProp('prop')['eht_data'],\
                                              params.getForProp('prop')['ig'],\
                                              params.getForProp('prop')['intc'],\
                                              params.getForProp('prop')['prefix'])

        # plot abar equation						       
        ransAbar.plot_abar_equation(params.getForProp('prop')['laxis'],\
                                    params.getForEqs('abreq')['xbl'],\
                                    params.getForEqs('abreq')['xbr'],\
                                    params.getForEqs('abreq')['ybu'],\
                                    params.getForEqs('abreq')['ybd'],\
                                    params.getForEqs('abreq')['ilg'])								  
					
    def execFabarx(self):
	
        params = self.params

        # instantiate 
        ransFabarx = fabarx.AbarFluxTransportEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      params.getForProp('prop')['prefix'])

        # plot fabarx
        ransFabarx.plot_abarflux(params.getForProp('prop')['laxis'],\
                                 params.getForEqs('abflx')['xbl'],\
                                 params.getForEqs('abflx')['xbr'],\
                                 params.getForEqs('abflx')['ybu'],\
                                 params.getForEqs('abflx')['ybd'],\
                                 params.getForEqs('abflx')['ilg'])

    def execFabarxEq(self):
						  
        params = self.params						  
						  
        # instantiate 
        ransFabarx = fabarx.AbarFluxTransportEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      params.getForProp('prop')['prefix'])

        # plot fabarx equation						       
        ransFabarx.plot_abarflux_equation(params.getForProp('prop')['laxis'],\
                                          params.getForEqs('fabxeq')['xbl'],\
                                          params.getForEqs('fabxeq')['xbr'],\
                                          params.getForEqs('fabxeq')['ybu'],\
                                          params.getForEqs('fabxeq')['ybd'],\
                                          params.getForEqs('fabxeq')['ilg'])			
					
    def execZbar(self):
	
        params = self.params

        # instantiate 
        ransZbar = zbar.ZbarTransportEquation(params.getForProp('prop')['eht_data'],\
                                              params.getForProp('prop')['ig'],\
                                              params.getForProp('prop')['intc'],\
                                              params.getForProp('prop')['prefix'])

        # plot zbar
        ransZbar.plot_zbar(params.getForProp('prop')['laxis'],\
                           params.getForEqs('zbar')['xbl'],\
                           params.getForEqs('zbar')['xbr'],\
                           params.getForEqs('zbar')['ybu'],\
                           params.getForEqs('zbar')['ybd'],\
                           params.getForEqs('zbar')['ilg'])

    def execZbarEq(self):
					  
        params = self.params						  
						  
        # instantiate 
        ransZbar = zbar.ZbarTransportEquation(params.getForProp('prop')['eht_data'],\
                                              params.getForProp('prop')['ig'],\
                                              params.getForProp('prop')['intc'],\
                                              params.getForProp('prop')['prefix'])

        # plot zbar equation						       
        ransZbar.plot_zbar_equation(params.getForProp('prop')['laxis'],\
                                    params.getForEqs('zbreq')['xbl'],\
                                    params.getForEqs('zbreq')['xbr'],\
                                    params.getForEqs('zbreq')['ybu'],\
                                    params.getForEqs('zbreq')['ybd'],\
                                    params.getForEqs('zbreq')['ilg'])			

    def execFzbarx(self):
	
        params = self.params

        # instantiate 
        ransFzbarx = fzbarx.ZbarFluxTransportEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      params.getForProp('prop')['prefix'])

        # plot fzbarx
        ransFzbarx.plot_zbarflux(params.getForProp('prop')['laxis'],\
                                 params.getForEqs('zbflx')['xbl'],\
                                 params.getForEqs('zbflx')['xbr'],\
                                 params.getForEqs('zbflx')['ybu'],\
                                 params.getForEqs('zbflx')['ybd'],\
                                 params.getForEqs('zbflx')['ilg'])

    def execFzbarxEq(self):
						  
        params = self.params						  
						  
        # instantiate 
        ransFzbarx = fzbarx.ZbarFluxTransportEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      params.getForProp('prop')['prefix'])

        # plot fzbarx equation						       
        ransFzbarx.plot_zbarflux_equation(params.getForProp('prop')['laxis'],\
                                          params.getForEqs('fzbxeq')['xbl'],\
                                          params.getForEqs('fzbxeq')['xbr'],\
                                          params.getForEqs('fzbxeq')['ybu'],\
                                          params.getForEqs('fzbxeq')['ybd'],\
                                          params.getForEqs('fzbxeq')['ilg'])
					  
    def execPP(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransPP =  pp.PressureEquation(params.getForProp('prop')['eht_data'],\
                                      params.getForProp('prop')['ig'],\
                                      params.getForProp('prop')['intc'],\
                                      tke_diss,\
                                      params.getForProp('prop')['prefix'])
								   
        ransPP.plot_pp(params.getForProp('prop')['laxis'],\
                       params.getForEqs('press')['xbl'],\
                       params.getForEqs('press')['xbr'],\
                       params.getForEqs('press')['ybu'],\
                       params.getForEqs('press')['ybd'],\
                       params.getForEqs('press')['ilg'])
										  
										  
    def execPPeq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransPP =  pp.PressureEquation(params.getForProp('prop')['eht_data'],\
                                      params.getForProp('prop')['ig'],\
                                      params.getForProp('prop')['intc'],\
                                      tke_diss,\
                                      params.getForProp('prop')['prefix'])

									   
        ransPP.plot_pp_equation(params.getForProp('prop')['laxis'],\
                                params.getForEqs('ppeq')['xbl'],\
                                params.getForEqs('ppeq')['xbr'],\
                                params.getForEqs('ppeq')['ybu'],\
                                params.getForEqs('ppeq')['ybd'],\
                                params.getForEqs('ppeq')['ilg'])								  

    def execPPflx(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransPPflx =  fppx.PressureFluxEquation(params.getForProp('prop')['eht_data'],\
                                               params.getForProp('prop')['ig'],\
                                               params.getForProp('prop')['intc'],\
                                               tke_diss,\
                                               params.getForProp('prop')['prefix'])
								   
        ransPPflx.plot_fpp(params.getForProp('prop')['laxis'],\
                           params.getForEqs('pressflx')['xbl'],\
                           params.getForEqs('pressflx')['xbr'],\
                           params.getForEqs('pressflx')['ybu'],\
                           params.getForEqs('pressflx')['ybd'],\
                           params.getForEqs('pressflx')['ilg'])
										  
										  
    def execPPflxEq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransPPflx =  fppx.PressureFluxEquation(params.getForProp('prop')['eht_data'],\
                                               params.getForProp('prop')['ig'],\
                                               params.getForProp('prop')['intc'],\
                                               tke_diss,\
                                               params.getForProp('prop')['prefix'])

									   
        ransPPflx.plot_fpp_equation(params.getForProp('prop')['laxis'],\
                                    params.getForEqs('ppflxeq')['xbl'],\
                                    params.getForEqs('ppflxeq')['xbr'],\
                                    params.getForEqs('ppflxeq')['ybu'],\
                                    params.getForEqs('ppflxeq')['ybd'],\
                                    params.getForEqs('ppflxeq')['ilg'])					
				
				
    def execPPvar(self):
						  
        params = self.params			
        tke_diss = 0.
        tauL = 1.
						  
        # instantiate 		
        ransPPvar =  sigmapp.PressureVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
					                                  params.getForProp('prop')['intc'],\
                                                      tke_diss,tauL,\
                                                      params.getForProp('prop')['prefix'])
								   
        ransPPvar.plot_sigma_pp(params.getForProp('prop')['laxis'],\
	                        params.getForEqs('pressvar')['xbl'],\
				params.getForEqs('pressvar')['xbr'],\
				params.getForEqs('pressvar')['ybu'],\
				params.getForEqs('pressvar')['ybd'],\
				params.getForEqs('pressvar')['ilg'])
										  
										  
    def execPPvarEq(self,tke_diss,tauL):
						  
        params = self.params						  
						  
        # instantiate 		
        ransPPvar =  sigmapp.PressureVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      tke_diss,tauL, \
                                                      params.getForProp('prop')['prefix'])

									   
        ransPPvar.plot_sigma_pp_equation(params.getForProp('prop')['laxis'],\
                     params.getForEqs('ppvareq')['xbl'],\
					 params.getForEqs('ppvareq')['xbr'],\
					 params.getForEqs('ppvareq')['ybu'],\
					 params.getForEqs('ppvareq')['ybd'],\
					 params.getForEqs('ppvareq')['ilg'])	
									  
				

				
    def execTT(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransTT =  tt.TemperatureEquation(params.getForProp('prop')['eht_data'],\
                                         params.getForProp('prop')['ig'],\
                                         params.getForProp('prop')['intc'],\
                                         tke_diss,\
                                         params.getForProp('prop')['prefix'])
								   
        ransTT.plot_tt(params.getForProp('prop')['laxis'],\
                       params.getForEqs('temp')['xbl'],\
                       params.getForEqs('temp')['xbr'],\
                       params.getForEqs('temp')['ybu'],\
                       params.getForEqs('temp')['ybd'],\
                       params.getForEqs('temp')['ilg'])
										  
										  
    def execTTeq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTT =  tt.TemperatureEquation(params.getForProp('prop')['eht_data'],\
                                         params.getForProp('prop')['ig'],\
                                         params.getForProp('prop')['intc'],\
                                         tke_diss,\
                                         params.getForProp('prop')['prefix'])

									   
        ransTT.plot_tt_equation(params.getForProp('prop')['laxis'],\
                                params.getForEqs('tteq')['xbl'],\
                                params.getForEqs('tteq')['xbr'],\
                                params.getForEqs('tteq')['ybu'],\
                                params.getForEqs('tteq')['ybd'],\
                                params.getForEqs('tteq')['ilg'])								  
	  	  
    def execTTvar(self):
						  
        params = self.params			
        tke_diss = 0.
        tauL = 1.
						  
        # instantiate 		
        ransTTvar =  sigmatt.TemperatureVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
					                                  params.getForProp('prop')['intc'],\
                                                      tke_diss,tauL,\
                                                      params.getForProp('prop')['prefix'])
								   
        ransTTvar.plot_sigma_tt(params.getForProp('prop')['laxis'],\
	                        params.getForEqs('tempvar')['xbl'],\
				params.getForEqs('tempvar')['xbr'],\
				params.getForEqs('tempvar')['ybu'],\
				params.getForEqs('tempvar')['ybd'],\
				params.getForEqs('tempvar')['ilg'])
										  
										  
    def execTTvarEq(self,tke_diss,tauL):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTTvar =  sigmatt.TemperatureVarianceEquation(params.getForProp('prop')['eht_data'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['intc'],\
                                                      tke_diss,tauL, \
                                                      params.getForProp('prop')['prefix'])

									   
        ransTTvar.plot_sigma_tt_equation(params.getForProp('prop')['laxis'],\
                     params.getForEqs('ttvareq')['xbl'],\
					 params.getForEqs('ttvareq')['xbr'],\
					 params.getForEqs('ttvareq')['ybu'],\
					 params.getForEqs('ttvareq')['ybd'],\
					 params.getForEqs('ttvareq')['ilg'])			  


    def execTTflx(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransTTflx =  fttx.TemperatureFluxEquation(params.getForProp('prop')['eht_data'],\
                                                  params.getForProp('prop')['ig'],\
                                                  params.getForProp('prop')['intc'],\
                                                  tke_diss,\
                                                  params.getForProp('prop')['prefix'])
								   
        ransTTflx.plot_ftt(params.getForProp('prop')['laxis'],\
                           params.getForEqs('tempflx')['xbl'],\
                           params.getForEqs('tempflx')['xbr'],\
                           params.getForEqs('tempflx')['ybu'],\
                           params.getForEqs('tempflx')['ybd'],\
                           params.getForEqs('tempflx')['ilg'])
										  
										  
    def execTTflxEq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTTflx =  fttx.TemperatureFluxEquation(params.getForProp('prop')['eht_data'],\
                                                  params.getForProp('prop')['ig'],\
                                                  params.getForProp('prop')['intc'],\
                                                  tke_diss,\
                                                  params.getForProp('prop')['prefix'])

									   
        ransTTflx.plot_ftt_equation(params.getForProp('prop')['laxis'],\
                                    params.getForEqs('ttflxeq')['xbl'],\
                                    params.getForEqs('ttflxeq')['xbr'],\
                                    params.getForEqs('ttflxeq')['ybu'],\
                                    params.getForEqs('ttflxeq')['ybd'],\
                                    params.getForEqs('ttflxeq')['ilg'])	
					 
		  
    def execHH(self):
						  
        params = self.params			
        tke_diss = 0.
						  
        # instantiate 		
        ransHH =  hh.EnthalpyEquation(params.getForProp('prop')['eht_data'],\
                                      params.getForProp('prop')['ig'],\
                                      params.getForProp('prop')['intc'],\
                                      tke_diss,\
                                      params.getForProp('prop')['prefix'])
								   
        ransHH.plot_hh(params.getForProp('prop')['laxis'],\
                       params.getForEqs('enth')['xbl'],\
                       params.getForEqs('enth')['xbr'],\
                       params.getForEqs('enth')['ybu'],\
                       params.getForEqs('enth')['ybd'],\
                       params.getForEqs('enth')['ilg'])
										  
										  
    def execHHeq(self,tke_diss):
						  
        params = self.params						  
						  
        # instantiate 		
        ransHH =  hh.EnthalpyEquation(params.getForProp('prop')['eht_data'],\
                                      params.getForProp('prop')['ig'],\
                                      params.getForProp('prop')['intc'],\
                                      tke_diss,\
                                      params.getForProp('prop')['prefix'])

									   
        ransHH.plot_hh_equation(params.getForProp('prop')['laxis'],\
                                params.getForEqs('hheq')['xbl'],\
                                params.getForEqs('hheq')['xbr'],\
                                params.getForEqs('hheq')['ybu'],\
                                params.getForEqs('hheq')['ybd'],\
                                params.getForEqs('hheq')['ilg'])			  
				
				
    def SetMatplotlibParams(self):
        """ This routine sets some standard values for matplotlib """ 
        """ to obtain publication-quality figures """

        # plt.rc('text',usetex=True)
        # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        plt.rc('font',**{'family':'serif','serif':['Times New Roman']})
        plt.rc('font',size=16.)
        plt.rc('lines',linewidth=2,markeredgewidth=2.,markersize=12)
        plt.rc('axes',linewidth=1.5)
        plt.rcParams['xtick.major.size']=8.
        plt.rcParams['xtick.minor.size']=4.
        plt.rcParams['figure.subplot.bottom']=0.15
        plt.rcParams['figure.subplot.left']=0.17		
        plt.rcParams['figure.subplot.right']=0.85
        plt.rcParams.update({'figure.max_open_warning': 0})		
				
				