import EQUATIONS.ContinuityEquation as cont
#import EQUATIONS.XmomentumEquation as xmom
#import EQUATIONS.YmomentumEquation as ymom N/A YET
#import EQUATIONS.ZmomentumEquation as zmom N/A YET

#import EQUATIONS.ReynoldsStressXX as rxx N/A YET
#import EQUATIONS.ReynoldsStressYY as ryy N/A YET
#import EQUATIONS.ReynoldsStressZZ as rzz N/A YET
#import EQUATIONS.ReynoldsStressXY as rxy N/A YET
#import EQUATIONS.ReynoldsStressXZ as rxz N/A YET
#import EQUATIONS.ReynoldsStressYZ as ryz N/A YET

import EQUATIONS.TurbulentKineticEnergyEquation as tke
#import EQUATIONS.RadialTurbulentKineticEnergyEquation as rtke N/A YET
#import EQUATIONS.HorizontalTurbulentKineticEnergyEquation as htke N/A YET

#import EQUATIONS.InternalEnergyEquation as ei
#import EQUATIONS.InternalEnergyVarianceEquation as sigmaei
#import EQUATIONS.InternalEnergyFluxEquation as feix

#import EQUATIONS.EntropyEquation as ss
#import EQUATIONS.EntropyVarianceEquation as sigmass
#import EQUATIONS.EntropyFluxEquation as fssx

#import EQUATIONS.PressureEquation as pp N/A YET
#import EQUATIONS.PressureVarianceEquation as sigmapp N/A YET

#import EQUATIONS.TemperatureEquation as tt N/A YET
#import EQUATIONS.TemperatureFluxEquation as fttx N/A YET
#import EQUATIONS.TemperatureVarianceEquation as sigmatt N/A YET

#import EQUATIONS.EnthalpyEquation as hh N/A YET
#import EQUATIONS.EnthalpyFluxEquation as sigmahh N/A YET

#import EQUATIONS.DensityVarianceEquation as sigmadd
#import EQUATIONS.TurbulentMassFluxEquation as a
#import EQUATIONS.DensitySpecificVolumeCovarianceEquation as b

import EQUATIONS.XtransportEquation as xtra 
#import EQUATIONS.XfluxEquation as xflx
#import EQUATIONS.XvarianceEquation as xvar
#import EQUATIONS.Xdiffusivity as xdif

#import EQUATIONS.ABARtransportEquation as abar N/A YET
#import EQUATIONS.ZBARtransportEquation as zbar N/A YET

import ReadParams as params

class MasterPlot():

    def __init__(self,params):

        self.params = params

    def execRho(self):
	
        params = self.params

        # instantiate 
        ransCONT = cont.ContinuityEquation(params.getForProp('prop')['eht_data'],\
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
        ransCONT = cont.ContinuityEquation(params.getForProp('prop')['eht_data'],\
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
        ransCONT = cont.ContinuityEquation(params.getForProp('prop')['eht_data'],\
	                                   params.getForProp('prop')['ig'],\
					                   params.getForProp('prop')['intc'],\
						               params.getForProp('prop')['prefix'])

        # plot continuity equation bar									   
        ransCONT.plot_continuity_equation_bar(params.getForProp('prop')['laxis'],\
	                                      params.getForEqsBar('conteqBar')['xbl'],\
									      params.getForEqsBar('conteqBar')['xbr'],\
									      params.getForEqsBar('conteqBar')['ybu'],\
									      params.getForEqsBar('conteqBar')['ybd'])

						
    def execXrho(self,inuc,element,x):
	
    	params = self.params	

        # instantiate 		
        ransXtra = xtra.XtransportEquation(params.getForProp('prop')['eht_data'],\
	                                   params.getForProp('prop')['ig'],\
	                                   inuc,element,\
	                                   params.getForProp('prop')['intc'],\
				                       params.getForProp('prop')['prefix'])

        # plot Xrho									   
        ransXtra.plot_Xrho(params.getForProp('prop')['laxis'],\
	                       params.getForEqs(x)['xbl'],\
		    	      	   params.getForEqs(x)['xbr'],\
			      		   params.getForEqs(x)['ybu'],\
		             	   params.getForEqs(x)['ybd'],\
		                   params.getForEqs(x)['ilg'])

    def execXtrs(self,inuc,element,x):

        params = self.params
								  									  
	    # instantiate 
        ransXtra = xtra.XtransportEquation(params.getForProp('prop')['eht_data'],\
	                                       params.getForProp('prop')['ig'],\
	                                       inuc,element,\
	    	    	                       params.getForProp('prop')['intc'],\
	     			                       params.getForProp('prop')['prefix'])
							
        ransXtra.plot_Xtransport_equation(params.getForProp('prop')['laxis'],\
	                                      params.getForEqs('xtrs_'+element)['xbl'],\
    	        						  params.getForEqs('xtrs_'+element)['xbr'],\
				        	    		  params.getForEqs('xtrs_'+element)['ybu'],\
					        	    	  params.getForEqs('xtrs_'+element)['ybd'],\
						            	  params.getForEqs('xtrs_'+element)['ilg'])	


    def execTke(self):
						  
        params = self.params			
        kolmrate = 0.		
						  
        # instantiate 		
        ransTke =  tke.TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],\
	                                   params.getForProp('prop')['ig'],\
					                   params.getForProp('prop')['intc'],\
								       -kolmrate,\
						               params.getForProp('prop')['prefix'])

        # plot continuity equation									   
        ransTke.plot_tke(params.getForProp('prop')['laxis'],\
	                     params.getForEqs('tkeeq')['xbl'],\
						 params.getForEqs('tkeeq')['xbr'],\
						 params.getForEqs('tkeeq')['ybu'],\
				     	 params.getForEqs('tkeeq')['ybd'],\
					     params.getForEqs('tkeeq')['ilg'])
										  
										  
    def execTkeEq(self,kolmrate):
						  
        params = self.params						  
						  
        # instantiate 		
        ransTke =  tke.TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],\
	                                   params.getForProp('prop')['ig'],\
					                   params.getForProp('prop')['intc'],\
								       -kolmrate,\
						               params.getForProp('prop')['prefix'])

        # plot continuity equation									   
        ransTke.plot_tke_equation(params.getForProp('prop')['laxis'],\
	                                  params.getForEqs('tkeeq')['xbl'],\
									  params.getForEqs('tkeeq')['xbr'],\
									  params.getForEqs('tkeeq')['ybu'],\
									  params.getForEqs('tkeeq')['ybd'],\
									  params.getForEqs('tkeeq')['ilg'])										  
										  
										  
