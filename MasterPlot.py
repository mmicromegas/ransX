import EQUATIONS.ContinuityEquation as cont
import EQUATIONS.MomentumEquationX as momx
#import EQUATIONS.MomentumEquationY as momy
#import EQUATIONS.MomentumEquationZ as momz

#import EQUATIONS.ReynoldsStressXX as rxx N/A YET
#import EQUATIONS.ReynoldsStressYY as ryy N/A YET
#import EQUATIONS.ReynoldsStressZZ as rzz N/A YET
#import EQUATIONS.ReynoldsStressXY as rxy N/A YET
#import EQUATIONS.ReynoldsStressXZ as rxz N/A YET
#import EQUATIONS.ReynoldsStressYZ as ryz N/A YET

import EQUATIONS.TurbulentKineticEnergyEquation as tke
#import EQUATIONS.RadialTurbulentKineticEnergyEquation as rtke N/A YET
#import EQUATIONS.HorizontalTurbulentKineticEnergyEquation as htke N/A YET

import EQUATIONS.InternalEnergyEquation as ei
import EQUATIONS.InternalEnergyVarianceEquation as sigmaei
import EQUATIONS.InternalEnergyFluxEquation as feix

#import EQUATIONS.KineticEnergyEquation as ek N/A YET
#import EQUATIONS.TotalEnergyEquation as et N/A YET

import EQUATIONS.EntropyEquation as ss
import EQUATIONS.EntropyVarianceEquation as sigmass
import EQUATIONS.EntropyFluxEquation as fssx

#import EQUATIONS.PressureEquation as pp N/A YET
#import EQUATIONS.PressureVarianceEquation as sigmapp N/A YET

#import EQUATIONS.TemperatureEquation as tt N/A YET
#import EQUATIONS.TemperatureFluxEquation as fttx N/A YET
#import EQUATIONS.TemperatureVarianceEquation as sigmatt N/A YET

#import EQUATIONS.EnthalpyEquation as hh N/A YET
#import EQUATIONS.EnthalpyFluxEquation as sigmahh N/A YET

import EQUATIONS.DensityVarianceEquation as sigmadd
import EQUATIONS.TurbulentMassFluxEquation as a
import EQUATIONS.DensitySpecificVolumeCovarianceEquation as b

import EQUATIONS.XtransportEquation as xtra 
import EQUATIONS.XfluxEquation as xflx
import EQUATIONS.XvarianceEquation as xvar
import EQUATIONS.Xdiffusivity as xdiff

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


										  
    def execXflx(self,inuc,element,x):
	
    	params = self.params	

        # instantiate 		
        ransXflx = xflx.XfluxEquation(params.getForProp('prop')['eht_data'],\
	                              params.getForProp('prop')['ig'],\
	                              inuc,element,\
	                              params.getForProp('prop')['intc'],\
				      params.getForProp('prop')['prefix'])

        # plot Xrho									   
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

        # plot Xrho									   
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
	                                  params.getForEqs('momx')['xbl'],\
				 params.getForEqs('momx')['xbr'],\
				 params.getForEqs('momx')['ybu'],\
				 params.getForEqs('momx')['ybd'],\
				 params.getForEqs('momx')['ilg'])
										  

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
	                         params.getForEqs('mony')['xbl'],\
				 params.getForEqs('mony')['xbr'],\
				 params.getForEqs('mony')['ybu'],\
				 params.getForEqs('mony')['ybd'],\
				 params.getForEqs('mony')['ilg'])
										  

    def execMomyEq(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomy =  momy.MomentumEquationY(params.getForProp('prop')['eht_data'],\
	                                   params.getForProp('prop')['ig'],\
					   params.getForProp('prop')['intc'],\
					   params.getForProp('prop')['prefix'])
								   
        ransMomy.plot_momentum_equation_y(params.getForProp('prop')['laxis'],\
	                                  params.getForEqs('monyeq')['xbl'],\
					  params.getForEqs('monyeq')['xbr'],\
					  params.getForEqs('monyeq')['ybu'],\
					  params.getForEqs('monyeq')['ybd'],\
					  params.getForEqs('monyeq')['ilg'])
	
    def execMomz(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomz =  momz.MomentumEquationZ(params.getForProp('prop')['eht_data'],\
	                                   params.getForProp('prop')['ig'],\
					   params.getForProp('prop')['intc'],\
					   params.getForProp('prop')['prefix'])
								   
        ransMomz.plot_momentum_z(params.getForProp('prop')['laxis'],\
	                         params.getForEqs('monz')['xbl'],\
				 params.getForEqs('monz')['xbr'],\
				 params.getForEqs('monz')['ybu'],\
				 params.getForEqs('monz')['ybd'],\
				 params.getForEqs('monz')['ilg'])
										  

    def execMomzEq(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransMomz =  momz.MomentumEquationZ(params.getForProp('prop')['eht_data'],\
	                                   params.getForProp('prop')['ig'],\
					   params.getForProp('prop')['intc'],\
					   params.getForProp('prop')['prefix'])
								   
        ransMomz.plot_momentum_equation_z(params.getForProp('prop')['laxis'],\
	                                  params.getForEqs('monzeq')['xbl'],\
					  params.getForEqs('monzeq')['xbr'],\
					  params.getForEqs('monzeq')['ybu'],\
					  params.getForEqs('monzeq')['ybd'],\
					  params.getForEqs('monzeq')['ilg'])

									  
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
	                        params.getForEqs('eieq')['xbl'],\
				params.getForEqs('eieq')['xbr'],\
				params.getForEqs('eieq')['ybu'],\
				params.getForEqs('eieq')['ybd'],\
				params.getForEqs('eieq')['ilg'])
	
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
