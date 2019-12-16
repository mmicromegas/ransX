import EVOLUTION.TurbulentKineticEnergyEquationEvolution as tkeevol
import EVOLUTION.ConvectionBoundariesPositionEvolution as cnvzpos
import EVOLUTION.EnergySourceTermEvolution as enesrc
import EVOLUTION.X0002Evolution as x2evol


import matplotlib.pyplot as plt 

class EvolMasterPlot():

    def __init__(self,params):

        self.params = params

    def execEvolTKE(self):
						  
        params = self.params			
						  
        # instantiate 		
        ransTkeEvol =  tkeevol.TurbulentKineticEnergyEquationEvolution(params.getForProp('prop')['dataout'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['prefix'])
										  
        # plot turbulent kinetic energy evolution	   
        ransTkeEvol.plot_tke_evolution(params.getForProp('prop')['laxis'],\
                          params.getForEvol('tkeevol')['xbl'],\
                          params.getForEvol('tkeevol')['xbr'],\
                          params.getForEvol('tkeevol')['ybu'],\
                          params.getForEvol('tkeevol')['ybd'],\
                          params.getForEvol('tkeevol')['ilg'])

								
    def execEvolCNVZbnry(self):
						  
        params = self.params			
						  
        # instantiate 		
        ransCnvzPositionEvol =  cnvzpos.ConvectionBoundariesPositionEvolution(params.getForProp('prop')['dataout'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['prefix'])
										
        # plot evolution of convection boundaries	   
        ransCnvzPositionEvol.plot_conv_bndry_location(params.getForProp('prop')['laxis'],\
                          params.getForEvol('cnvzbndry')['xbl'],\
                          params.getForEvol('cnvzbndry')['xbr'],\
                          params.getForEvol('cnvzbndry')['ybu'],\
                          params.getForEvol('cnvzbndry')['ybd'],\
                          params.getForEvol('cnvzbndry')['ilg'])

    def execEvolTenuc(self):
						  
        params = self.params			
						  
        # instantiate 		
        ransTenucEvol =  enesrc.EnergySourceTermEvolution(params.getForProp('prop')['dataout'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['prefix'])
										  
        # plot total energy source	   
        ransTenucEvol.plot_tenuc_evolution(params.getForProp('prop')['laxis'],\
                          params.getForEvol('enesource')['xbl'],\
                          params.getForEvol('enesource')['xbr'],\
                          params.getForEvol('enesource')['ybu'],\
                          params.getForEvol('enesource')['ybd'],\
                          params.getForEvol('enesource')['ilg'])
							
    def execEvolX0002(self):
						  
        params = self.params			
						  
        # instantiate 		
        ransX0002Evol =  x2evol.X0002Evolution(params.getForProp('prop')['dataout'],\
                                                      params.getForProp('prop')['ig'],\
                                                      params.getForProp('prop')['prefix'])
										
        # plot evolution of x0002	   
        ransX0002Evol.plot_x0002(params.getForProp('prop')['laxis'],\
                          params.getForEvol('x0002evol')['xbl'],\
                          params.getForEvol('x0002evol')['xbr'],\
                          params.getForEvol('x0002evol')['ybu'],\
                          params.getForEvol('x0002evol')['ybd'],\
                          params.getForEvol('x0002evol')['ilg'])

								
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
				
				