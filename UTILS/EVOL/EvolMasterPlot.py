import EVOLUTION.TurbulentKineticEnergyEquationEvolution as tkeevol

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
        ransTkeEvol.plot_tke_evolution()

        # plot evolution of convection boundaries	   
        ransTkeEvol.plot_conv_bndry_location()
								
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
				
				
