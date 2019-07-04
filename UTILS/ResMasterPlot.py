import EQUATIONS.FOR_RESOLUTION_STUDY.BruntVaisallaResolutionStudy as bruntv

import matplotlib.pyplot as plt 

class ResMasterPlot():

    def __init__(self,params):

        self.params = params

    def execBruntV(self):
						  
        params = self.params						  
						  
        # instantiate 		
        ransBruntV =  bruntv.BruntVaisallaResolutionStudy(params.getForProp('prop')['eht_data'],\
                                           params.getForProp('prop')['ig'],\
                                           params.getForProp('prop')['intc'],\
                                           params.getForProp('prop')['prefix'])

									   
        ransBruntV.plot_bruntvaisalla(params.getForProp('prop')['laxis'],\
                                      params.getForEqs('nsq')['xbl'],\
                                      params.getForEqs('nsq')['xbr'],\
                                      params.getForEqs('nsq')['ybu'],\
                                      params.getForEqs('nsq')['ybd'],\
                                      params.getForEqs('nsq')['ilg'])			
		

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
				
				