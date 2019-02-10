import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import CALCULUS as calc
import ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

# https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/

class TemperatureGradients(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(TemperatureGradients,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.data_prefix = data_prefix		

        # assign global data to be shared across whole class	
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('xzn0')) 
        self.nx      = np.asarray(eht.item().get('nx')) 		
		
        pp      = np.asarray(eht.item().get('pp')[intc]) 		
        tt      = np.asarray(eht.item().get('tt')[intc]) 
        mu      = np.asarray(eht.item().get('abar')[intc]) 		
        chim    = np.asarray(eht.item().get('chim')[intc]) 		
        chit    = np.asarray(eht.item().get('chit')[intc]) 	
        gamma2    = np.asarray(eht.item().get('gamma2')[intc]) 					
				
        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)
		
        self.nabla = self.deriv(lntt,lnpp) 
        self.nabla_ad = (gamma2-1.)/gamma2
        self.nabla_mu = (chim/chit)*self.deriv(lnmu,lnpp)
		
    def plot_nablas(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot temperature gradients in the model""" 

        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.nabla
        plt2 = self.nabla_ad
        plt3 = self.nabla_mu
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2,plt3]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('temperature gradients')
        plt.plot(grd1,plt1,color='brown',label = r'$\nabla$')
        plt.plot(grd1,plt2,color='red',label = r'$\nabla_{ad}$')
        plt.plot(grd1,plt3,color='green',label = r'$\nabla_{\mu}$')		
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\nabla$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)
		
        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_nablas.png')
	
	