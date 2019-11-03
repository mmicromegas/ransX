import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TemperatureGradients(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,data_prefix):
        super(TemperatureGradients,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0'))
        nx   = np.asarray(eht.item().get('nx')) 	

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		
		
        pp      = np.asarray(eht.item().get('pp')[intc]) 		
        tt      = np.asarray(eht.item().get('tt')[intc]) 
        mu      = np.asarray(eht.item().get('abar')[intc]) 		
        chim    = np.asarray(eht.item().get('chim')[intc]) 		
        chit    = np.asarray(eht.item().get('chit')[intc]) 	
        gamma2    = np.asarray(eht.item().get('gamma2')[intc]) 					

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if(ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])   
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma2 = cp/cv   # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
        
        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients		
        nabla = self.deriv(lntt,lnpp) 
        nabla_ad = (gamma2-1.)/gamma2

        if(ieos == 1):        
            nabla_mu = np.zeros(nx)
        else:
            nabla_mu = (chim/chit)*self.deriv(lnmu,lnpp)
            
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0  = xzn0
        self.nabla = nabla
        self.nabla_ad = nabla_ad
        self.nabla_mu = nabla_mu
		
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
	
	
