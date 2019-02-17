import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class BruntVaisalla(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(BruntVaisalla,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/ 
		
        dd        = np.asarray(eht.item().get('dd')[intc]) 
        pp        = np.asarray(eht.item().get('pp')[intc]) 
        gg        = np.asarray(eht.item().get('gg')[intc])
        gamma1    = np.asarray(eht.item().get('gamma1')[intc])
		
        dlnrhodr  = self.deriv(np.log(dd),xzn0)
        dlnpdr    = self.deriv(np.log(pp),xzn0)
        dlnrhodrs = (1./gamma1)*dlnpdr
        nsq       = gg*(dlnrhodr-dlnrhodrs)
	
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.nsq         = nsq
		
    def plot_bruntvaisalla(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot BruntVaisalla parameter in the model""" 

        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.nsq
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('Brunt-Vaisalla frequency')
        plt.plot(grd1,plt1,color='brown',label = r'N$^2$')
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"N$^2$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_BruntVaisalla.png')
	
	