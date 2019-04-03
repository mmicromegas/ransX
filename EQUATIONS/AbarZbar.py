import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class AbarZbar(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(AbarZbar,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        self.data_prefix = data_prefix		
		
        # load grid	
        self.xzn0 = np.asarray(eht.item().get('xzn0')) 		
		
        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	
		
        self.abar = np.asarray(eht.item().get('abar')[intc]) 
        self.zbar = np.asarray(eht.item().get('zbar')[intc]) 

    def plot_abarzbar(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot abarzbar in the model""" 

        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.abar
        plt2 = self.zbar
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('abar zbar')
        plt.plot(grd1,plt1,color='r',label = r"$\overline{A}$")
        plt.plot(grd1,plt2,color='b',label = r"$\overline{Z}$")		
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{Z},\overline{A}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_abarzbar.png')
	
	