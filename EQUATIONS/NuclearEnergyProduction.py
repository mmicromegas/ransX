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

class NuclearEnergyProduction(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(NuclearEnergyProduction,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.data_prefix = data_prefix		

        # assign global data to be shared across whole class	
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('xzn0')) 
        self.nx      = np.asarray(eht.item().get('nx')) 
		
        self.enuc        = np.asarray(eht.item().get('enuc1')[intc]) + \
                           np.asarray(eht.item().get('enuc2')[intc])
		
    def plot_enuc(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot nuclear energy production stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
 
        to_plt1 = self.enuc
	
        # calculate indices of grid boundaries 
        xzn0 = np.asarray(self.xzn0)
        xlm = np.abs(xzn0-xbl)
        xrm = np.abs(xzn0-xbr)
        idxl = int(np.where(xlm==xlm.min())[0][0])
        idxr = int(np.where(xrm==xrm.min())[0][0])	
	
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        plt.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
		
        # plot DATA 
        plt.title('Nuclear energy production')
        plt.plot(grd1,to_plt1,color='brown',label = r'$\overline{\varepsilon_{nuc}}$')
		
        # define and show x/y LABELS
        setxlabel = r'r (10$^{8}$ cm)'
        setylabel = r'log $\overline{\varepsilon_{enuc}}$ (erg s$^{-1}$)'		

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_enuc.png')
	
	