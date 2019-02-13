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

class PressureInternalEnergy(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(PressureInternalEnergy,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.data_prefix = data_prefix		

        # assign global data to be shared across whole class	
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('xzn0')) 
        self.nx      = np.asarray(eht.item().get('nx')) 
		
        self.pp        = np.asarray(eht.item().get('pp')[intc])
        self.ei        = np.asarray(eht.item().get('ei')[intc])
		
    def plot_ppei(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot pressure and internal energy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
 
        to_plt1 = np.log10(self.pp)
        to_plt2 = np.log10(self.ei)
	
        xlabel_1 = r'r (10$^{8}$ cm)'
		
        ylabel_1 = r'log $\overline{P}$ (erg cm$^{-3}$)'
        ylabel_2 = r'log $\overline{\epsilon}$ (ergs)'
	
        plabel_1 = r'$\overline{P}$'
        plabel_2 = r'$\overline{\epsilon}$'
	
        # calculate indices of grid boundaries 
        xzn0 = np.asarray(self.xzn0)
        xlm = np.abs(xzn0-xbl)
        xrm = np.abs(xzn0-xbr)
        idxl = int(np.where(xlm==xlm.min())[0][0])
        idxr = int(np.where(xrm==xrm.min())[0][0])	
	
        # create FIGURE	
        fig, ax1 = plt.subplots(figsize=(7,6))
		
        ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.plot(xzn0,to_plt1,color='b',label = plabel_1)

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7,prop={'size':18})

        ax2 = ax1.twinx()
        ax2.axis([xbl,xbr,np.min(to_plt2[idxl:idxr]),np.max(to_plt2[idxl:idxr])])
        ax2.plot(xzn0, to_plt2,color='r',label = plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1,prop={'size':18})
		
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_ppei.png')
	