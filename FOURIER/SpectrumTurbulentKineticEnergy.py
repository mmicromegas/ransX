import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.PROMPI_data as pd

class SpectrumTurbulentKineticEnergy():

    def __init__(self,filename,data_prefix,lhc):
	
        block = pd.PROMPI_bindata(filename,['velx','vely','velz'])

        xzn0 = block.datadict['xzn0']
        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']        

        xlm = np.abs(np.asarray(xzn0)-np.float(lhc))
        ilhc = int(np.where(xlm==xlm.min())[0][0])

        print('in SpectrumTurbulentKineticEnergy.py',ilhc)

        self.xzn0 = xzn0
        
    def plot_TKEspectrum(self):
        """Plot TKE spectrum"""
	
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
 						
        # plot DATA 
        #plt.title('turbulent kinetic energy')
        #plt.plot(grd1,plt1,color='brown',label = r"$\frac{1}{2} \widetilde{u''_i u''_i}$")
     
        #setxlabel = r'x (10$^{8}$ cm)'	
        #setylabel = r"$\widetilde{k}$ (erg g$^{-1}$)"
        
        #plt.xlabel(setxlabel)
        #plt.ylabel(setylabel)
		
        # show LEGEND
        #plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        #plt.show(block=False)

        # save PLOT
        #plt.savefig('RESULTS/'+self.data_prefix+'tkespectrum.png')		


        
