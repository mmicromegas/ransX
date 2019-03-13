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
	
        chim = np.asarray(eht.item().get('chim')[intc]) 
        chit = np.asarray(eht.item().get('chit')[intc]) 
        chid = np.asarray(eht.item().get('chid')[intc])
        mu   = np.asarray(eht.item().get('abar')[intc]) 		
        tt   = np.asarray(eht.item().get('tt')[intc])
        gamma2   = np.asarray(eht.item().get('gamma2')[intc])
		
        alpha = 1./chid
        delta = -chit/chid
        phi   = chid/chim
        hp    = -pp/self.Grad(pp,xzn0)  		
	
        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients		
        nabla = self.deriv(lntt,lnpp) 
        nabla_ad = (gamma2-1.)/gamma2
        nabla_mu = (chim/chit)*self.deriv(lnmu,lnpp)	
		
		# Kippenhahn and Weigert, p.42 but with opposite (minus) sign at the (phi/delta)*nabla_mu
        self.nsq_version2 = (gg*delta/hp)*(nabla_ad - nabla - (phi/delta)*nabla_mu) 		
	
    def plot_bruntvaisalla(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot BruntVaisalla parameter in the model""" 

        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.nsq
        plt2 = self.nsq_version2
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('Brunt-Vaisalla frequency')
        plt.plot(grd1,plt1,color='r',label = r'N$^2$')
        plt.plot(grd1,plt2,color='b',linestyle='--',label = r'N$^2$ version 2')
		
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
	
	