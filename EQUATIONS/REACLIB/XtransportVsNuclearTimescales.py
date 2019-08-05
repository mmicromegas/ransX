import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XtransportVsNuclearTimescales(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,bconv,tconv,tc,intc,data_prefix):
        super(XtransportVsNuclearTimescales,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	
        nx   = np.asarray(eht.item().get('nx'))		
		
        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		
        # assign global data to be shared across whole class

        dd     = np.asarray(eht.item().get('dd')[intc])
        ddxi   = np.asarray(eht.item().get('ddx'+inuc)[intc])
        ddux   = np.asarray(eht.item().get('ddux')[intc])
        ddxiux = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        ddxidot = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])
	
        # construct equation-specific mean fields
        fht_ux = ddux/dd
        fht_xi = ddxi/dd
        fxi    = ddxiux - ddxi*ddux/dd	

        # calculate damkohler number 		
        tau_trans = fht_xi/self.Div(fxi/dd,xzn0) 
        tau_nuc   = fht_xi/(ddxidot/dd)

        # Damkohler number		
        self.xda = tau_trans/tau_nuc
	
        self.data_prefix = data_prefix
        self.xzn0    = np.asarray(eht.item().get('xzn0')) 
        self.element = element
        self.inuc    = inuc
        self.bconv   = bconv
        self.tconv	 = tconv 
        self.tc    = tc		
		
        #self.tau_trans = np.abs(tau_trans)
        #self.tau_nuc = np.abs(tau_nuc)		
		
        self.tau_trans = tau_trans
        self.tau_nuc = tau_nuc			
		
    def plot_Xtimescales(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        # Damkohler number
	
        # convert nuc ID to string
        xnucid = str(self.inuc)		
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0				
	
        # get data
        plt0 = self.tau_trans
        plt1 = self.tau_nuc		
				
        # create FIGURE
        plt.figure(figsize=(7,6))
	
        # set plot boundaries   
        to_plot = [plt0,plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
	
        plt.yscale('symlog')	
	
        # plot DATA 		
        plt.title(r"$timescales \ for \ $"+self.element)
        #plt.semilogy(grd1,plt0,label=r"$-\tau_{trans}^i$",color='r') 
        #plt.semilogy(grd1,plt1,label=r"$-\tau_{nuc}^i$",color='b')

        xlimitrange = np.where((grd1 > self.bconv) & (grd1 < self.tconv))

        plt.plot(grd1[xlimitrange],plt0[xlimitrange],label=r"$\tau_{trans}^i$",color='r') 
        plt.plot(grd1[xlimitrange],plt1[xlimitrange],label=r"$\tau_{nuc}^i$",color='b')

        xlimitbottom = np.where(grd1 < self.bconv)
        plt.plot(grd1[xlimitbottom],plt0[xlimitbottom],'.',color='r',markersize=0.5)		
        plt.plot(grd1[xlimitbottom],plt1[xlimitbottom],'.',color='b',markersize=0.5)		
		
        xlimittop = np.where(grd1 > self.tconv)		
        plt.plot(grd1[xlimittop],plt0[xlimittop],'.',color='r',markersize=0.5)		
        plt.plot(grd1[xlimittop],plt1[xlimittop],'.',color='b',markersize=0.5)

        # oplot convective turnover timescale
        plt.axhline(y=self.tc, color='k', linestyle='--',linewidth=0.5)
        plt.text(self.tconv, self.tc, r"$\tau_{conv}$")		

        # convective boundary markers
        plt.axvline(self.bconv,linestyle='-',linewidth=0.7,color='k')		
        plt.axvline(self.tconv,linestyle='-',linewidth=0.7,color='k') 		
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\tau (s)$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':15})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'xTimescales_'+element+'.png')			

		
		
	
