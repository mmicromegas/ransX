import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al
import os
import sys


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class VelocitiesMLTturb(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,data_prefix):
        super(VelocitiesMLTturb,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0')) 	
		
        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        ux     = np.asarray(eht.item().get('ux')[intc])		
        dd     = np.asarray(eht.item().get('dd')[intc])				
        tt     = np.asarray(eht.item().get('tt')[intc])	
        hh     = np.asarray(eht.item().get('hh')[intc])	
        cp     = np.asarray(eht.item().get('cp')[intc])
        gg     = np.asarray(eht.item().get('gg')[intc])
        pp     = np.asarray(eht.item().get('pp')[intc])
        chit     = np.asarray(eht.item().get('chit')[intc])
        chid     = np.asarray(eht.item().get('chid')[intc])		
        ddux   = np.asarray(eht.item().get('ddux')[intc])
        dduxux = np.asarray(eht.item().get('dduxux')[intc])  		
        ddtt   = np.asarray(eht.item().get('ddtt')[intc])
        ddhh   = np.asarray(eht.item().get('ddhh')[intc])
        ddcp   = np.asarray(eht.item().get('ddcp')[intc])
        ddhhux = np.asarray(eht.item().get('ddhhux')[intc])
        hhux = np.asarray(eht.item().get('hhux')[intc])
        ttsq = np.asarray(eht.item().get('ttsq')[intc])
        ddttsq = np.asarray(eht.item().get('ddttsq')[intc])
        gamma2 = np.asarray(eht.item().get('gamma2')[intc])

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if(ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])   
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma2 = cp/cv   # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
        
        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))		
        t_mm    = np.asarray(eht.item().get('mm')) 		
		
        minus_dt_mm = -self.dt(t_mm,xzn0,t_timec,intc)
		
        vexp1 = ddux/dd		
        vexp2 = minus_dt_mm/(4.*np.pi*(xzn0**2.)*dd)
        vturb = ((dduxux - ddux*ddux/dd)/dd)**0.5

        fht_cp = ddcp/dd 		

        # variance of temperature fluctuations		
        #sigmatt = (ddttsq-ddtt*ddtt/dd)/dd			
        sigmatt = ttsq-tt*tt		
		
        # T_rms fluctuations
        tt_rms = sigmatt**0.5			
		
        # enthalpy flux 
        fhh = ddhhux - ddhh*ddux/dd	
        #fhh = dd*(hhux - hh*ux)		
		
        # mlt velocity		
        alphae = 0.7 # Meakin,Arnett,2007			
        vmlt_1 = fhh/(alphae*dd*fht_cp*tt_rms)			
		
        Hp = 2.e8 # this is for oburn
        alpha_mlt = 1.7		
        lbd = alpha_mlt*Hp		
		
        lntt = np.log(tt)
        lnpp = np.log(pp)		
		
        # calculate temperature gradients		
        nabla = self.deriv(lntt,lnpp) 
        nabla_ad = (gamma2-1.)/gamma2		
		
        betaT = -chit/chid		
		
        vmlt_2 = gg*betaT*(nabla-nabla_ad)*((lbd**2.)/(8.*Hp))
        vmlt_2 = vmlt_2.clip(min=1.) # get rid of negative values, set to min 1.		
        vmlt_2 = (vmlt_2)**0.5		

        # this should be OS independent
        dir_model = os.path.join(os.path.realpath('.'),'DATA','INIMODEL', 'imodel.tycho')
        
        data = np.loadtxt(dir_model,skiprows=26)		
        nxmax = 500
		
        rr = data[1:nxmax,2]
        vmlt_3 = data[1:nxmax,8]
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0  = xzn0
        self.ux    = ux		
        self.vexp1 = vexp1	
        self.vexp2 = vexp2			
        self.vturb = vturb	
        self.vmlt_1 = vmlt_1
        self.vmlt_2 = vmlt_2

        self.rr = rr
        self.vmlt_3 = vmlt_3
		
    def plot_velocities(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot velocities in the model""" 
	
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ux
        plt2 = self.vexp1
        plt3 = self.vexp2
        plt4 = self.vturb
        plt5 = self.vmlt_1
        plt6 = self.vmlt_2
        plt7 = self.vmlt_3		

        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # temporary hack
        plt4 = np.nan_to_num(plt4)
        plt5 = np.nan_to_num(plt5)
        plt6 = np.nan_to_num(plt6)
        plt7 = np.nan_to_num(plt7)
        
        # set plot boundaries   
        to_plot = [plt4,plt5,plt6,plt7]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('velocities')
        #plt.plot(grd1,plt1,color='brown',label = r'$\overline{u}_r$')
        #plt.plot(grd1,plt2,color='red',label = r'$\widetilde{u}_r$')
        #plt.plot(grd1,plt3,color='green',linestyle='--',label = r'$\overline{v}_{exp} = -\dot{M}/(4 \pi r^2 \rho)$')		
        plt.plot(grd1,plt4,color='blue',label = r'$u_{turb}$')
#        plt.plot(grd1,plt5,color='red',label = r'$u_{MLT} 1$')
        #plt.plot(grd1,plt6,color='g',label = r'$u_{MLT} 2$')
        #plt.plot(self.rr,plt7,color='brown',label = r'$u_{MLT} 3 inimod$')
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"velocity (cm s$^{-1}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)
	
        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_velocities_turb.png')
	
	
