import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class VelocitiesMeanExp(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(VelocitiesMeanExp,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0')) 	
		
        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        ux     = np.asarray(eht.item().get('ux')[intc])		
        dd     = np.asarray(eht.item().get('dd')[intc])				
        ddux   = np.asarray(eht.item().get('ddux')[intc])
        dduxux = np.asarray(eht.item().get('dduxux')[intc])  		
		
        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))		
        t_mm    = np.asarray(eht.item().get('mm')) 		
		
        minus_dt_mm = -self.dt(t_mm,xzn0,t_timec,intc)
		
        vexp1 = ddux/dd		
        vexp2 = minus_dt_mm/(4.*np.pi*(xzn0**2.)*dd)
        vturb = ((dduxux - ddux*ddux/dd)/dd)**0.5		
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0  = xzn0
        self.ux    = ux
        self.ig    = ig		
        self.vexp1 = vexp1	
        self.vexp2 = vexp2			
        self.vturb = vturb	
		
    def plot_velocities(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot velocities in the model""" 
	
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ux
        plt2 = self.vexp1
        plt3 = self.vexp2
        plt4 = self.vturb
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2,plt3]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('velocities')
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{u}_r$')
        plt.plot(grd1,plt2,color='red',label = r'$\widetilde{u}_r$')
        plt.plot(grd1,plt3,color='green',linestyle='--',label = r'$\overline{v}_{exp} = -\dot{M}/(4 \pi r^2 \rho)$')		
        #plt.plot(grd1,plt4,color='blue',label = r'$u_{turb}$')
		
        if(self.ig == 1):			
            # define x LABEL
            setxlabel = r"x (cm)"		
        elif(self.ig == 2):
            # define x LABEL
            setxlabel = r"r (cm)"
        else:
            print("ERROR(VelocitiesMeanExp.py): geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 			
		
        # define y LABELS
        setylabel = r"velocity (cm s$^{-1}$)"

        # show x/y LABELS
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)
	
        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_velocities_mean.png')
	
	