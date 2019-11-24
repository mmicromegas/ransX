import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al
import UTILS.EVOL.PropertiesEvolution as propevol
import UTILS.EVOL.EvolReadParams as rp

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentKineticEnergyEquationEvolution(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,dataout,ig,data_prefix):
        super(TurbulentKineticEnergyEquationEvolution,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(dataout)		

        # load grid
        xznr    = np.asarray(eht.item().get('xznr')) 
        xznl    = np.asarray(eht.item().get('xznl'))

        # load temporal evolution
        t_timec    = np.asarray(eht.item().get('t_timec')) 	
        t_TKEsum      = np.asarray(eht.item().get('t_TKEsum'))		
        t_xzn0inc  = np.asarray(eht.item().get('t_xzn0inc'))
        t_xzn0outc = np.asarray(eht.item().get('t_xzn0outc'))		
         
        # share data across the whole class
        self.t_timec = t_timec
        self.t_TKEsum = t_TKEsum
        self.t_xzn0inc = t_xzn0inc
        self.t_xzn0outc = t_xzn0outc 
        self.xznr = xznr
        self.xznl = xznl
        self.data_prefix = data_prefix	

    def plot_tke_evolution(self):

        xznr = self.xznr
        xznl = self.xznl

        grd1 = self.t_timec
        plt1 = self.t_TKEsum	

        # handle volume for different geometries
        #if (self.ig == 1):	
	    #Vol = xznr**3-xznl**3
        #elif (self.ig == 2):	
        #    Vol = 4./3.*np.pi*(xznr**3-xznl**3)
        #else:
        #    print("ERROR (TurbulentKineticEnergyEquationEvolution.py): geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
        #    sys.exit()   

        # Calculate 
        #tke_int = np.zeros(grd1.size)
        #for i in range(0,grd1.size):
        #    dd = self.t_dd[i,:]
        #    tke = self.t_tke[i,:]
        #    tke_int[i] = (dd*tke*Vol).sum()
		
        # create FIGURE
        plt.figure(figsize=(7,6))

        plt.axis([0.,1500.,0.,1.e47])	
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # plot DATA 
        plt.title('turbulent kinetic energy evolution')
        plt.plot(grd1,plt1,color='r',label = r'$tke$')		
		
        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"ergs"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'tke_evol.png')				
		  
    def plot_conv_bndry_location(self):

        # get data 
        t_timec = self.t_timec
        t_xzn0inc = self.t_xzn0inc
        t_xzn0outc = self.t_xzn0outc 

        # create FIGURE
        plt.figure(figsize=(7,6))

        plt.axis([0.,1500.,0.,1.e9])	
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # plot DATA 
        plt.title('convection boundary')
        plt.plot(t_timec,t_xzn0inc,color='r',label = r'$inner$')		
        plt.plot(t_timec,t_xzn0outc,color='g',label = r'$outer$')	
        plt.plot(t_timec,t_xzn0outc-t_xzn0inc,color='b',label = r'$l_c$')
		
        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"cm"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'cnvzboundary_evol.png')			
			
			