import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.EVOL.ALIMITevol as al
import UTILS.EVOL.PropertiesEvolution as propevol
import UTILS.EVOL.EvolReadParams as rp

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class X0002Evolution(calc.CALCULUS,al.ALIMITevol,object):

    def __init__(self,dataout,ig,data_prefix):
        super(X0002Evolution,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(dataout)		

        # load temporal evolution
        t_timec    = np.asarray(eht.item().get('t_timec')) 	
        t_x0002mean_cnvz = np.asarray(eht.item().get('t_x0002mean_cnvz'))
         
        # share data across the whole class
        self.t_timec = t_timec
        self.data_prefix = data_prefix	

        self.t_x0002mean_cnvz = t_x0002mean_cnvz		 	
			
    def plot_x0002(self,LAXIS,xbl,xbr,ybu,ybd,ilg):

        # get data 
        grd1 = self.t_timec
        plt1 = self.t_x0002mean_cnvz

        tc = 300.
        x0002tc = np.interp(tc,grd1,plt1)
        print("x0002: ",x0002tc)

        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
		
        # plot DATA 
        plt.title(r'bottom 2/3 of cnvz')
        plt.plot(grd1,plt1,color='r',label = r'$X_2$')			
		
        # define and show x/y LABELS
        setxlabel = r"t (s)"
        setylabel = r"avg(X2)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'x0002mean_evol.png')

	