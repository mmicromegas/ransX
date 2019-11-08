import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class MomentumEquationZ(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(MomentumEquationZ,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])		
        pp = np.asarray(eht.item().get('pp')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])
		
        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])		
		
        dduxuz     = np.asarray(eht.item().get('dduxuz')[intc])
        dduzuycoty = np.asarray(eht.item().get('dduzuycoty')[intc])
		
        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))		
        t_dd    = np.asarray(eht.item().get('dd')) 
        t_dduz  = np.asarray(eht.item().get('dduz')) 		
		
        # construct equation-specific mean fields
        fht_ux = ddux/dd  		
        fht_uz = dduz/dd 		
        rzx    = dduxuz - ddux*dduz/dd
		
        #####################
        # Z MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_dduz = -self.dt(t_dduz,xzn0,t_timec,intc)
     
        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_uz = -self.Div(dd*fht_ux*fht_uz,xzn0)	 
		 
        # RHS -div rzx
        self.minus_div_rzx = -self.Div(rzx,xzn0)
		
        # RHS -G
        self.minus_G = -(dduxuz + dduzuycoty)/xzn0
	
        # -res
        self.minus_resResZmomentumEquation = \
          -(self.minus_dt_dduz + self.minus_div_eht_dd_fht_ux_fht_uz + self.minus_div_rzx \
            + self.minus_G)
		
        #########################
        # END Z MOMENTUM EQUATION 
        #########################		
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.dduz        = dduz		
		
    def plot_momentum_z(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot dduz stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.dduz
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
			
        # plot DATA 
        plt.title('dduz')
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho} \widetilde{u}_z$')
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{u}_z$ (g cm$^{-2}$ s$^{-1}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_dduz.png')
	
    def plot_momentum_equation_z(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot momentum z equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dduz
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_uz
		
        rhs0 = self.minus_div_rzx 
        rhs1 = self.minus_G
 		
        res = self.minus_resResZmomentumEquation
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))					
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('z momentum equation')
        if (self.ig == 1):					
            plt.plot(grd1,lhs0,color='c',label = r"$-\partial_t ( \overline{\rho} \widetilde{u}_z ) $")
            plt.plot(grd1,lhs1,color='m',label = r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{u}_z ) $")		
            plt.plot(grd1,rhs0,color='b',label=r"$-\nabla_x (\widetilde{R}_{zx})$")
            #plt.plot(grd1,rhs1,color='g',label=r"$-\overline{G^{M}_\phi}$")
            plt.plot(grd1,lhs0+lhs1+rhs0,color='k',linestyle='--',label='res')
            setxlabel = r"x (cm)"
        elif(self.ig == 2):  
            plt.plot(grd1,lhs0,color='c',label = r"$-\partial_t ( \overline{\rho} \widetilde{u}_\phi ) $")
            plt.plot(grd1,lhs1,color='m',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{u}_\phi ) $")		
            plt.plot(grd1,rhs0,color='b',label=r"$-\nabla_r (\widetilde{R}_{\phi r})$")
            plt.plot(grd1,rhs1,color='g',label=r"$-\overline{G^{M}_\phi}$")
            plt.plot(grd1,res,color='k',linestyle='--',label='res')
            setxlabel = r"r (cm)"			
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()		


        # define and show x/y LABELS
        setylabel = r"g cm$^{-2}$  s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'momentum_z_eq.png')			
		



		
		
