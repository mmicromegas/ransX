import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseContinuityEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(HsseContinuityEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0')) 	
        xznl = np.asarray(eht.item().get('xznl')) 
        xznr = np.asarray(eht.item().get('xznr')) 
		
        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	

        mm    = np.asarray(eht.item().get('mm')[intc])			
        dd    = np.asarray(eht.item().get('dd')[intc])
        ux    = np.asarray(eht.item().get('ux')[intc])			
        ddux  = np.asarray(eht.item().get('ddux')[intc])		
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 	
				
        # construct equation-specific mean fields
        fht_ux = ddux/dd		
        fdd = ddux-dd*ux	
	
        #####################
        # CONTINUITY EQUATION 
        #####################
				
        # LHS -gradx mm
        self.minus_gradx_mm = -self.Grad(mm,xzn0)
		
        # RHS +4 pi r^2 dd
        self.plus_four_pi_rsq_dd = +4.*np.pi*(xzn0**2.)*dd

        # scale factor +4 pi r^3/ 3 fht_ux
        self.plus_four_pi_rcu_o_three_fht_ux = 4.*np.pi*(xzn0**3)/(3.*fht_ux)        		

        # RHS -4 pi r^3/ 3 fht_ux Div fdd
        self.minus_four_pi_rcu_o_three_fht_ux_div_fdd = -self.plus_four_pi_rcu_o_three_fht_ux*self.Div(fdd,xzn0)
		
        # RHS +4 pi r^3/ 3 fht_ux fdd_o_dd gradx dd				
        self.plus_four_pi_rcu_o_three_fht_ux_fdd_o_dd_gradx_dd = +self.plus_four_pi_rcu_o_three_fht_ux*(fdd/dd)*self.Grad(dd,xzn0)		

        # RHS +4 pi r^3/ 3 fht_ux dd Div ux 
        self.minus_four_pi_rcu_o_three_fht_ux_dd_div_ux = -self.plus_four_pi_rcu_o_three_fht_ux*dd*self.Div(ux,xzn0) 
		
        # RHS -dq/dt 		
        self.minus_four_pi_rcu_o_three_fht_ux_dt_dd = -self.plus_four_pi_rcu_o_three_fht_ux*self.dt(t_dd,xzn0,t_timec,intc)

        # -res
        self.minus_resContEquation = -(self.minus_gradx_mm+self.plus_four_pi_rsq_dd+self.plus_four_pi_rcu_o_three_fht_ux+\
          self.minus_four_pi_rcu_o_three_fht_ux_div_fdd+self.plus_four_pi_rcu_o_three_fht_ux_fdd_o_dd_gradx_dd+\
           self.minus_four_pi_rcu_o_three_fht_ux_dd_div_ux+self.minus_four_pi_rcu_o_three_fht_ux_dt_dd)
		
        #########################	
        # END CONTINUITY EQUATION
        #########################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.dd        = dd	
		
		
    def plot_rho(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot rho stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.dd
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('density')
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho}$ (g cm$^{-3}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_rho.png')
	
    def plot_continuity_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot continuity equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_mm
		
        rhs0 = self.plus_four_pi_rcu_o_three_fht_ux		
        rhs1 = self.minus_four_pi_rcu_o_three_fht_ux_div_fdd
        rhs2 = self.plus_four_pi_rcu_o_three_fht_ux_fdd_o_dd_gradx_dd
        rhs3 = self.minus_four_pi_rcu_o_three_fht_ux_dd_div_ux
        rhs4 = self.minus_four_pi_rcu_o_three_fht_ux_dt_dd		
		
        res = self.minus_resContEquation
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,rhs1,rhs2,rhs3,rhs4,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('hsse continuity equation')
        plt.plot(grd1,lhs0,color='g',label = r'$-\partial_r (\overline{M})$')
        plt.plot(grd1,rhs0,color='r',label = r"$+4 \pi r^2 \overline{\rho}$")
        plt.plot(grd1,rhs1,color='c',label = r"$-(4 \pi r^3/3 \widetilde{u}_r) \nabla_r f_\rho$")		
        plt.plot(grd1,rhs2,color='m',label = r"$+(4 \pi r^3/3 \widetilde{u}_r) f_\rho / \overline{\rho} \partial_r \overline{\rho}$")
        plt.plot(grd1,rhs3,color='b',label=r"$-(4 \pi r^3/3 \widetilde{u}_r) \overline{\rho} \overline{d}$")
        plt.plot(grd1,rhs4,color='y',label=r"$-(4 \pi r^3/3 \widetilde{u}_r) \partial_t \overline{\rho}$")
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hsse_continuity_eq.png')