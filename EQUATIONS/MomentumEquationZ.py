import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import CALCULUS as calc
import ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

# https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf

class MomentumEquationZ(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(MomentumEquationZ,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.data_prefix = data_prefix		

        # assign global data to be shared across whole class	
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('xzn0'))
        self.yzn0      = np.asarray(eht.item().get('yzn0'))		
        self.nx        = np.asarray(eht.item().get('nx')) 

        self.dd        = np.asarray(eht.item().get('dd')[intc])		
        self.pp        = np.asarray(eht.item().get('pp')[intc])
		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])
        self.dduy      = np.asarray(eht.item().get('dduy')[intc])
        self.dduz      = np.asarray(eht.item().get('dduz')[intc])		
		
        self.dduzux      = np.asarray(eht.item().get('dduzux')[intc])
        self.dduzuy      = np.asarray(eht.item().get('dduzuy')[intc])		
        self.dduzuz      = np.asarray(eht.item().get('dduzuz')[intc])		
		
        xzn0 = self.xzn0
        yzn0 = self.yzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_dduz    = np.asarray(eht.item().get('dduz')) 		

 		# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf	
		
        dd = self.dd
        ux = self.ux
        pp = self.pp
        ddux = self.ddux
        dduz = self.dduz		
        dduzux = self.dduzux
        dduzuy = self.dduzuy		
        dduzuz = self.dduzuz 		
		
        # construct equation-specific mean fields
        fht_ux = ddux/dd  		
        fht_uz = dduz/dd 		
        rzx = dduzux - dduz*ddux/dd
		
        #####################
        # Z MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_dduz = -self.dt(t_dduz,xzn0,t_timec,intc)
     
        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_uz = -self.Div(dd*fht_ux*fht_uz,xzn0)	 
		 
        # RHS -div ryz
        self.minus_div_ryz = -self.Div(ryz,xzn0)
		
        # RHS -G
        self.minus_G = -(dduzux/xzn0 + dduzuy_o_rtany))
				
        # -res
        self.minus_resResZmomentumEquation = \
          -(self.minus_dt_dduz + self.minus_div_eht_dd_fht_ux_fht_uz + self.minus_div_ryz \
            + self.minus_G)
		
        #########################
        # END Y MOMENTUM EQUATION 
        #########################		
		
    def plot_momentum_z(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot dduz stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.dduy
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
			
        # plot DATA 
        plt.title('dduy')
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho} \widetilde{u}_z$')
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{u}_y$ (g cm$^{-2}$ s$^{-1}$)"

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
		
        rhs0 = self.minus_div_ryx 
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
        plt.title('y momentum equation')
        plt.plot(grd1,lhs0,color='c',label = r"$-\partial_t ( \overline{\rho} \widetilde{u}_\theta ) $")
        plt.plot(grd1,lhs1,color='m',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{u}_r ) $")		
        plt.plot(grd1,rhs0,color='b',label=r"$-\nabla_r (\widetilde{R}_{\theta \phi})$")
        plt.plot(grd1,rhs1,color='g',label=r"$-\overline{G^{M}_\theta}$")
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-2}$  s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'momentum_z_eq.png')			
		



		
		