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

class MomentumEquationX(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(MomentumEquationX,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.data_prefix = data_prefix		

        # assign global data to be shared across whole class	
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('xzn0')) 
        self.nx        = np.asarray(eht.item().get('nx')) 

        self.dd        = np.asarray(eht.item().get('dd')[intc])
        self.ux        = np.asarray(eht.item().get('ux')[intc])	
        self.pp        = np.asarray(eht.item().get('pp')[intc])
        self.gg        = np.asarray(eht.item().get('gg')[intc])
		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])		

        self.dduxux      = np.asarray(eht.item().get('dduxux')[intc])
        self.dduyuy      = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz      = np.asarray(eht.item().get('dduzuz')[intc])		
		
        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddux    = np.asarray(eht.item().get('ddux')) 		

 	# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf	
		
        dd = self.dd
        ux = self.ux
        pp = self.pp
        gg = self.gg
        ddux = self.ddux
        dduxux = self.dduxux
        dduyuy = self.dduyuy
        dduzuz = self.dduzuz
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd
        rxx = dduxux - ddux*ddux/dd
		
        #####################
        # X MOMENTUM EQUATION 
        #####################

        # LHS -dq/dt 		
        self.minus_dt_ddux = -self.dt(t_ddux,xzn0,t_timec,intc)
     
        # LHS -div rho fht_ux fht_ux
        self.minus_div_eht_dd_fht_ux_fht_ux = -self.Div(dd*fht_ux*fht_ux,xzn0)	 
		
        # RHS -div rxx
        self.minus_div_rxx = -self.Div(rxx,xzn0)
		
        # RHS -G
        self.minus_G = -(dduyuy+dduzuz)/xzn0
		
        # RHS -(grad P - rho g)
        #self.minus_gradx_pp_eht_dd_eht_gg = self.Grad(pp,xzn0) - dd*gg
        self.minus_gradx_pp_eht_dd_eht_gg = np.zeros(self.nx)   		
		
        # -res
        self.minus_resResXmomentumEquation = \
          -(self.minus_dt_ddux + self.minus_div_eht_dd_fht_ux_fht_ux + self.minus_div_rxx \
            + self.minus_G + self.minus_gradx_pp_eht_dd_eht_gg)
		
        #########################
        # END X MOMENTUM EQUATION 
        #########################		
		
    def plot_momentum_x(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot ddux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        # load DATA to plot
        plt1 = self.ddux
        plt2 = self.ux
        #plt3 = self.vexp
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
			
        # plot DATA 
        plt.title('ddux')
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho} \widetilde{u}_x$')
        #plt.plot(grd1,plt2,color='green',label = r'$\overline{u}_x$')
        #plt.plot(grd1,plt3,color='red',label = r'$v_{exp}$')		

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{u}_x$ (g cm$^{-2}$ s$^{-1}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_ddux.png')
	
    def plot_momentum_equation_x(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot momentum x equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_ddux
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_ux
		
        rhs0 = self.minus_div_rxx 
        rhs1 = self.minus_G
        rhs2 = self.minus_gradx_pp_eht_dd_eht_gg
		
        res = self.minus_resResXmomentumEquation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('x momentum equation')
        plt.plot(grd1,lhs0,color='c',label = r"$-\partial_t ( \overline{\rho} \widetilde{u}_r ) $")
        plt.plot(grd1,lhs1,color='m',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{u}_r ) $")		
        plt.plot(grd1,rhs0,color='b',label=r"$-\nabla_r (\widetilde{R}_{rr})$")
        plt.plot(grd1,rhs1,color='g',label=r"$-\overline{G^{M}_r}$")
        plt.plot(grd1,rhs2,color='r',label=r"$-(\partial_r \overline{P} - \bar{\rho}\tilde{g}_r) \ 0$")		
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
        plt.savefig('RESULTS/'+self.data_prefix+'momentum_x_eq.png')			
		
