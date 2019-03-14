import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseMomentumEquationX(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(HsseMomentumEquationX,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	
		
        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])	
        pp = np.asarray(eht.item().get('pp')[intc])
        gg = np.asarray(eht.item().get('gg')[intc])
		
        ddux = np.asarray(eht.item().get('ddux')[intc])		

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])		
		
        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))		
        t_dd    = np.asarray(eht.item().get('dd')) 
        t_ddux  = np.asarray(eht.item().get('ddux')) 		
        t_fht_ux  = t_ddux/t_dd 
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd
        rxx = dduxux - ddux*ddux/dd
		
        ##########################
        # HSSE X MOMENTUM EQUATION 
        ##########################
 
        # LHS -gradx p
        self.minus_gradx_pp = -self.Grad(pp,xzn0)		
		
        # RHS + dd gg		
        self.plus_dd_gg = +dd*gg

        # RHS -dd dt fht_ux 		
        self.minus_dd_dt_fht_ux = -dd*self.dt(t_fht_ux,xzn0,t_timec,intc)		

        # RHS -div rxx
        self.minus_div_rxx = -self.Div(rxx,xzn0)		
		
        # RHS -G
        self.minus_G = -(-dduyuy-dduzuz)/xzn0
     
        # RHS -dd fht_ux gradx fht_ux
        self.minus_dd_fht_ux_gradx_fht_ux = -dd*fht_ux*self.Grad(fht_ux,xzn0)  		
		
        # -res
        self.minus_resResXmomentumEquation = \
          -(self.minus_gradx_pp+self.plus_dd_gg+self.minus_dd_dt_fht_ux+self.minus_div_rxx+\
            self.minus_G+self.minus_dd_fht_ux_gradx_fht_ux)
		
        ##############################
        # END HSSE X MOMENTUM EQUATION 
        ##############################	

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.ddux        = ddux
        self.ux          = ux      		
		
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

        lhs0 = self.minus_gradx_pp
		
        rhs0 = self.plus_dd_gg
        rhs1 = self.minus_dd_dt_fht_ux
        rhs2 = self.minus_div_rxx
        rhs3 = self.minus_G 
        rhs4 = self.minus_dd_fht_ux_gradx_fht_ux		
	
        res = self.minus_resResXmomentumEquation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,rhs1,rhs2,rhs3,rhs4,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('hsse x momentum equation')
        plt.plot(grd1,lhs0,color='c',label = r"$-\partial_r \overline{P} $")
        plt.plot(grd1,rhs0,color='m',label = r"$-\overline{\rho} \widetilde{g}_r$")
        plt.plot(grd1,rhs1,color='r',label = r"$-\overline{\rho} \partial_t \widetilde{u}_r$")		
        plt.plot(grd1,rhs2,color='b',label=r"$-\nabla_r (\widetilde{R}_{rr})$")
        plt.plot(grd1,rhs3,color='g',label=r"$-\overline{G^{M}_r}$")
        plt.plot(grd1,rhs4,color='y',label=r"$-\overline{\rho} \widetilde{u}_r \partial_r \widetilde{u}_r$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$  cm$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hsse_momentum_x_eq.png')			
		
