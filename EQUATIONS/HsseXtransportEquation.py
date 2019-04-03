import numpy as np
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseXtransportEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,intc,data_prefix):
        super(HsseXtransportEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd      = np.asarray(eht.item().get('dd')[intc])
        ddux    = np.asarray(eht.item().get('ddux')[intc])	
        ddxi    = np.asarray(eht.item().get('ddx'+inuc)[intc])
        ddxiux  = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        ddxidot = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])	
		
        ############################
        # HSSE Xi TRANSPORT EQUATION 
        ############################
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))
        t_dd      = np.asarray(eht.item().get('dd'))		
        t_ddxi    = np.asarray(eht.item().get('ddx'+inuc))			
        t_fht_xi  = t_ddxi/t_dd		
		
        # construct equation-specific mean fields
        fht_ux = ddux/dd
        fht_xi = ddxi/dd
        fxi    = ddxiux - ddxi*ddux/dd
		
        # LHS -dq/dt 		
        self.minus_dt_fht_xi = -self.dt(t_fht_xi,xzn0,t_timec,intc)
		
        # RHS +fht Xidot 
        self.plus_fht_xidot = +ddxidot/dd 		

        # RHS -(1/dd)div fxi 
        self.minus_one_o_dd_div_fxi = -(1./dd)*self.Div(fxi,xzn0)
		
        # LHS -fht_ux gradx fht_xi
        self.minus_div_eht_dd_fht_ux_fht_xi = -fht_ux*self.Grad(fht_xi,xzn0)
		
        # -res
        self.minus_resXiTransport = -(self.minus_dt_fht_xi+self.plus_fht_xidot+self.minus_one_o_dd_div_fxi+\
         self.minus_div_eht_dd_fht_ux_fht_xi)
		
        ################################		
        # END HSSE Xi TRANSPORT EQUATION
        ################################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0    = xzn0		
        self.inuc    = inuc
        self.element = element
        self.ddxi    = ddxi	
		
    def plot_Xrho(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xrho stratification in the model""" 

        # convert nuc ID to string
        #xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddxi
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title('rhoX for '+element)
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho} \widetilde{X}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_rhoX_'+element+'.png')
	
    def plot_Xtransport_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xrho transport equation in the model""" 

        # convert nuc ID to string
        #xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
				
        lhs0 = self.minus_dt_fht_xi 
		
        rhs0 = self.plus_fht_xidot
        rhs1 = self.minus_one_o_dd_div_fxi
        rhs2 = self.minus_div_eht_dd_fht_ux_fht_xi		

        res = self.minus_resXiTransport
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,rhs1,rhs2,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
				
        # plot DATA 
        #plt.title('hsse rhoX transport for '+element)
        plt.title(element)
        plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t \widetilde{X}$')
        plt.plot(grd1,rhs0,color='g',label=r'$+\widetilde{\dot{X}}^{\rm nuc}$')
        plt.plot(grd1,rhs1,color='b',label=r'$-(1/\overline{\rho}) \nabla_r f$')
        plt.plot(grd1,rhs2,color='y',label=r"$-\widetilde{u}_r \partial_r \widetilde{X}_\alpha$")

        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"s$^{-1}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hsse_mean_Xtransport_'+element+'.eps')