import numpy as np
import matplotlib.pyplot as plt
import CALCULUS as calc
import ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

# https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/

class XfluxEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,intc,data_prefix):
        super(XfluxEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.data_prefix = data_prefix
        self.inuc = inuc
        self.element = element
		
        # assign global data to be shared across whole class
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('xzn0')) 

        self.dd        = np.asarray(eht.item().get('dd')[intc])
        self.ux        = np.asarray(eht.item().get('ux')[intc])	
        self.pp        = np.asarray(eht.item().get('pp')[intc])
        self.xi        = np.asarray(eht.item().get('x'+inuc)[intc])	
		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])
        self.dduy      = np.asarray(eht.item().get('dduy')[intc])
        self.dduz      = np.asarray(eht.item().get('dduz')[intc])

        self.dduxux    = np.asarray(eht.item().get('dduxux')[intc])
        self.dduyuy    = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz    = np.asarray(eht.item().get('dduzuz')[intc])
	
        self.ddxi      = np.asarray(eht.item().get('ddx'+inuc)[intc])
        self.ddxiux    = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        self.ddxidot   = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])	
	 		
        self.ddxidotux = np.asarray(eht.item().get('ddx'+inuc+'dotux')[intc]) 	
        self.ddxiuxux  = np.asarray(eht.item().get('ddx'+inuc+'uxux')[intc])		
        self.ddxiuyuy  = np.asarray(eht.item().get('ddx'+inuc+'uyuy')[intc])
        self.ddxiuzuz  = np.asarray(eht.item().get('ddx'+inuc+'uzuz')[intc])		

        self.xigradxpp = np.asarray(eht.item().get('x'+inuc+'gradxpp')[intc]) 
		
        # store time series for time derivatives
        self.t_timec   = np.asarray(eht.item().get('timec')) 
        self.t_dd      = np.asarray(eht.item().get('dd')) 
        self.t_ddux    = np.asarray(eht.item().get('ddux')) 
        self.t_ddxi    = np.asarray(eht.item().get('ddx'+inuc))		
        self.t_ddxiux  = np.asarray(eht.item().get('ddx'+inuc+'ux'))
				
        ##################
        # Xi FLUX EQUATION 
        ##################		
 
 	# pick equation-specific Reynolds-averaged mean fields according to: 
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	
                      
        dd = self.dd       
        ux = self.ux       
        pp = self.pp       
        xi = self.xi       
		           
        ddux = self.ddux     
        dduy = self.dduy     
        dduz =  self.dduz     
                   
        dduxux = self.dduxux   
        dduyuy = self.dduyuy   
        dduzuz = self.dduzuz   
	               
        ddxi    = self.ddxi     
        ddxiux  = self.ddxiux   
        ddxidot = self.ddxidot  
	 	           	
        ddxidotux =	self.ddxidotux
        ddxiuxux  =	self.ddxiuxux 
        ddxiuyuy  = self.ddxiuyuy 
        ddxiuzuz  =	self.ddxiuzuz 
                   
        xigradxpp = self.xigradxpp
 
        xzn0 = self.xzn0
 
        # store time series for time derivatives
        t_timec   = self.t_timec
        t_dd      = self.t_dd
        t_ddux    = self.t_ddux
        t_ddxi    = self.t_ddxi	
        t_ddxiux  = self.t_ddxiux
 
        # construct equation-specific mean fields
        t_fxi  = t_ddxiux - t_ddxi*t_ddux/t_dd
			
        fht_ux = ddux/dd
        fht_xi = ddxi/dd
		
        rxx    = dduxux - ddux*ddux/dd		
        fxi    = ddxiux - ddxi*ddux/dd
        frxi   = ddxiuxux - (ddxi/dd)*dduxux - 2.*(ddux/dd)*ddxiux + 2.*ddxi*ddux*ddux/(dd*dd) 		
			  
        # LHS -dq/dt 
        self.minus_dt_fxi = -self.dt(t_fxi,xzn0,t_timec,intc)
		
        # LHS -div(dduxfxi)
        self.minus_div_fht_ux_fxi = -self.Div(fht_ux*fxi,xzn0)		

        # RHS -div frxi  
        self.minus_div_frxi = -self.Div(frxi,xzn0)

        # RHS -fxi d_r fu_r
        self.minus_fxi_gradx_fht_ux = -fxi*self.Grad(fht_ux,xzn0)
		
        # RHS -rxx d_r xi
        self.minus_rxx_gradx_fht_xi = -rxx*self.Grad(fht_xi,xzn0)

        # RHS - X''i gradx P - X''_i gradx P'
        self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff = \
          -(xi*self.Grad(pp,xzn0) - fht_xi*self.Grad(pp,xzn0)) - (xigradxpp - xi*self.Grad(pp,xzn0)) 		

        # RHS +uxff_eht_dd_xidot
        self.plus_uxff_eht_dd_xidot = +(ddxidotux - (ddux/dd)*ddxidot)  		
		
        # RHS +gi 
        self.plus_gi = \
                  -(ddxiuyuy - (ddxi/dd)*dduyuy - 2.*(dduy/dd) + 2.*ddxi*dduy*dduy/(dd*dd))/xzn0 - \
                   (ddxiuzuz - (ddxi/dd)*dduzuz - 2.*(dduz/dd) + 2.*ddxi*dduz*dduz/(dd*dd))/xzn0 + \
                   (ddxiuyuy - (ddxi/dd)*dduyuy)/xzn0 + \
                   (ddxiuzuz - (ddxi/dd)*dduzuz)/xzn0

        # -res				   
        self.minus_resXiFlux = -(self.minus_dt_fxi + self.minus_div_fht_ux_fxi + self.minus_div_frxi + \
                         self.minus_fxi_gradx_fht_ux + self.minus_rxx_gradx_fht_xi + \
                         self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff + \
                         self.plus_uxff_eht_dd_xidot + self.plus_gi)     
		
        ######################
        # END Xi FLUX EQUATION 
        ######################	

        #print('#----------------------------------------------------#')		
        #print('Loading RA-ILES COMPOSITION FLUX EQUATION terms')	
        #print('Central time (in s): ',round(self.timec,1))	
        #print('Averaging windows (in s): ',self.tavg.item(0))
        #print('Time range (in s from-to): ',round(self.trange[0],1),round(self.trange[1],1))
		
			
    def plot_Xflux(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xflux stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0		
		
        # load and calculate DATA to plot
        plt1 = self.ddxiux - self.ddxi*self.ddux/self.dd
		
        # create FIGURE
        plt.figure(figsize=(7,6))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
					
        # plot DATA 
        plt.title('Xflux for '+self.element)
        plt.plot(grd1,plt1,color='k',label = r'f')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_Xflux_'+element+'.png')
		
    def plot_Xflux_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xi flux equation in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
		
        lhs0 = self.minus_dt_fxi
        lhs1 = self.minus_div_fht_ux_fxi
		
        rhs0 = self.minus_div_frxi
        rhs1 = self.minus_fxi_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_xi
        rhs3 = self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff
        rhs4 = self.plus_uxff_eht_dd_xidot 
        rhs5 = self.plus_gi
		
        res =  self.minus_resXiFlux
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
				
        # plot DATA 
        plt.title('Xflux equation for '+self.element)
        plt.plot(grd1,lhs0,color='#8B3626',label = r'$-\partial_t f_i$')
        plt.plot(grd1,lhs1,color='#FF7256',label = r'$-\nabla_r (\widetilde{u}_r f)$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f^r_i$')
        plt.plot(grd1,rhs1,color='g',label=r'$-f \partial_r \widetilde{u}_r$')
        plt.plot(grd1,rhs2,color='r',label=r'$-R_{rr} \partial_r \widetilde{X}$')	
        plt.plot(grd1,rhs3,color='cyan',label=r'$-\overline{X^{,,}} \partial_r \overline{P} - \overline{X^{,,} \partial_r P^{,}}$')
        plt.plot(grd1,rhs4,color='purple',label=r'$+\overline{u^{,,}_r \rho \dot{X}}$')
        plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')		
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-2}$ s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_XfluxEquation_'+element+'.png')		
						
    #def gauss(x, *p): 
    # Define model function to be used to fit to the data above:
    #    A, mu, sigma = p
    #    return A*np.exp(-(x-mu)**2/(2.*sigma**2))		

	
	
