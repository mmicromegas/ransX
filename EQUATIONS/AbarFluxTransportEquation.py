import numpy as np
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class AbarFluxTransportEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(AbarFluxTransportEquation,self).__init__(ig) 
					
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])	
        pp = np.asarray(eht.item().get('pp')[intc])
        abar = np.asarray(eht.item().get('abar')[intc])	
		
        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])
	
        ddabar    = np.asarray(eht.item().get('ddabar')[intc])
        ddabarux  = np.asarray(eht.item().get('ddabarux')[intc])
			
        ddabaruxux  = np.asarray(eht.item().get('ddabaruxux')[intc])		
        ddabaruyuy  = np.asarray(eht.item().get('ddabaruyuy')[intc])
        ddabaruzuz  = np.asarray(eht.item().get('ddabaruzuz')[intc])		

        abargradxpp = np.asarray(eht.item().get('abargradxpp')[intc]) 
		
        uxddabarsq_sum_xdn_o_an	= np.asarray(eht.item().get('uxddabarsq_sum_xdn_o_an')[intc]) 	
        ddabarsq_sum_xdn_o_an = np.asarray(eht.item().get('ddabarsq_sum_xdn_o_an')[intc]) 	

        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddux    = np.asarray(eht.item().get('ddux')) 
        t_ddabar    = np.asarray(eht.item().get('ddabar'))		
        t_ddabarux  = np.asarray(eht.item().get('ddabarux'))
				
        ####################
        # Abar FLUX EQUATION 
        ####################		
   
        # construct equation-specific mean fields
        t_fabar  = t_ddabarux - t_ddabar*t_ddux/t_dd
			
        fht_ux = ddux/dd
        fht_abar = ddabar/dd
		
        rxx      = dduxux - dd*fht_ux*fht_ux		
        fabar    = ddabarux - dd*fht_abar*fht_ux
        fabarx   = ddabaruxux - fht_abar*dduxux - 2.*fht_ux*ddabarux + 2.*dd*fht_ux*fht_abar*fht_abar 		
			  
        # LHS -dq/dt 
        self.minus_dt_fabar = -self.dt(t_fabar,xzn0,t_timec,intc)
		
        # LHS -div fht_ux fabar
        self.minus_div_fht_ux_fabar = -self.Div(fht_ux*fabar,xzn0)		

        # RHS -div fabarx  
        self.minus_div_fabarx = -self.Div(fabarx,xzn0)

        # RHS -fabar d_r fu_r
        self.minus_fabar_gradx_fht_ux = -fabar*self.Grad(fht_ux,xzn0)
		
        # RHS -rxx d_r abar
        self.minus_rxx_gradx_fht_abar = -rxx*self.Grad(fht_abar,xzn0)

        # RHS - A''gradx P - A''gradx P'
        self.minus_abarff_gradx_pp_minus_abarff_gradx_ppf = \
          -(abar*self.Grad(pp,xzn0) - fht_abar*self.Grad(pp,xzn0)) - (abargradxpp - abar*self.Grad(pp,xzn0)) 		

        # RHS -uxffddabarsq_sum_xdn_o_an
        self.minus_uxffddabarsq_sum_xdn_o_an = -(uxddabarsq_sum_xdn_o_an - fht_ux*ddabarsq_sum_xdn_o_an)  		
		
        # RHS +gi 
        self.plus_gabar = \
                  -(ddabaruyuy - (ddabar/dd)*dduyuy - 2.*(dduy/dd) + 2.*ddabar*dduy*dduy/(dd*dd))/xzn0 - \
                   (ddabaruzuz - (ddabar/dd)*dduzuz - 2.*(dduz/dd) + 2.*ddabar*dduz*dduz/(dd*dd))/xzn0 + \
                   (ddabaruyuy - (ddabar/dd)*dduyuy)/xzn0 + \
                   (ddabaruzuz - (ddabar/dd)*dduzuz)/xzn0

        # -res				   
        self.minus_resAbarFlux = -(self.minus_dt_fabar + self.minus_div_fht_ux_fabar + self.minus_div_fabarx + \
                         self.minus_fabar_gradx_fht_ux + self.minus_rxx_gradx_fht_abar + \
                         self.minus_abarff_gradx_pp_minus_abarff_gradx_ppf + \
                         self.minus_uxffddabarsq_sum_xdn_o_an + self.plus_gabar)     
		
        ########################
        # END Abar FLUX EQUATION 
        ########################	

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0		
        self.fabarx       = fabarx		
		
    def plot_abarflux(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Abarflux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0		
		
        # load and calculate DATA to plot
        plt1 = self.fabarx
		
        # create FIGURE
        plt.figure(figsize=(7,6))	

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))			
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
					
        # plot DATA 
        plt.title('Abar flux')
        plt.plot(grd1,plt1,color='k',label = r'f')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{A'' u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_abarflux.png')
		
    def plot_abarflux_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot abar flux equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
		
        lhs0 = self.minus_dt_fabar
        lhs1 = self.minus_div_fht_ux_fabar
		
        rhs0 = self.minus_div_fabarx
        rhs1 = self.minus_fabar_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_abar
        rhs3 = self.minus_abarff_gradx_pp_minus_abarff_gradx_ppf
        rhs4 = self.minus_uxffddabarsq_sum_xdn_o_an
        rhs5 = self.plus_gabar
		
        res =  self.minus_resAbarFlux
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
				
        # plot DATA 
        plt.title('Abar flux equation')
        plt.plot(grd1,lhs0,color='#8B3626',label = r'$-\partial_t f_A$')
        plt.plot(grd1,lhs1,color='#FF7256',label = r'$-\nabla_r (\widetilde{u}_r f_A)$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f^r_A$')
        plt.plot(grd1,rhs1,color='g',label=r'$-f_A \partial_r \widetilde{u}_r$')
        plt.plot(grd1,rhs2,color='r',label=r'$-R_{rr} \partial_r \widetilde{A}$')	
        plt.plot(grd1,rhs3,color='cyan',label=r"$-\overline{A''} \partial_r \overline{P} - \overline{A'' \partial_r P'}$")
        plt.plot(grd1,rhs4,color='purple',label=r"$-\overline{u''_r \rho A^2 \sum_\alpha \dot{X}_\alpha^{nuc} / A_\alpha}$")
        plt.plot(grd1,rhs5,color='yellow',label=r'$+G_A$')		
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
        plt.savefig('RESULTS/'+self.data_prefix+'mean_AbarFluxTransportEquation.png')			