import numpy as np
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ZbarFluxTransportEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(ZbarFluxTransportEquation,self).__init__(ig) 
					
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])	
        pp = np.asarray(eht.item().get('pp')[intc])
        zbar = np.asarray(eht.item().get('zbar')[intc])	
		
        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])
	
        ddzbar    = np.asarray(eht.item().get('ddzbar')[intc])
        ddzbarux  = np.asarray(eht.item().get('ddzbarux')[intc])
			
        ddzbaruxux  = np.asarray(eht.item().get('ddzbaruxux')[intc])		
        ddzbaruyuy  = np.asarray(eht.item().get('ddzbaruyuy')[intc])
        ddzbaruzuz  = np.asarray(eht.item().get('ddzbaruzuz')[intc])		

        zbargradxpp = np.asarray(eht.item().get('zbargradxpp')[intc]) 
		
        ddabazbar_sum_xdn_o_an = np.asarray(eht.item().get('ddabazbar_sum_xdn_o_an')[intc])  
        uxddabazbar_sum_xdn_o_an = np.asarray(eht.item().get('uxddabazbar_sum_xdn_o_an')[intc]) 
		
        ddabar_sum_znxdn_o_an  = np.asarray(eht.item().get('ddabar_sum_znxdn_o_an')[intc])  
        uxddabar_sum_znxdn_o_an = np.asarray(eht.item().get('uxddabar_sum_znxdn_o_an')[intc]) 
		

        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddux    = np.asarray(eht.item().get('ddux')) 
        t_ddzbar    = np.asarray(eht.item().get('ddzbar'))		
        t_ddzbarux  = np.asarray(eht.item().get('ddzbarux'))
				
        ####################
        # Zbar FLUX EQUATION 
        ####################		
   
        # construct equation-specific mean fields
        t_fzbar  = t_ddzbarux - t_ddzbar*t_ddux/t_dd
			
        fht_ux = ddux/dd
        fht_zbar = ddzbar/dd
		
        rxx      = dduxux - dd*fht_ux*fht_ux		
        fzbar    = ddzbarux - dd*fht_zbar*fht_ux
        fzbarx   = ddzbaruxux - fht_zbar*dduxux - 2.*fht_ux*ddzbarux + 2.*dd*fht_ux*fht_zbar*fht_zbar 		
			  
        # LHS -dq/dt 
        self.minus_dt_fzbar = -self.dt(t_fzbar,xzn0,t_timec,intc)
		
        # LHS -div fht_ux fzbar
        self.minus_div_fht_ux_fzbar = -self.Div(fht_ux*fzbar,xzn0)		

        # RHS -div fzbarx  
        self.minus_div_fzbarx = -self.Div(fzbarx,xzn0)

        # RHS -fzbar d_r fu_r
        self.minus_fzbar_gradx_fht_ux = -fzbar*self.Grad(fht_ux,xzn0)
		
        # RHS -rxx d_r zbar
        self.minus_rxx_gradx_fht_zbar = -rxx*self.Grad(fht_zbar,xzn0)

        # RHS - Z''gradx P - Z''gradx P'
        self.minus_zbarff_gradx_pp_minus_zbarff_gradx_ppf = \
          -(zbar*self.Grad(pp,xzn0) - fht_zbar*self.Grad(pp,xzn0)) - (zbargradxpp - zbar*self.Grad(pp,xzn0)) 		

        # RHS -uxffddzbarsq_sum_xdn_o_an
        self.minus_uxffddabazbar_sum_xdn_o_an = -(uxddabazbar_sum_xdn_o_an - fht_ux*ddabazbar_sum_xdn_o_an)  	

        # RHS -uxffddabar_sum_znxdn_o_an
        self.minus_uxffddabar_sum_znxdn_o_an = -(uxddabar_sum_znxdn_o_an - fht_ux*ddabar_sum_znxdn_o_an)		
		
		
        # RHS +gi 
        self.plus_gzbar = \
                  -(ddzbaruyuy - (ddzbar/dd)*dduyuy - 2.*(dduy/dd) + 2.*ddzbar*dduy*dduy/(dd*dd))/xzn0 - \
                   (ddzbaruzuz - (ddzbar/dd)*dduzuz - 2.*(dduz/dd) + 2.*ddzbar*dduz*dduz/(dd*dd))/xzn0 + \
                   (ddzbaruyuy - (ddzbar/dd)*dduyuy)/xzn0 + \
                   (ddzbaruzuz - (ddzbar/dd)*dduzuz)/xzn0

        # -res				   
        self.minus_resZbarFlux = -(self.minus_dt_fzbar + self.minus_div_fht_ux_fzbar + self.minus_div_fzbarx + \
                         self.minus_fzbar_gradx_fht_ux + self.minus_rxx_gradx_fht_zbar + \
                         self.minus_zbarff_gradx_pp_minus_zbarff_gradx_ppf + \
                         self.minus_uxffddabazbar_sum_xdn_o_an + self.minus_uxffddabar_sum_znxdn_o_an + self.plus_gzbar)     
		
        ########################
        # END Zbar FLUX EQUATION 
        ########################	

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0		
        self.fzbarx       = fzbarx		
		
    def plot_zbarflux(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot zbarflux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0		
		
        # load and calculate DATA to plot
        plt1 = self.fzbarx
		
        # create FIGURE
        plt.figure(figsize=(7,6))	

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))			
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
					
        # plot DATA 
        plt.title('zbar flux')
        plt.plot(grd1,plt1,color='k',label = r'f')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{Z'' u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_zbarflux.png')
		
    def plot_zbarflux_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot zbar flux equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
		
        lhs0 = self.minus_dt_fzbar
        lhs1 = self.minus_div_fht_ux_fzbar
		
        rhs0 = self.minus_div_fzbarx
        rhs1 = self.minus_fzbar_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_zbar
        rhs3 = self.minus_zbarff_gradx_pp_minus_zbarff_gradx_ppf
        rhs4 = self.minus_uxffddabazbar_sum_xdn_o_an
        rhs5 = self.minus_uxffddabar_sum_znxdn_o_an
        rhs6 = self.plus_gzbar
		
        res =  self.minus_resZbarFlux
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
				
        # plot DATA 
        plt.title('zbar flux equation')
        plt.plot(grd1,lhs0,color='#8B3626',label = r'$-\partial_t f_Z$')
        plt.plot(grd1,lhs1,color='#FF7256',label = r'$-\nabla_r (\widetilde{u}_r f_Z)$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f^r_Z$')
        plt.plot(grd1,rhs1,color='g',label=r'$-f_Z \partial_r \widetilde{u}_r$')
        plt.plot(grd1,rhs2,color='r',label=r'$-R_{rr} \partial_r \widetilde{Z}$')	
        plt.plot(grd1,rhs3,color='cyan',label=r"$-\overline{Z''} \partial_r \overline{P} - \overline{Z'' \partial_r P'}$")
        plt.plot(grd1,rhs4,color='purple',label=r"$-\overline{u''_r \rho A Z \sum_\alpha \dot{X}_\alpha^{nuc} / A_\alpha}$")
        plt.plot(grd1,rhs5,color='m',label=r"$-\overline{u''_r \rho A \sum_\alpha Z_\alpha \dot{X}_\alpha^{nuc} / A_\alpha}$")		
        plt.plot(grd1,rhs6,color='yellow',label=r'$+G_Z$')		
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
        plt.savefig('RESULTS/'+self.data_prefix+'mean_ZbarFluxTransportEquation.png')			