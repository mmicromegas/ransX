import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class DensityVarianceEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tauL,data_prefix):
        super(DensityVarianceEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd        = np.asarray(eht.item().get('dd')[intc])
        ux        = np.asarray(eht.item().get('ux')[intc])	
		
        ddux      = np.asarray(eht.item().get('ddux')[intc])		
        ddsq      = np.asarray(eht.item().get('ddsq')[intc])
        divu      = np.asarray(eht.item().get('divu')[intc])		
        ddddux    = np.asarray(eht.item().get('ddddux')[intc])
        dddivu    = np.asarray(eht.item().get('dddivu')[intc])
        dddddivu  = np.asarray(eht.item().get('dddddivu')[intc])
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddux    = np.asarray(eht.item().get('ddux')) 			
        t_ddsq    = np.asarray(eht.item().get('ddsq'))		
		
        # construct equation-specific mean fields
        fht_ux   = ddux/dd
        fht_divu = dddivu/dd		
        sigma_dd = ddsq - dd*dd

        eht_ddf_uxff       = ddux - dd*fht_ux - dd*ux + dd*fht_ux 		
        eht_dd_eht_ddf_dff = dddivu - dd*fht_divu - dd*divu + dd*fht_divu 		
        eht_ddf_ddf_dff    = dddddivu - 2.*dddivu*dd + dd*dd*divu - ddsq*fht_divu + dd*dd*fht_divu 	
		
        f_sigma_dd = ddddux - 2.*ddux*dd + dd*dd*ux - ddsq*fht_ux + dd*dd*fht_ux		
    
        ###########################   		
        # DENSITY VARIANCE EQUATION
        ###########################		
		
        # time-series of density variance 
        t_sigma_dd = t_ddsq - t_dd*t_dd
		
        # LHS -dq/dt 		
        self.minus_dt_sigma_dd = -self.dt(t_sigma_dd,xzn0,t_timec,intc)
     
        # LHS -div fht_ux sigma_dd
        self.minus_fht_ux_div_sigma_dd = -fht_ux*self.Div(sigma_dd,xzn0)	 
		
        # RHS -div f_sigma_dd
        self.minus_div_f_sigma_dd = -self.Div(f_sigma_dd,xzn0)        
        
        # RHS minus_two_eht_dd_eht_ddf_dff
        self.minus_two_eht_dd_eht_ddf_dff = -2.*dd*eht_dd_eht_ddf_dff
		
        # RHS minus_two_eht_ddf_uxff_gradx_eht_rho
        self.minus_two_eht_ddf_uxff_gradx_eht_dd = -2.*eht_ddf_uxff*self.Grad(dd,xzn0) 
		
        # RHS minus_two_fht_divu_sigma_dd
        self.minus_two_fht_divu_sigma_dd = -2.*fht_divu*(sigma_dd)

        # RHS -eht_ddf_ddf_dff
        self.minus_eht_ddf_ddf_dff = -eht_ddf_ddf_dff

        # -res
        self.minus_resSigmaDDequation = -(self.minus_dt_sigma_dd + self.minus_fht_ux_div_sigma_dd + \
           self.minus_div_f_sigma_dd + self.minus_two_eht_dd_eht_ddf_dff + self.minus_two_eht_ddf_uxff_gradx_eht_dd + \
           self.minus_two_fht_divu_sigma_dd + self.minus_eht_ddf_ddf_dff)

        # Kolmogorov dissipation, tauL is Kolmogorov damping timescale		   
        self.minus_sigmaDDkolmdiss = -sigma_dd/tauL		

        ###############################   		
        # END DENSITY VARIANCE EQUATION
        ###############################	

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.sigma_dd    = sigma_dd			
					
    def plot_sigma_dd(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean density variance in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.sigma_dd
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'density variance')
        plt.plot(grd1,plt1,color='brown',label = r"$\sigma_{\rho}$")

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_{\rho}$ (g$^{-2}$ cm$^{-6}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_sigma_dd.png')
					
    def plot_sigma_dd_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """ density variance equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_sigma_dd
        lhs1 = self.minus_fht_ux_div_sigma_dd
		
        rhs0 = self.minus_div_f_sigma_dd
        rhs1 = self.minus_two_eht_dd_eht_ddf_dff
        rhs2 = self.minus_two_eht_ddf_uxff_gradx_eht_dd 
        rhs3 = self.minus_two_fht_divu_sigma_dd 
        rhs4 = self.minus_eht_ddf_ddf_dff
		
        res = self.minus_resSigmaDDequation
		
        rhs5 = self.minus_sigmaDDkolmdiss		
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('density variance equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r'$-\partial_t (\sigma_{\rho})$')
        plt.plot(grd1,lhs1,color='k',label = r"$-(\widetilde{u}_r \nabla_r \sigma_{\rho})$")	
		
        plt.plot(grd1,rhs0,color='r',label = r"$-\nabla f_{\sigma_{\rho}}$")     
        plt.plot(grd1,rhs1,color='c',label = r"$-2 \overline{\rho} \ \overline{\rho' d''}$") 
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-2 \overline{\rho' u''_r} \partial_r \overline{\rho}$") 
        #plt.plot(grd1,rhs1+rhs2,color='#802A2A',label = r"$-2 \overline{\rho' u''_r} \partial_r \overline{\rho} - 2 \overline{\rho} \ \overline{\rho' d''}$")
        plt.plot(grd1,rhs3,color='m',label = r"$-2 \widetilde{d} \ \sigma_\rho$")
        plt.plot(grd1,rhs4,color='g',label = r"$-\overline{\rho' \rho' d''}$")
        plt.plot(grd1,rhs5,color='k',linewidth=0.8,label = r"$-\sigma_\rho / \tau_L$")		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_{\sigma_\rho}$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_\rho$ (g$^2$ cm$^{-6}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'sigma_dd_eq.png')									
		
		
