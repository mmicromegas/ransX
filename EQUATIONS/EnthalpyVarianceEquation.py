import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class EnthalpyVarianceEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,tke_diss,tauL,data_prefix):
        super(EnthalpyVarianceEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0'))
        nx     = np.asarray(eht.item().get('nx')) 		

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd  = np.asarray(eht.item().get('dd')[intc])
        ux  = np.asarray(eht.item().get('ux')[intc])	
        pp  = np.asarray(eht.item().get('pp')[intc])	
        hh  = np.asarray(eht.item().get('hh')[intc])	
		
        ddux   = np.asarray(eht.item().get('ddux')[intc])		
        ddhh   = np.asarray(eht.item().get('ddhh')[intc])	
        divu   = np.asarray(eht.item().get('divu')[intc])		
        dddivu = np.asarray(eht.item().get('dddivu')[intc])		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])          
		
        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])		
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])

        hhddenuc1 = np.asarray(eht.item().get('hhddenuc1')[intc])		
        hhddenuc2 = np.asarray(eht.item().get('hhddenuc2')[intc])

        hhppdivu = np.asarray(eht.item().get('hhppdivu')[intc])  


        ddhhhh   = np.asarray(eht.item().get('ddhhhh')[intc]) 		
        ddhhux   = np.asarray(eht.item().get('ddhhux')[intc]) 		
        ddhhhhux = np.asarray(eht.item().get('ddhhhhux')[intc])   		

        gamma1  = np.asarray(eht.item().get('gamma1')[intc])  
        gamma3  = np.asarray(eht.item().get('gamma3')[intc])  

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if(ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])   
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma1 = cp/cv   # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1
        
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddhh    = np.asarray(eht.item().get('ddhh')) 
        t_ddhhhh  = np.asarray(eht.item().get('ddhhhh')) 
		
        t_sigma_hh	= (t_ddhhhh/t_dd) -(t_ddhh*t_ddhh)/(t_dd*t_dd)	
 		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_hh   = ddhh/dd
        fhh     = ddhhux - ddux*ddhh/dd
        fht_d    = dddivu/dd
        sigma_hh = (ddhhhh/dd)-(ddhh*ddhh)/(dd*dd)		
		
        f_sigma_hh = dd*(ddhhhhux/dd - 2.*ddhh*ddhhux/(dd*dd)-ddux*ddhhhh/(dd*dd) + \
                     2.*(ddhh*ddhh*ddux)/(dd*dd*dd))

        disstke = tke_diss
		
        #####################################
        # ENTHALPY VARIANCE SIGMA HH EQUATION 
        #####################################

        # LHS -dt dd sigma_hh 		
        self.minus_dt_dd_sigma_hh = -self.dt(t_dd*t_sigma_hh,xzn0,t_timec,intc)

        # LHS -div dd fht_ux sigma_ss
        self.minus_div_dd_fht_ux_sigma_hh = -self.Div(dd*fht_ux*sigma_hh,xzn0)
				
        # RHS -div f_sigma_hh
        self.minus_div_f_sigma_hh = -self.Div(f_sigma_hh,xzn0)
				
        # RHS minus_two_fhh_gradx_fht_hh
        self.minus_two_fhh_gradx_fht_hh = -2.*fhh*self.Grad(fht_hh,xzn0)
		
        # RHS -2 gamma1 eht_hhff pp divu
        self.minus_two_gamma1_eht_hhff_pp_divu = -2.*gamma1*(hhppdivu-fht_hh*ppdivu)
			
        # RHS +2 gamma3 eht_hhff dd enuc
        self.plus_two_gamma3_eht_hhff_dd_enuc = +2.*gamma3*(hhddenuc1+hhddenuc2) - 2.*gamma3*fht_hh*(ddenuc1+ddenuc2)

        # RHS +2 gamma3 eht_hhff_tke_diss_approx
        self.plus_two_gamma3_eht_hhff_tke_diss_approx = +2.*gamma3*(hh - fht_hh)*disstke 	
		
        # -res
        self.minus_resSigmaHHequation = -(self.minus_dt_dd_sigma_hh+self.minus_div_dd_fht_ux_sigma_hh+\
         self.minus_div_f_sigma_hh+self.minus_two_fhh_gradx_fht_hh+self.minus_two_gamma1_eht_hhff_pp_divu+\
         self.plus_two_gamma3_eht_hhff_dd_enuc+self.plus_two_gamma3_eht_hhff_tke_diss_approx) 
		  
        # Kolmogorov dissipation, tauL is Kolmogorov damping timescale 		 
        self.minus_sigmaHHkolmdiss = -dd*sigma_hh/tauL	
		
        #########################################
        # END ENTHALPY VARIANCE SIGMA HH EQUATION 
        #########################################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.sigma_hh    = sigma_hh		
		
    def plot_sigma_hh(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian ENTHALPY variance stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.sigma_hh
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title(r'enthalpy variance')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{\sigma}_{h}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_{h}$ (erg$^2$ g$^{-2}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_sigma_hh.png')		

		
    def plot_sigma_hh_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """ sigma hh variance equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_sigma_hh
        lhs1 = self.minus_div_dd_fht_ux_sigma_hh
		
        rhs0 = self.minus_div_f_sigma_hh
        rhs1 = self.minus_two_fhh_gradx_fht_hh
        rhs2 = self.minus_two_gamma1_eht_hhff_pp_divu
        rhs3 = self.plus_two_gamma3_eht_hhff_dd_enuc
        rhs4 = self.plus_two_gamma3_eht_hhff_tke_diss_approx
		
        res = self.minus_resSigmaHHequation
		
        rhs5 = self.minus_sigmaHHkolmdiss		
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        #to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,rhs8,res]
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('hh variance equation')
        plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t (\rho \sigma_{h})$')
        plt.plot(grd1,-lhs1,color='k',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \sigma_{h})$")	
		
        plt.plot(grd1,rhs0,color='r',label = r'$-\nabla f_{\sigma \epsilon_h}$')     
        plt.plot(grd1,rhs1,color='c',label = r'$-2 f_\sigma \partial_r \widetilde{h}$') 
        plt.plot(grd1,rhs2,color='m',label = r"$+2 \Gamma_1 \overline{h'' P d}$")
        #plt.plot(grd1,rhs1+rhs2,color='c',label = r"$-2 f_\sigma \partial_r \widetilde{h}+2 \Gamma_1 \overline{h'' P d}$")		
        plt.plot(grd1,rhs3,color='b',label = r"$-2 \Gamma_3 \overline{h'' \rho \varepsilon_{nuc}} $")		
        plt.plot(grd1,rhs4,color='deeppink',label = r"$+2 \Gamma_3 \overline{h'' \varepsilon_{k}} $")		
        plt.plot(grd1,rhs5,color='k',linewidth=0.8,label = r"$-\sigma_h / \tau_L$")				
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_\sigma$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_{h}$ (erg$^2$ g$^{-1}$ cm$^{-3}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'sigma_hh_eq.png')		
		
		
		
