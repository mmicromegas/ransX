import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class InternalEnergyVarianceEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,tke_diss,tauL,data_prefix):
        super(InternalEnergyVarianceEquation,self).__init__(ig) 
	
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
        ei  = np.asarray(eht.item().get('ei')[intc])	
		
        ddux   = np.asarray(eht.item().get('ddux')[intc])		
        ddei   = np.asarray(eht.item().get('ddei')[intc])
        eipp   = np.asarray(eht.item().get('eipp')[intc])		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        dddivu = np.asarray(eht.item().get('dddivu')[intc])		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])          
		
        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])		
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])

        eiddenuc1 = np.asarray(eht.item().get('eiddenuc1')[intc])		
        eiddenuc2 = np.asarray(eht.item().get('eiddenuc2')[intc])

        eippdivu = np.asarray(eht.item().get('eippdivu')[intc])  
        eidivu  = np.asarray(eht.item().get('eidivu')[intc])  

        ddeiei   = np.asarray(eht.item().get('ddeiei')[intc]) 		
        ddeiux   = np.asarray(eht.item().get('ddeiux')[intc]) 		
        ddeieiux = np.asarray(eht.item().get('ddeieiux')[intc])   		
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddei    = np.asarray(eht.item().get('ddei')) 
        t_ddeiei  = np.asarray(eht.item().get('ddeiei')) 
		
        t_sigma_ei	= (t_ddeiei/t_dd) -(t_ddei*t_ddei)/(t_dd*t_dd)	
 		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_ei   = ddei/dd
        fei     = ddeiux - ddux*ddei/dd
        fht_d    = dddivu/dd
        sigma_ei = (ddeiei/dd)-(ddei*ddei)/(dd*dd)		

        eht_eiff_ppf = eipp	- ei*pp 	
        eht_eiff_dff = eidivu - ei*dddivu/dd - divu*ddei/dd + ddei*dddivu/(dd*dd)		
        eht_eiff_ppf_dff = eippdivu - eidivu*pp - (ddei/dd)*ppdivu + (ddei/dd)*pp*divu - eipp*dddivu/dd + ei*pp*dddivu/dd		
		
        f_sigma_ei = dd*(ddeieiux/dd - 2.*ddei*ddeiux/(dd*dd)-ddux*ddeiei/(dd*dd) + \
                     2.*(ddei*ddei*ddux)/(dd*dd*dd))

        disstke = tke_diss
		
        ############################################
        # INTERNAL ENERGY VARIANCE SIGMA EI EQUATION 
        ############################################

        # LHS -dt dd sigma_ei 		
        self.minus_dt_dd_sigma_ei = -self.dt(t_dd*t_sigma_ei,xzn0,t_timec,intc)

        # LHS -div dd fht_ux sigma_ss
        self.minus_div_dd_fht_ux_sigma_ei = -self.Div(dd*fht_ux*sigma_ei,xzn0)
				
        # RHS -div f_sigma_ei
        self.minus_div_f_sigma_ei = -self.Div(f_sigma_ei,xzn0)
				
        # RHS minus_two_fei_gradx_fht_ei
        self.minus_two_fei_gradx_fht_ei = -2.*fei*self.Grad(fht_ei,xzn0)
		
        # RHS -2 eiff eht_pp fht_d
        self.minus_two_eiff_eht_pp_fht_d = -2.*(ei-fht_ei)*pp*fht_d

        # RHS -2 eht_pp eht_eiff dff
        self.minus_two_eht_pp_eht_eiff_dff = -2.*pp*eht_eiff_dff

        # RHS -2 fht_d eht_eiff ppf
        self.minus_two_fht_d_eht_eiff_ppf = -2.*fht_d*eht_eiff_ppf

        # RHS -2 eht_eiff ppf dff
        self.minus_two_eht_eiff_ppf_dff =  -2.*eht_eiff_ppf_dff		
		
        # RHS +2 eht_eiff dd enuc
        self.plus_two_eht_eiff_dd_enuc = 2.*(eiddenuc1+eiddenuc2) - 2.*fht_ei*(ddenuc1+ddenuc2)

        # RHS +2 eht_eiff_tke_diss_approx
        self.plus_two_eht_eiff_tke_diss_approx = 2.*(ei - ddei/dd)*disstke 	
		
        # -res
        self.minus_resSigmaEIequation = -(self.minus_dt_dd_sigma_ei + self.minus_div_dd_fht_ux_sigma_ei + \
          self.minus_div_f_sigma_ei + self.minus_two_fei_gradx_fht_ei + self.minus_two_eiff_eht_pp_fht_d + \
          self.minus_two_eht_pp_eht_eiff_dff + self.minus_two_fht_d_eht_eiff_ppf + self.minus_two_eht_eiff_ppf_dff + \
          self.plus_two_eht_eiff_dd_enuc + self.plus_two_eht_eiff_tke_diss_approx) 
		  
        # Kolmogorov dissipation, tauL is Kolmogorov damping timescale 		 
        self.minus_sigmaEIkolmdiss = -dd*sigma_ei/tauL	
		
        ################################################
        # END INTERNAL ENERGY VARIANCE SIGMA EI EQUATION 
        ################################################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.sigma_ei    = sigma_ei		
		
    def plot_sigma_ei(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian internal energy variance stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.sigma_ei
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title(r'internal energy variance')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{\sigma}_{\epsilon I}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_{\epsilon I}$ (erg$^2$ g$^{-2}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_sigma_ei.png')		

		
    def plot_sigma_ei_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """ sigma ei variance equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_sigma_ei
        lhs1 = self.minus_div_dd_fht_ux_sigma_ei
		
        rhs0 = self.minus_div_f_sigma_ei
        rhs1 = self.minus_two_fei_gradx_fht_ei		
        rhs2 = self.minus_two_eiff_eht_pp_fht_d
        rhs3 = self.minus_two_eht_pp_eht_eiff_dff
        rhs4 = self.minus_two_fht_d_eht_eiff_ppf
        rhs5 = self.minus_two_eht_eiff_ppf_dff
        rhs6 = self.plus_two_eht_eiff_dd_enuc
        rhs7 = self.plus_two_eht_eiff_tke_diss_approx 
		
        #rhs9 = self.minus_two_fei_gradx_fht_ei + self.minus_two_eht_pp_eht_eiff_dff
		
        res = self.minus_resSigmaEIequation
		
        rhs8 = self.minus_sigmaEIkolmdiss		
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        #to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,rhs8,res]
        to_plot = [lhs0,lhs1,rhs0,rhs2,rhs4,rhs5,rhs6,rhs7,rhs8,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('ei variance equation')
        plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t (\rho \sigma_{\epsilon_I})$')
        plt.plot(grd1,-lhs1,color='k',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \sigma_{\epsilon_I})$")	
		
        plt.plot(grd1,rhs0,color='r',label = r'$-\nabla f_{\sigma \epsilon_I}$')     
        plt.plot(grd1,rhs1,color='c',label = r'$-2 f_\sigma \partial_r \widetilde{\epsilon_I}$') 
        #plt.plot(grd1,rhs9,color='c',label = r'$-2 f_\sigma \partial_r \widetilde{\epsilon_I} -2 \overline{P} \ \overline{\epsilon''_I d''}$')
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-2 \overline{\epsilon''_i} \ \overline{P} \ \widetilde{d}$") 
        plt.plot(grd1,rhs3,color='m',label = r"$-2 \overline{P} \ \overline{\epsilon''_I d''}$")
        plt.plot(grd1,rhs4,color='g',label = r"$-2 \widetilde{d} \ \overline{\epsilon''_I P'}$")	
        plt.plot(grd1,rhs5,color='olive',label = r"$-2 \overline{\epsilon''_I P' d''} $")	
        plt.plot(grd1,rhs6,color='b',label = r"$+2\overline{\epsilon''_I \rho \varepsilon_{nuc}} $")	
        plt.plot(grd1,rhs7,color='deeppink',label = r"$+2\overline{\epsilon''_I \varepsilon_{k}} $")	
        plt.plot(grd1,rhs8,color='k',linewidth=0.8,label = r"$-\sigma_\epsilon / \tau_L$")				
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_\sigma$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_{\epsilon I}$ (erg$^2$ g$^{-1}$ cm$^{-3}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'sigma_ei_eq.png')		
		
		
		
