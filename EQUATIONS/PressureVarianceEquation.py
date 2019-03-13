import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class PressureVarianceEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,tauL,data_prefix):
        super(PressureVarianceEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0'))
        nx     = np.asarray(eht.item().get('nx')) 		

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	

        dd  = np.asarray(eht.item().get('dd')[intc])
        ux  = np.asarray(eht.item().get('ux')[intc])	
        pp  = np.asarray(eht.item().get('pp')[intc])	
		
        ppsq   = np.asarray(eht.item().get('ppsq')[intc])   		
        ddux   = np.asarray(eht.item().get('ddux')[intc])				
        divu   = np.asarray(eht.item().get('divu')[intc])     		
        ppux   = np.asarray(eht.item().get('ppux')[intc]) 		
        ppppux = np.asarray(eht.item().get('ppppux')[intc])  		
        dddivu = np.asarray(eht.item().get('dddivu')[intc])		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])          
		
        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])		
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])

        ppddenuc1 = np.asarray(eht.item().get('ppddenuc1')[intc])		
        ppddenuc2 = np.asarray(eht.item().get('ppddenuc2')[intc])

        ppppdivu = np.asarray(eht.item().get('ppppdivu')[intc])  
        ppdivu  = np.asarray(eht.item().get('ppdivu')[intc])  

        gamma1 = np.asarray(eht.item().get('gamma1')[intc])		
        gamma3 = np.asarray(eht.item().get('gamma3')[intc])
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_pp      = np.asarray(eht.item().get('pp'))
        t_ppsq    = np.asarray(eht.item().get('ppsq'))  
		
        t_sigma_pp	= t_ppsq - t_pp*t_pp	
 		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fpp      = ppux - pp*ux
        fht_d    = dddivu/dd
        sigma_pp = ppsq - pp*pp	

        eht_ppf_dff = ppdivu - pp*divu 	
        eht_ppf_ppf_dff = ppppdivu - 2.*ppdivu*pp + pp*pp*divu - ppsq*dddivu/dd + pp*pp*dddivu/dd           
	
        f_sigma_pp = ppppux - 2.*ppux*pp + pp*pp*ux - ppsq*ddux/dd + pp*pp*ddux/dd

        disstke = tke_diss
		
        #####################################
        # PRESSURE VARIANCE SIGMA PP EQUATION 
        #####################################

        # LHS -dt sigma_pp 		
        self.minus_dt_sigma_pp = -self.dt(t_sigma_pp,xzn0,t_timec,intc)

        # LHS -fht_ux gradx sigma_pp
        self.minus_fht_ux_gradx_sigma_pp = -fht_ux*self.Grad(sigma_pp,xzn0)
				
        # RHS -div f_sigma_pp
        self.minus_div_f_sigma_pp = -self.Div(f_sigma_pp,xzn0)
				
        # RHS -2_gamma1_pp_ppf_ddff
        self.minus_two_gamma1_pp_ppf_ddff = -2.*gamma1*pp*eht_ppf_dff		
				
        # RHS minus_two_fpp_gradx_pp
        self.minus_two_fpp_gradx_pp = -2.*fpp*self.Grad(pp,xzn0)
		
        # RHS minus_two_gamma1_fht_d_sigma_pp
        self.minus_two_gamma1_fht_d_sigma_pp = -2.*gamma1*fht_d*sigma_pp		
		
        # RHS minus_two_gamma1_minus_one_eht_ppf_ppf_dff	
        self.minus_two_gamma1_minus_one_eht_ppf_ppf_dff = -(2.*gamma1-1.)*eht_ppf_ppf_dff

        # RHS plus_two_gamma3_minus_one_ppf_dd_enuc	
        self.plus_two_gamma3_minus_one_ppf_dd_enuc = +(2.*gamma3-1.)*((ppddenuc1 + ppddenuc2)-pp*(ddenuc1+ddenuc2))
		
        # -res
        self.minus_resSigmaPPequation = -(self.minus_dt_sigma_pp+self.minus_fht_ux_gradx_sigma_pp+self.minus_div_f_sigma_pp+\
          self.minus_two_gamma1_pp_ppf_ddff+self.minus_two_fpp_gradx_pp+self.minus_two_gamma1_fht_d_sigma_pp+\
          self.minus_two_gamma1_minus_one_eht_ppf_ppf_dff+self.plus_two_gamma3_minus_one_ppf_dd_enuc) 
		  
        # Kolmogorov dissipation, tauL is Kolmogorov damping timescale 		 
        self.minus_sigmaPPkolmdiss = -sigma_pp/tauL	
		
        #########################################
        # END PRESSURE VARIANCE SIGMA PP EQUATION 
        #########################################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.sigma_pp    = sigma_pp		
		
    def plot_sigma_pp(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean pressure variance stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.sigma_pp
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title(r'pressure variance')
        plt.plot(grd1,plt1,color='brown',label = r'$\sigma_{P}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_{P}$ (erg$^2$ cm$^{-6}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_sigma_pp.png')		

		
    def plot_sigma_pp_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """ pressure variance equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_sigma_pp
        lhs1 = self.minus_fht_ux_gradx_sigma_pp
		
        rhs0 = self.minus_div_f_sigma_pp
        rhs1 = self.minus_two_gamma1_pp_ppf_ddff
        rhs2 = self.minus_two_fpp_gradx_pp
        rhs3 = self.minus_two_gamma1_fht_d_sigma_pp
        rhs4 = self.minus_two_gamma1_minus_one_eht_ppf_ppf_dff
        rhs5 = self.plus_two_gamma3_minus_one_ppf_dd_enuc 
		
        res = self.minus_resSigmaPPequation
		
        rhs6 = self.minus_sigmaPPkolmdiss		
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs4,rhs5,rhs6,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('pp variance equation')
        plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t \sigma_P$')
        plt.plot(grd1,-lhs1,color='k',label = r"$-\widetilde{u}_r \partial_r \sigma_P$")	
		
        plt.plot(grd1,rhs0,color='r',label = r"$-\nabla f_{\sigma P}$")     
        plt.plot(grd1,rhs1,color='c',label = r"$-2 \Gamma_1 \overline{P} \ \overline{P'd''}$")
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-2 f_P \partial_r \overline{P}$") 
        #plt.plot(grd1,rhs1+rhs2,color='#802A2A',label = r"$-2 f_P \partial_r \overline{P}-2 \Gamma_1 \overline{P} \ \overline{P'd''}$")
        plt.plot(grd1,rhs3,color='m',label = r"$+2 \Gamma_1 \widetilde{d} \sigma_P$")
        plt.plot(grd1,rhs4,color='g',label = r"$-2(\Gamma_1 -1)\overline{P'P'd''}$")	
        plt.plot(grd1,rhs5,color='olive',label = r"$+2(\Gamma_3 -1)\overline{P' \rho \varepsilon_{nuc}}$")			
        plt.plot(grd1,rhs6,color='k',linewidth=0.8,label = r"$-\sigma_P / \tau_L$")				
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_\sigma$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\sigma_{P}$ (erg$^2$ cm$^{-3}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'sigma_pp_eq.png')		
		
		
		
