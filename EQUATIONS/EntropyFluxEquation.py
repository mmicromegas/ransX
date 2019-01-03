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

class EntropyFluxEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(EntropyFluxEquation,self).__init__(ig) 
	
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
        self.ss        = np.asarray(eht.item().get('ss')[intc])	
        self.tt        = np.asarray(eht.item().get('tt')[intc])
		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])		
        self.dduy      = np.asarray(eht.item().get('dduy')[intc])
        self.dduz      = np.asarray(eht.item().get('dduz')[intc])		
        self.ddss      = np.asarray(eht.item().get('ddss')[intc])
		
        self.dduxux      = np.asarray(eht.item().get('dduxux')[intc])		
        self.dduyuy      = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz      = np.asarray(eht.item().get('dduzuz')[intc])	
		
        self.ddssux      = np.asarray(eht.item().get('ddssux')[intc])
        self.ddssuy      = np.asarray(eht.item().get('ddssuy')[intc])
        self.ddssuz      = np.asarray(eht.item().get('ddssuz')[intc])		

        self.ddssuxux      = np.asarray(eht.item().get('ddssuxux')[intc])
        self.ddssuyuy      = np.asarray(eht.item().get('ddssuyuy')[intc])
        self.ddssuzuz      = np.asarray(eht.item().get('ddssuzuz')[intc])
		
        self.divu        = np.asarray(eht.item().get('divu')[intc])		
        self.ppdivu      = np.asarray(eht.item().get('ppdivu')[intc])

        self.ddenuc1_tt      = np.asarray(eht.item().get('ddenuc1_tt')[intc])		
        self.ddenuc2_tt      = np.asarray(eht.item().get('ddenuc2_tt')[intc])

        self.dduxenuc1_tt      = np.asarray(eht.item().get('dduxenuc1_tt')[intc])		
        self.dduxenuc2_tt      = np.asarray(eht.item().get('dduxenuc2_tt')[intc])
		
        self.ssgradxpp      = np.asarray(eht.item().get('ssgradxpp')[intc])				
		
        self.ppdivu      = np.asarray(eht.item().get('ppdivu')[intc])		
        self.uxppdivu    = np.asarray(eht.item().get('uxppdivu')[intc])		
		
        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd = np.asarray(eht.item().get('dd'))
        t_ddux = np.asarray(eht.item().get('ddux')) 
        t_ddss = np.asarray(eht.item().get('ddss'))
        t_ddssux = np.asarray(eht.item().get('ddssux')) 		

 		# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf	

        ux = self.ux
        dd = self.dd
        pp = self.pp
        ss = self.ss
        tt = self.tt
		
        ddux = self.ddux
        dduy = self.dduy
        dduz = self.dduz		
        ddss = self.ddss

        dduxux = self.dduxux
        dduyuy = self.dduyuy
        dduzuz = self.dduzuz		
        ddssux = self.ddssux
		
        ddssux   = self.ddssux
        ddssuy   = self.ddssuy
        ddssuz   = self.ddssuz		
		
        ddssuxux   = self.ddssuxux
        ddssuyuy   = self.ddssuyuy
        ddssuzuz   = self.ddssuzuz
        ddenuc1_tt    = self.ddenuc1_tt
        ddenuc2_tt    = self.ddenuc2_tt
        dduxenuc1_tt  = self.dduxenuc1_tt
        dduxenuc2_tt  = self.dduxenuc2_tt
        ssgradxpp  = self.ssgradxpp

        ddss     = self.ddss		
        ppdivu   = self.ppdivu
        uxppdivu = self.uxppdivu
		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_ss   = ddss/dd
        rxx  = dduxux - ddux*ddux/dd 		

        f_ss       = ddssux - ddux*ddss/dd
        fr_ss      = ddssuxux - ddss*dduxux/dd - 2.*fht_ux*ddssux + 2.*dd*fht_ux*fht_ss*fht_ux

        ssff = ss - ddss/dd				
        ssff_gradx_ppf = ssgradxpp - ss*self.Grad(pp,xzn0)
		
        uxff_dd_enuc_T = (dduxenuc1_tt + dduxenuc2_tt) - fht_ux*(ddenuc1_tt + ddenuc2_tt)
		
        uxff_epsilonk_approx = (ux - ddux/dd)*tke_diss
		
        Grss = -(ddssuyuy-ddss*dduyuy/dd-2.*(dduy/dd)*(ddssuy/dd)+2.*ddss*dduy*dduy/(dd*dd*dd))/xzn0- \
                (ddssuzuz-ddss*dduzuz/dd-2.*(dduz/dd)*(ddssuz/dd)+2.*ddss*dduz*dduz/(dd*dd*dd))/xzn0
		
        ssff_GrM = -(ddssuyuy - (ddss/dd)*dduyuy)/xzn0 - (ddssuzuz - (ddss/dd)*dduzuz)/xzn0		
		
        #######################		
		# ENTROPY FLUX EQUATION
        #######################
					   
        # time-series of entropy flux 
        t_f_ss = t_ddssux/t_dd - t_ddss*t_ddux/(t_dd*t_dd)
		
        # LHS -dq/dt 		
        self.minus_dt_f_ss = -self.dt(t_f_ss,xzn0,t_timec,intc)
     
        # LHS -div fht_ux f_ss
        self.minus_div_fht_ux_f_ss = -self.Div(fht_ux*f_ss,xzn0)	 
		
        # RHS -div flux internal energy flux
        self.minus_div_fr_ss = -self.Div(fr_ss,xzn0)
        
		# RHS -f_ss_gradx_fht_ux
        self.minus_f_ss_gradx_fht_ux = -f_ss*self.Grad(fht_ux,xzn0)
		
		# RHS -rxx_gradx_fht_ss
        self.minus_rxx_gradx_fht_ss = -rxx*self.Grad(fht_ss,xzn0)	

		# RHS -eht_ssff_gradx_eht_pp
        self.minus_eht_ssff_gradx_eht_pp = -(ss - ddss/dd)*self.Grad(pp,xzn0)
		
        # RHS -eht_ssff_gradx_ppf
        self.minus_eht_ssff_gradx_ppf = -(ssgradxpp - (ddss/dd)*self.Grad(pp,xzn0))

        # RHS eht_uxff_dd_nuc_T	
        self.plus_eht_uxff_dd_nuc_T =  (dduxenuc1_tt + dduxenuc2_tt) - fht_ux*(ddenuc1_tt+ddenuc2_tt) 		

		# RHS eht_uxff_div_ftt_T (not calculated)
        eht_uxff_div_f_tt_T = np.zeros(self.nx)  		
        self.plus_eht_uxff_div_ftt_T = eht_uxff_div_f_tt_T
		
        # RHS eht_uxff_epsilonk_approx_T	
        self.plus_eht_uxff_epsilonk_approx_T =  (ux - fht_ux)*tke_diss/tt 		

        # RHS Gss
        self.plus_Gss = -Grss-ssff_GrM	

        # -res  
        self.minus_resSSfluxEquation = -(self.minus_dt_f_ss + self.minus_div_fht_ux_f_ss + \
          self.minus_div_fr_ss + self.minus_f_ss_gradx_fht_ux + self.minus_rxx_gradx_fht_ss + \
          self.minus_eht_ssff_gradx_eht_pp + self.minus_eht_ssff_gradx_ppf +\
          self.plus_eht_uxff_dd_nuc_T + self.plus_eht_uxff_div_ftt_T + self.plus_eht_uxff_epsilonk_approx_T + \
          self.plus_Gss)

        ###########################		
		# END ENTROPY FLUX EQUATION
        ###########################

    def plot_fss(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian entropy flux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddssux - self.ddux*self.ddss/self.dd
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'entropy flux')
        plt.plot(grd1,plt1,color='brown',label = r'f$_s$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$f_s$ (erg K$^{-1}$ cm$^{-2}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_fss.png')			
		  
    def plot_fss_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot entropy flux equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_f_ss  
        lhs1 = self.minus_div_fht_ux_f_ss 
		
        rhs0 = self.minus_div_fr_ss
        rhs1 = self.minus_f_ss_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_ss
        rhs3 = self.minus_eht_ssff_gradx_eht_pp
        rhs4 = self.minus_eht_ssff_gradx_ppf
        rhs5 = self.plus_eht_uxff_dd_nuc_T
        rhs6 = self.plus_eht_uxff_div_ftt_T
        rhs7 = self.plus_eht_uxff_epsilonk_approx_T
        rhs8 = self.plus_Gss		

        res = self.minus_resSSfluxEquation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,rhs8,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('entropy flux equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t f_s$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\widetilde{u}_r f_s$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_s^r $")     
        plt.plot(grd1,rhs1,color='#802A2A',label = r"$-f_s \partial_r \widetilde{u}_r$") 
        plt.plot(grd1,rhs2,color='r',label = r"$-\widetilde{R}_{rr} \partial_r \widetilde{s}$") 
        plt.plot(grd1,rhs3,color='c',label = r"$-\overline{s''} \ \partial_r \overline{P}$")
        plt.plot(grd1,rhs4,color='mediumseagreen',label = r"$- \overline{s''\partial_r P'}$")
        plt.plot(grd1,rhs5,color='b',label = r"$+\overline{u''_r \rho \varepsilon_{nuc} /T}$")
        plt.plot(grd1,rhs6,color='m',label = r"$+\overline{u''_r \nabla \cdot T /T}$")
        plt.plot(grd1,rhs7,color='g',label = r"$+\overline{u''_r \varepsilon_k /T}$")
        plt.plot(grd1,rhs8,color='y',label = r"$+G_s$")

		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_fs$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg K$^{-1}$ cm$^{-2}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'fss_eq.png')		
		
	
		
