import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class EntropyFluxEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(EntropyFluxEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 
        nx   = np.asarray(eht.item().get('nx'))		

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])	
        pp = np.asarray(eht.item().get('pp')[intc])
        ss = np.asarray(eht.item().get('ss')[intc])	
        tt = np.asarray(eht.item().get('tt')[intc])
		
        ddux = np.asarray(eht.item().get('ddux')[intc])		
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])		
        ddss = np.asarray(eht.item().get('ddss')[intc])
		
        dduxux = np.asarray(eht.item().get('dduxux')[intc])		
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])	
		
        ddssux = np.asarray(eht.item().get('ddssux')[intc])
        ddssuy = np.asarray(eht.item().get('ddssuy')[intc])
        ddssuz = np.asarray(eht.item().get('ddssuz')[intc])		

        ddssuxux = np.asarray(eht.item().get('ddssuxux')[intc])
        ddssuyuy = np.asarray(eht.item().get('ddssuyuy')[intc])
        ddssuzuz = np.asarray(eht.item().get('ddssuzuz')[intc])
		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])

        ddenuc1_o_tt = np.asarray(eht.item().get('ddenuc1_o_tt')[intc])		
        ddenuc2_o_tt = np.asarray(eht.item().get('ddenuc2_o_tt')[intc])

        dduxenuc1_o_tt = np.asarray(eht.item().get('dduxenuc1_o_tt')[intc])		
        dduxenuc2_o_tt = np.asarray(eht.item().get('dduxenuc2_o_tt')[intc])
		
        ssgradxpp = np.asarray(eht.item().get('ssgradxpp')[intc])				
		
        ppdivu   = np.asarray(eht.item().get('ppdivu')[intc])		
        uxppdivu = np.asarray(eht.item().get('uxppdivu')[intc])		
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd'))
        t_ddux    = np.asarray(eht.item().get('ddux')) 
        t_ddss    = np.asarray(eht.item().get('ddss'))
        t_ddssux  = np.asarray(eht.item().get('ddssux')) 		
		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_ss   = ddss/dd
        rxx      = dduxux - ddux*ddux/dd 		

        f_ss  = ddssux - ddux*ddss/dd
        fr_ss = ddssuxux - ddss*dduxux/dd - 2.*fht_ux*ddssux + 2.*dd*fht_ux*fht_ss*fht_ux

        ssff = ss - ddss/dd				
        ssff_gradx_ppf = ssgradxpp - ss*self.Grad(pp,xzn0)
		
        uxff_dd_enuc_T = (dduxenuc1_o_tt + dduxenuc2_o_tt) - fht_ux*(ddenuc1_o_tt + ddenuc2_o_tt)
		
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
        self.plus_eht_uxff_dd_nuc_T =  (dduxenuc1_o_tt + dduxenuc2_o_tt) - fht_ux*(ddenuc1_o_tt+ddenuc2_o_tt) 		

        # RHS eht_uxff_div_ftt_T (not calculated)
        eht_uxff_div_f_o_tt_T = np.zeros(nx)  		
        self.plus_eht_uxff_div_ftt_T = eht_uxff_div_f_o_tt_T
		
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
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.f_ss        = f_ss			
		

    def plot_fss(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian entropy flux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.f_ss
				
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
        setylabel = r"$f_s$ (erg K$^{-1}$ cm$^{-2}$ s$^{-1}$)"
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
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'fss_eq.png')		
		
	
		
