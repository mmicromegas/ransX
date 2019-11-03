import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TemperatureFluxEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,tke_diss,data_prefix):
        super(TemperatureFluxEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0'))
        nx     = np.asarray(eht.item().get('nx')) 		

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])	
        pp = np.asarray(eht.item().get('pp')[intc])
        tt = np.asarray(eht.item().get('tt')[intc])
        cv = np.asarray(eht.item().get('cv')[intc])
		
        ux = np.asarray(eht.item().get('ux')[intc])		
        uy = np.asarray(eht.item().get('uy')[intc])
        uz = np.asarray(eht.item().get('uz')[intc])		
		
        ddux = np.asarray(eht.item().get('ddux')[intc])		
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])		
		
        uxux = np.asarray(eht.item().get('uxux')[intc])		
        uyuy = np.asarray(eht.item().get('uyuy')[intc])
        uzuz = np.asarray(eht.item().get('uzuz')[intc])	
		
        ttux = np.asarray(eht.item().get('ttux')[intc])
        ttuy = np.asarray(eht.item().get('ttuy')[intc])
        ttuz = np.asarray(eht.item().get('ttuz')[intc])		

        ttuxux = np.asarray(eht.item().get('ttuxux')[intc])
        ttuyuy = np.asarray(eht.item().get('ttuyuy')[intc])
        ttuzuz = np.asarray(eht.item().get('ttuzuz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])		
		
        ddttuxux = np.asarray(eht.item().get('ddttuxux')[intc])
        ddttuyuy = np.asarray(eht.item().get('ddttuyuy')[intc])
        ddttuzuz = np.asarray(eht.item().get('ddttuzuz')[intc])
		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        dddivu = np.asarray(eht.item().get('dddivu')[intc])
        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])
        ttdivu = np.asarray(eht.item().get('ttdivu')[intc])
		
        ttgradxpp_o_dd = np.asarray(eht.item().get('ttgradxpp_o_dd')[intc])	
        gradxpp_o_dd = np.asarray(eht.item().get('gradxpp_o_dd')[intc])		
				
        uxttdivu = np.asarray(eht.item().get('uxttdivu')[intc])		
		
        uxenuc1_o_cv = np.asarray(eht.item().get('uxenuc1_o_cv')[intc]) 
        uxenuc2_o_cv = np.asarray(eht.item().get('uxenuc2_o_cv')[intc])

        enuc1_o_cv = np.asarray(eht.item().get('enuc1_o_cv')[intc]) 
        enuc2_o_cv = np.asarray(eht.item().get('enuc2_o_cv')[intc])

        gamma3 = np.asarray(eht.item().get('gamma3')[intc])

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if(ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])   
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma3 = cp/cv   # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
        
        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))		
        t_tt    = np.asarray(eht.item().get('tt'))
        t_ux    = np.asarray(eht.item().get('ux'))		
        t_ttux  = np.asarray(eht.item().get('ttux'))  		
		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_d    = dddivu/dd
		
        ftt  = ttux - tt*ux
        fttx = ttuxux - 2.*ux*ttux - tt*uxux - 2.*tt*ux*ux
			
        eht_uxf_uxff = uxux - ux*ux 
        eht_uxf_dff = uxdivu - ux*divu
        eht_ttf_dff = ttdivu - tt*divu
		
        eht_ttf_gradx_pp_o_dd = ttgradxpp_o_dd - tt*gradxpp_o_dd 
		
        eht_uxf_ttf_dff = uxttdivu - uxdivu*tt - ux*ttdivu + ux*tt*divu	- ftt*fht_d + ux*tt*fht_d	
		
        eht_uxf_enuc_o_cv =  (uxenuc1_o_cv + uxenuc2_o_cv) - ux*(enuc1_o_cv + enuc2_o_cv)
		
        eht_uxff_epsilonk_approx_o_cv = (ux-fht_ux)*tke_diss/cv
		
        Grtt = -(ttuyuy - 2.*uy*ttuy - tt*uyuy - 2.*tt*uy*uy)/xzn0- \
                (ttuzuz - 2.*uz*ttuz - tt*uzuz - 2.*tt*uz*uz)/xzn0
		
        ttf_GrM = -(ddttuyuy - tt*dduyuy)/xzn0 - (ddttuzuz - tt*dduzuz)/xzn0		
		
        ###########################		
        # TEMPERATURE FLUX EQUATION
        ###########################
					   
        # time-series of temperature flux 
        t_ftt = t_ttux - t_tt*t_ux
		
        # LHS -dq/dt 		
        self.minus_dt_ftt = -self.dt(t_ftt,xzn0,t_timec,intc)
     
        # LHS -fht_ux gradx ftt
        self.minus_fht_ux_gradx_ftt = -fht_ux*self.Grad(ftt,xzn0)	 
		
        # RHS -div flux temperature flux
        self.minus_div_fttx = -self.Div(fttx,xzn0)
        
        # RHS -ftt_gradx_fht_ux
        self.minus_ftt_gradx_fht_ux = -ftt*self.Grad(fht_ux,xzn0)
		
        # RHS -eht_uxf_uxff_gradx_tt
        self.minus_eht_uxf_uxff_gradx_tt = -eht_uxf_uxff*self.Grad(tt,xzn0)	

        # RHS -eht_eht_ttf_gradx_pp_o_dd
        #self.minus_eht_ttf_gradx_pp_o_dd = -(eht_ttf_gradx_pp_o_dd)
        self.minus_eht_ttf_gradx_pp_o_dd = np.zeros(nx) # replace gradx pp with rho gg and you get the 0
		
        # RHS -gamma3_minus_one_tt_eht_uxf_dff
        self.minus_gamma3_minus_one_tt_eht_uxf_dff = -(gamma3-1.)*tt*eht_uxf_dff
		
        # RHS -gamma3_minus_one_fht_d_ftt
        self.minus_gamma3_minus_one_fht_d_ftt = -(gamma3-1.)*fht_d*ftt

        # RHS -gamma3_eht_uxf_ttf_dff
        self.minus_gamma3_eht_uxf_ttf_dff = -gamma3*eht_uxf_ttf_dff
		    
        # RHS eht_uxf_enuc_o_cv	
        self.plus_eht_uxf_enuc_o_cv = (uxenuc1_o_cv + uxenuc2_o_cv) - ux*(enuc1_o_cv + enuc2_o_cv) 
		
        # RHS eht_uxf_div_fth_o_cv (not calculated)
		# fth is flux due to thermal transport (conduction/radiation)
        eht_uxf_div_fth_o_cv = np.zeros(nx)  		
        self.plus_eht_uxf_div_fth_o_cv = eht_uxf_div_fth_o_cv		

        # RHS Gtt
        #self.plus_Gtt = -Grtt-ttf_GrM	
        self.plus_Gtt = np.zeros(nx)		

        # -res  
        self.minus_resTTfluxEquation = -(self.minus_dt_ftt+self.minus_fht_ux_gradx_ftt+\
         self.minus_div_fttx+self.minus_ftt_gradx_fht_ux+self.minus_eht_uxf_uxff_gradx_tt+\
         self.minus_eht_ttf_gradx_pp_o_dd+self.minus_gamma3_minus_one_tt_eht_uxf_dff+\
         self.minus_gamma3_minus_one_fht_d_ftt+self.minus_gamma3_eht_uxf_ttf_dff+\
         self.plus_eht_uxf_enuc_o_cv+self.plus_eht_uxf_div_fth_o_cv+self.plus_Gtt)
                                       
        ###############################		
        # END TEMPERATURE FLUX EQUATION
        ###############################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.ftt         = ftt
		
    def plot_ftt(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot temperature flux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ftt
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'temperature flux')
        plt.plot(grd1,plt1,color='brown',label = r'f$_T$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$f_T$ (K cm s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_ftt.png')

									   
    def plot_ftt_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot temperature flux equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_ftt
        lhs1 = self.minus_fht_ux_gradx_ftt
		
        rhs0 = self.minus_div_fttx
        rhs1 = self.minus_ftt_gradx_fht_ux
        rhs2 = self.minus_eht_uxf_uxff_gradx_tt
        rhs3 = self.minus_eht_ttf_gradx_pp_o_dd
        rhs4 = self.minus_gamma3_minus_one_tt_eht_uxf_dff
        rhs5 = self.minus_gamma3_minus_one_fht_d_ftt
        rhs6 = self.minus_gamma3_eht_uxf_ttf_dff
        rhs7 = self.plus_eht_uxf_enuc_o_cv
        rhs8 = self.plus_eht_uxf_div_fth_o_cv
        rhs9 = self.plus_Gtt
	  
        res = self.minus_resTTfluxEquation
	
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs3,rhs2,rhs4,rhs5,rhs6,rhs7,rhs8,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
		
        # plot DATA 
        plt.title('temperature flux equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t f_T$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\widetilde{u}_r \partial_r f_T$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_T^r $")     
        plt.plot(grd1,rhs1,color='#802A2A',label = r"$-f_T \partial_r \widetilde{u}_r$") 
        plt.plot(grd1,rhs2,color='r',label = r"$-\overline{u'_r u''_r} \partial_r \overline{T}$") 
        plt.plot(grd1,rhs3,color='firebrick',label = r"$-\overline{T'\partial_r P / \rho}$") 
        plt.plot(grd1,rhs4,color='c',label = r"$-(\Gamma_3 -1)\overline{T} \ \overline{u'_r d''}$")
        #plt.plot(grd1,rhs2+rhs4,color='r',label = r"$-\overline{u'_r u''_r} \partial_r \overline{T}-(\Gamma_3 -1)\overline{T} \ \overline{u'_r d''}$") 		
        plt.plot(grd1,rhs5,color='mediumseagreen',label = r"$-(\Gamma_3 -1)\widetilde{d} f_T$")
        plt.plot(grd1,rhs6,color='b',label = r"$+\Gamma_3 \overline{u'_r T' d''}$")
        plt.plot(grd1,rhs7,color='g',label = r"$+\overline{u'_r \varepsilon_{nuc} / c_v }$")		
        plt.plot(grd1,rhs8,color='m',label = r"$+\overline{u'_r \nabla \cdot T / c_v}$")
        plt.plot(grd1,rhs9,color='y',label = r"$+G_T$ (not calc.)")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_T$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"K cm$^{-2}$ s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'ftt_eq.png')	
        plt.savefig('RESULTS/'+self.data_prefix+'ftt_eq.eps')
