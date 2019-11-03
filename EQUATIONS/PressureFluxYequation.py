import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class PressureFluxYequation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,tke_diss,data_prefix):
        super(PressureFluxYequation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0'))
        nx     = np.asarray(eht.item().get('nx')) 		

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])	
        uy = np.asarray(eht.item().get('uy')[intc])
        uz = np.asarray(eht.item().get('uz')[intc])		
        pp = np.asarray(eht.item().get('pp')[intc])
		
        ddux = np.asarray(eht.item().get('ddux')[intc])		
        dduy = np.asarray(eht.item().get('dduy')[intc])	
        dduz = np.asarray(eht.item().get('dduz')[intc])	
		
        ppux = np.asarray(eht.item().get('ppux')[intc])	
        ppuy = np.asarray(eht.item().get('ppuy')[intc])	
        ppuz = np.asarray(eht.item().get('ppuz')[intc])
		
        uxux = np.asarray(eht.item().get('uxux')[intc])		
        uyuy = np.asarray(eht.item().get('uyuy')[intc])
        uzuz = np.asarray(eht.item().get('uzuz')[intc])	
        uxuy = np.asarray(eht.item().get('uxuy')[intc])	
        uxuz = np.asarray(eht.item().get('uxuz')[intc])			
		
        ddppux = np.asarray(eht.item().get('ddppux')[intc])	
        ddppuy = np.asarray(eht.item().get('ddppuy')[intc])	
        ddppuz = np.asarray(eht.item().get('ddppuz')[intc])			
		
        ppuxux = np.asarray(eht.item().get('ppuxux')[intc])	
        ppuyuy = np.asarray(eht.item().get('ppuyuy')[intc])	
        ppuzuz = np.asarray(eht.item().get('ppuzuz')[intc])			
        ppuzuy = np.asarray(eht.item().get('ppuzuy')[intc])			
        ppuzux = np.asarray(eht.item().get('ppuzux')[intc])			
        ppuyux = np.asarray(eht.item().get('ppuyux')[intc])	
		
        divu   = np.asarray(eht.item().get('divu')[intc])	
		
        uxdivu   = np.asarray(eht.item().get('uxdivu')[intc])
        uydivu   = np.asarray(eht.item().get('uydivu')[intc])
        uzdivu   = np.asarray(eht.item().get('uzdivu')[intc])		

        dddivu   = np.asarray(eht.item().get('dddivu')[intc])
        ppdivu   = np.asarray(eht.item().get('ppdivu')[intc])		

        uxppdivu = np.asarray(eht.item().get('uxppdivu')[intc])	
        uyppdivu = np.asarray(eht.item().get('uyppdivu')[intc])
        uzppdivu = np.asarray(eht.item().get('uzppdivu')[intc])
		
        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])		
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])

        dduxenuc1 = np.asarray(eht.item().get('dduxenuc1')[intc])		
        dduyenuc1 = np.asarray(eht.item().get('dduyenuc1')[intc])
        dduzenuc1 = np.asarray(eht.item().get('dduzenuc1')[intc])		
		
        dduxenuc2 = np.asarray(eht.item().get('dduxenuc2')[intc])				
        dduyenuc2 = np.asarray(eht.item().get('dduyenuc2')[intc])
        dduzenuc2 = np.asarray(eht.item().get('dduzenuc2')[intc])		
		
        gamma1   = np.asarray(eht.item().get('gamma1')[intc])
        gamma3   = np.asarray(eht.item().get('gamma3')[intc])		

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if(ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])   
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma1 = cp/cv   # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1
        
        uzuzcoty = np.asarray(eht.item().get('uzuzcoty')[intc])
        ppuzuzcoty = np.asarray(eht.item().get('ppuzuzcoty')[intc])
			
        gradxpp_o_dd = np.asarray(eht.item().get('gradxpp_o_dd')[intc])		
        ppgradxpp_o_dd = np.asarray(eht.item().get('ppgradxpp_o_dd')[intc])
		
        gradypp_o_dd = np.asarray(eht.item().get('gradypp_o_dd')[intc])		
        ppgradypp_o_dd = np.asarray(eht.item().get('ppgradypp_o_dd')[intc])

		
        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))		
        t_uy    = np.asarray(eht.item().get('uy'))
        t_pp    = np.asarray(eht.item().get('pp'))
        t_ppuy  = np.asarray(eht.item().get('ppuy')) 		
		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_uy   = dduy/dd
		
        fht_ppux = ddppux/dd
        fht_ppuy = ddppuy/dd
		
        fht_divu = dddivu/dd		
        eht_uyf_uxff = uxuy - ux*uy
		
        eht_ppf_uyff_divuff	= uyppdivu - fht_ppuy*divu - pp*uydivu - pp*fht_uy*divu - ppuy*fht_divu	+ \
            fht_ppuy*fht_divu + pp*uy*fht_divu + pp*fht_uy*fht_divu

        fppx  = ppux-pp*ux
        fppy  = ppuy-pp*uy
        fppyx = ppuyux - ppuy*ux - pp*uxuy + pp*uy*ux - ppuy*fht_ux + pp*uy*fht_ux

			
        ########################		
        # PRESSURE FLUX EQUATION
        ########################
					   
        # time-series of pressure flux 
        t_fppy = t_ppuy - t_pp*t_uy
		
        # LHS -dq/dt 		
        self.minus_dt_fppy = -self.dt(t_fppy,xzn0,t_timec,intc)
     
        # LHS -fht_ux gradx fppy
        self.minus_fht_ux_gradx_fppy = -fht_ux*self.Grad(fppy,xzn0)	 
		
        # RHS -div pressure flux in y
        self.minus_div_fppyx = -self.Div(fppyx,xzn0)
        
        # RHS -fppx_gradx_uy
        self.minus_fppx_gradx_uy = -fppx*self.Grad(uy,xzn0)
		
        # RHS +eht_uyf_uxff_gradx_pp
        self.plus_eht_uyf_uxff_gradx_pp = +eht_uyf_uxff*self.Grad(pp,xzn0)	

        # RHS +gamma1_eht_uyf_pp_divu
        self.plus_gamma1_eht_uyf_pp_divu = +gamma1*(uyppdivu - uy*ppdivu)		
		
        # RHS +gamma3_minus_one_eht_uyf_dd_enuc 		
        self.plus_gamma3_minus_one_eht_uyf_dd_enuc = +(gamma3-1.)*((dduyenuc1 - uy*ddenuc1)+(dduyenuc2 - uy*ddenuc2))
				
        # RHS +eht_ppf_uyff_divuff 	
        self.plus_eht_ppf_uyff_divuff = +eht_ppf_uyff_divuff 	

        # RHS -eht_ppf_GtM_o_dd	
        self.minus_eht_ppf_GtM_o_dd = -1.*(ppuyux/xzn0-ppuzuzcoty/xzn0 - pp*(uxuy/xzn0-uzuzcoty/xzn0))

        # RHS -eht_ppf_grady_pp_o_ddrr 		
        self.minus_eht_ppf_grady_pp_o_ddrr = -(ppgradypp_o_dd/xzn0	- pp*gradypp_o_dd/xzn0)
	
        # -res  
        self.minus_resPPfluxEquation = -(self.minus_dt_fppy+ self.minus_fht_ux_gradx_fppy+self.minus_div_fppyx+\
          self.minus_fppx_gradx_uy+self.plus_eht_uyf_uxff_gradx_pp+self.plus_gamma1_eht_uyf_pp_divu+\
          self.plus_gamma3_minus_one_eht_uyf_dd_enuc+self.plus_eht_ppf_uyff_divuff+self.minus_eht_ppf_GtM_o_dd+\
          self.minus_eht_ppf_grady_pp_o_ddrr)
                                       
        ########################		
        # PRESSURE FLUX EQUATION
        ########################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.fppy        = fppy
		
    def plot_fppy(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean pressure flux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.fppy
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'pressure flux y')
        plt.plot(grd1,plt1,color='brown',label = r'f$_{py}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$f_{py}$ (erg cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_fppy.png')
									   
    def plot_fppy_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot acoustic flux equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fppy
        lhs1 = self.minus_fht_ux_gradx_fppy
		
        rhs0 = self.minus_div_fppyx
        rhs1 = self.minus_fppx_gradx_uy
        rhs2 = self.plus_eht_uyf_uxff_gradx_pp
        rhs3 = self.plus_gamma1_eht_uyf_pp_divu
        rhs4 = self.plus_gamma3_minus_one_eht_uyf_dd_enuc
        rhs5 = self.plus_eht_ppf_uyff_divuff
        rhs6 = self.minus_eht_ppf_GtM_o_dd
        rhs7 = self.minus_eht_ppf_grady_pp_o_ddrr
	  
        res = self.minus_resPPfluxEquation
	
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
		
        # plot DATA 
        plt.title('acoustic flux y equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t f_{py}$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\widetilde{u}_r \partial_r f_{py}$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_p^r $")     
        plt.plot(grd1,rhs1,color='#802A2A',label = r"$-f_{py} \partial_r \overline{u}_r$") 
        plt.plot(grd1,rhs2,color='r',label = r"$+\overline{u'_\theta u''_r} \partial_r \overline{P}$") 
        plt.plot(grd1,rhs3,color='firebrick',label = r"$+\Gamma_1 \overline{u'_\theta P d}$") 
        plt.plot(grd1,rhs4,color='c',label = r"$+(\Gamma_3-1)\overline{u'_\theta \rho \epsilon_{nuc}}$")
        plt.plot(grd1,rhs5,color='mediumseagreen',label = r"$+\overline{P'u''_\theta d''}$")
        plt.plot(grd1,rhs6,color='b',label = r"$+\overline{P' G_\theta^M/ \rho}$")
        plt.plot(grd1,rhs7,color='m',label = r"$+\overline{P'\partial_\theta P/ \rho r}$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_p$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-2}$ s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'fppy_eq.png')	
        plt.savefig('RESULTS/'+self.data_prefix+'fppy_eq.eps')
		
