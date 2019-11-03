import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class PressureFluxZequation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,tke_diss,data_prefix):
        super(PressureFluxZequation,self).__init__(ig) 
	
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
				
        gradxpp_o_dd = np.asarray(eht.item().get('gradxpp_o_dd')[intc])		
        ppgradxpp_o_dd = np.asarray(eht.item().get('ppgradxpp_o_dd')[intc])		

        #gradzpp_o_ddsiny = np.asarray(eht.item().get('gradzpp_o_ddsiny')[intc])		
        #ppgradzpp_o_ddsiny = np.asarray(eht.item().get('ppgradzpp_o_ddsiny')[intc])

        ppuzuycoty = np.asarray(eht.item().get('ppuzuycoty')[intc])		
        uzuycoty = np.asarray(eht.item().get('uzuycoty')[intc])
		
        gradzpp_o_ddsiny = np.asarray(eht.item().get('gradzpp_o_ddsiny')[intc])		
        ppgradzpp_o_ddsiny = np.asarray(eht.item().get('ppgradzpp_o_ddsiny')[intc])
		
        # store time series for time derivatives
        t_timec = np.asarray(eht.item().get('timec'))		
        t_uz    = np.asarray(eht.item().get('uz'))
        t_pp    = np.asarray(eht.item().get('pp'))
        t_ppuz  = np.asarray(eht.item().get('ppuz')) 		
		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_uy   = dduy/dd
        fht_uz   = dduz/dd
		
        fht_ppux = ddppux/dd
        fht_ppuy = ddppuy/dd
        fht_ppuz = ddppuz/dd
		
        fht_divu = dddivu/dd
		
        eht_uzf_uxff = uxuz - ux*uz
		
        eht_ppf_uzff_divuff	= uzppdivu - fht_ppuz*divu - pp*uzdivu - pp*fht_uz*divu - ppuz*fht_divu	+ \
            fht_ppuz*fht_divu + pp*uz*fht_divu + pp*fht_uz*fht_divu

        fppx  = ppux-pp*ux
        fppy  = ppuy-pp*uy
        fppz  = ppuz-pp*uz
		
        fppzx = ppuzux - ppuz*ux - pp*uxuz + pp*uz*ux - ppuz*fht_ux + pp*uz*fht_ux

			
        ########################		
        # PRESSURE FLUX EQUATION
        ########################
					   
        # time-series of pressure flux 
        t_fppz = t_ppuz - t_pp*t_uz
		
        # LHS -dq/dt 		
        self.minus_dt_fppz = -self.dt(t_fppz,xzn0,t_timec,intc)
     
        # LHS -fht_ux gradx fppz
        self.minus_fht_ux_gradx_fppz = -fht_ux*self.Grad(fppz,xzn0)	 
		
        # RHS -div pressure flux in z
        self.minus_div_fppzx = -self.Div(fppzx,xzn0)
        
        # RHS -fppx_gradx_uz
        self.minus_fppx_gradx_uz = -fppx*self.Grad(uz,xzn0)
		
        # RHS +eht_uzf_uxff_gradx_pp
        self.plus_eht_uzf_uxff_gradx_pp = +eht_uzf_uxff*self.Grad(pp,xzn0)	

        # RHS +gamma1_eht_uzf_pp_divu
        self.plus_gamma1_eht_uzf_pp_divu = +gamma1*(uzppdivu - uz*ppdivu)		
		
        # RHS +gamma3_minus_one_eht_uyf_dd_enuc 		
        self.plus_gamma3_minus_one_eht_uzf_dd_enuc = +(gamma3-1.)*((dduzenuc1 - uz*ddenuc1)+(dduzenuc2 - uz*ddenuc2))
				
        # RHS +eht_ppf_uzff_divuff 	
        self.plus_eht_ppf_uzff_divuff = +eht_ppf_uzff_divuff 	

        # RHS -eht_ppf_GpM_o_dd	
        self.minus_eht_ppf_GpM_o_dd = -1.*(ppuzux/xzn0+ppuzuycoty/xzn0 - pp*(uxuz/xzn0+uzuycoty/xzn0))

        # RHS -eht_ppf_gradz_pp_o_ddrrsiny 		
        self.minus_eht_ppf_gradz_pp_o_ddrrsiny = -(ppgradzpp_o_ddsiny/xzn0 - pp*gradzpp_o_ddsiny/xzn0)
	
        # -res  
        self.minus_resPPfluxEquation = -(self.minus_dt_fppz+ self.minus_fht_ux_gradx_fppz+self.minus_div_fppzx+\
          self.minus_fppx_gradx_uz+self.plus_eht_uzf_uxff_gradx_pp+self.plus_gamma1_eht_uzf_pp_divu+\
          self.plus_gamma3_minus_one_eht_uzf_dd_enuc+self.plus_eht_ppf_uzff_divuff+self.minus_eht_ppf_GpM_o_dd+\
          self.minus_eht_ppf_gradz_pp_o_ddrrsiny)
                                       
        ########################		
        # PRESSURE FLUX EQUATION
        ########################
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.fppz        = fppz
		
    def plot_fppz(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean pressure flux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.fppz
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'pressure flux z')
        plt.plot(grd1,plt1,color='brown',label = r'f$_{pz}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$f_{pz}$ (erg cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_fppz.png')

									   
    def plot_fppz_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot acoustic flux equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_fppz
        lhs1 = self.minus_fht_ux_gradx_fppz
		
        rhs0 = self.minus_div_fppzx
        rhs1 = self.minus_fppx_gradx_uz
        rhs2 = self.plus_eht_uzf_uxff_gradx_pp
        rhs3 = self.plus_gamma1_eht_uzf_pp_divu
        rhs4 = self.plus_gamma3_minus_one_eht_uzf_dd_enuc
        rhs5 = self.plus_eht_ppf_uzff_divuff
        rhs6 = self.minus_eht_ppf_GpM_o_dd
        rhs7 = self.minus_eht_ppf_gradz_pp_o_ddrrsiny
	  
        res = self.minus_resPPfluxEquation
	
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
		
        # plot DATA 
        plt.title('acoustic flux z equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t f_{pz}$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\widetilde{u}_r \partial_r f_{pz}$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_p^r $")     
        plt.plot(grd1,rhs1,color='#802A2A',label = r"$-f_{pz} \partial_r \overline{u}_r$") 
        plt.plot(grd1,rhs2,color='r',label = r"$+\overline{u'_\phi u''_r} \partial_r \overline{P}$") 
        plt.plot(grd1,rhs3,color='firebrick',label = r"$+\Gamma_1 \overline{u'_\phi P d}$") 
        plt.plot(grd1,rhs4,color='c',label = r"$+(\Gamma_3-1)\overline{u'_\phi \rho \epsilon_{nuc}}$")
        plt.plot(grd1,rhs5,color='mediumseagreen',label = r"$+\overline{P'u''_\phi d''}$")
        plt.plot(grd1,rhs6,color='b',label = r"$+\overline{P' G_\phi^M/ \rho}$")
        plt.plot(grd1,rhs7,color='m',label = r"$+\overline{P'\partial_\phi P/ \rho r \sin{\theta}}$")
		
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
        plt.savefig('RESULTS/'+self.data_prefix+'fppz_eq.png')	
        plt.savefig('RESULTS/'+self.data_prefix+'fppz_eq.eps')
		
