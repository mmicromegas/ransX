import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseTemperatureEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(HsseTemperatureEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	
        nx     = np.asarray(eht.item().get('nx')) 
		
        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd     = np.asarray(eht.item().get('dd')[intc])
        ux     = np.asarray(eht.item().get('ux')[intc])	
        tt     = np.asarray(eht.item().get('tt')[intc])
        cv     = np.asarray(eht.item().get('cv')[intc])
        gg     = np.asarray(eht.item().get('gg')[intc])
        pp     = np.asarray(eht.item().get('pp')[intc])
		
        ddux = np.asarray(eht.item().get('ddux')[intc])		
        ttux = np.asarray(eht.item().get('ttux')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])	
		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        ttdivu = np.asarray(eht.item().get('ttdivu')[intc])
        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])
		
        enuc1_o_cv = np.asarray(eht.item().get('enuc1_o_cv')[intc])		
        enuc2_o_cv = np.asarray(eht.item().get('enuc2_o_cv')[intc])
		
        gamma1  = np.asarray(eht.item().get('gamma1')[intc])		
        gamma3  = np.asarray(eht.item().get('gamma3')[intc])				
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_tt      = np.asarray(eht.item().get('tt')) 
		
        #t_mm    = np.asarray(eht.item().get('mm')) 		
        #minus_dt_mm = -self.dt(t_mm,xzn0,t_timec,intc)		
        #fht_ux = minus_dt_mm/(4.*np.pi*(xzn0**2.)*dd)			
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd		
        ftt    = ttux - tt*ux

        fht_rxx = dduxux - ddux*ddux/dd
        fdil = (uxdivu - ux*divu) 	
		
        gg = -gg		
		
        ##########################
        # HSS TEMPERATURE EQUATION 
        ##########################

        # LHS -gradx T
        self.minus_gradx_tt = -self.Grad(tt,xzn0) 		
		
        # RHS -dq/dt o ux 		
        self.minus_dt_tt_o_ux = -self.dt(t_tt,xzn0,t_timec,intc)/ux	
		
        # RHS -div ftt o ux
        self.minus_div_ftt_o_ux = -self.Div(ftt,xzn0)/ux
		
        # RHS +(1-gamma3) T d = +(1-gamma3) tt Div eht_ux
        self.plus_one_minus_gamma3_tt_div_ux_o_ux = +(1.-gamma3)*tt*self.Div(ux,xzn0)/ux		
				
        # RHS +(2-gamma3) Wt = +(2-gamma3) eht_ttf_df
        self.plus_two_minus_gamma3_eht_ttf_df_o_ux = +(2.-gamma3)*(ttdivu - tt*divu)/ux
		
        # RHS source +enuc/cv
        self.plus_enuc_o_cv_o_ux = enuc1_o_cv/ux+enuc2_o_cv/ux		
		
        # RHS +dissipated turbulent kinetic energy _o_ cv (this is a guess)
        self.plus_disstke_o_cv_o_ux = +(tke_diss/(dd*cv))/ux  	
		
        # RHS +div ftt/dd cv (not included)	
        self.plus_div_ftt_o_dd_cv_o_ux = np.zeros(nx)/ux			
		
        # -res
        self.minus_resHSSTTequation = -(self.minus_gradx_tt+self.minus_dt_tt_o_ux+\
         self.minus_div_ftt_o_ux+self.plus_one_minus_gamma3_tt_div_ux_o_ux+\
         self.plus_two_minus_gamma3_eht_ttf_df_o_ux+self.plus_enuc_o_cv_o_ux+\
         self.plus_disstke_o_cv_o_ux + self.plus_div_ftt_o_dd_cv_o_ux)
		
        ##########################
        # END TEMPERATURE EQUATION 
        ##########################			
		
        #################################
        # ALTERNATIVE CONTINUITY EQUATION 
        #################################		

        self.minus_gamma3_minus_one_tt_dd_fdil_o_fht_rxx = -(gamma3-1.)*tt*dd*fdil/fht_rxx
		
        self.minus_resHSSTTequation2 = -(self.minus_gradx_tt+self.minus_gamma3_minus_one_tt_dd_fdil_o_fht_rxx)		
		
        #####################################
        # END ALTERNATIVE CONTINUITY EQUATION 
        #####################################		
		
        ############################################
        # ALTERNATIVE CONTINUITY EQUATION SIMPLIFIED
        ############################################		

        self.minus_gamma3_minus_one_dd_tt_gg_o_gamma1_pp = -(gamma3-1.)*dd*tt*gg/(gamma1*pp)
		
        self.minus_resHSSTTequation3 = -(self.minus_gradx_tt+self.minus_gamma3_minus_one_dd_tt_gg_o_gamma1_pp)		
		
        ################################################
        # END ALTERNATIVE CONTINUITY EQUATION SIMPLIFIED
        ################################################			
		
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.tt          = tt			
		
    def plot_tt(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean temperature stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.tt
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'temperature')
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{T}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{T} (K)$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_tt.png')
	

    def plot_tt_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot temperature equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_tt

        rhs0 = self.minus_dt_tt_o_ux
        rhs1 = self.minus_div_ftt_o_ux
        rhs2 = self.plus_one_minus_gamma3_tt_div_ux_o_ux 
        rhs3 = self.plus_two_minus_gamma3_eht_ttf_df_o_ux
        rhs4 = self.plus_enuc_o_cv_o_ux
        rhs5 = self.plus_disstke_o_cv_o_ux
        rhs6 = self.plus_div_ftt_o_dd_cv_o_ux 

        res = self.minus_resHSSTTequation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('hsse temperature equation')
        plt.plot(grd1,lhs0,color='olive',label = r"$-\partial_r (\overline{T})$")		
        plt.plot(grd1,rhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{T})/ \overline{u}_r$")
        plt.plot(grd1,rhs1,color='#FF8C00',label = r"$-\nabla_r f_T/ \overline{u}_r $")     
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$+(1-\Gamma_3) \bar{T} \bar{d}/ \overline{u}_r$") 
        plt.plot(grd1,rhs3,color='r',label = r"$+(2-\Gamma_3) \overline{T'd'} / \overline{u}_r$")
        plt.plot(grd1,rhs4,color='b',label = r"$+(\overline{\epsilon_{nuc} / cv}/ \overline{u}_r$")
        plt.plot(grd1,rhs5,color='g',label = r"$+(\overline{\varepsilon / cv})/ \overline{u}_r$")
        plt.plot(grd1,rhs6,color='m',label = r"+$(\nabla \cdot F_T/ \rho c_v)/ \overline{u}_r$ (not incl.)")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_T$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"K cm$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hsse_temperature_eq.png')		
		
    def plot_tt_equation_2(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot temperature equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_tt

        rhs0 = self.minus_gamma3_minus_one_tt_dd_fdil_o_fht_rxx

        res = self.minus_resHSSTTequation2
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('alternative hsse temperature equation')
        plt.plot(grd1,lhs0,color='olive',label = r"$-\partial_r (\overline{T})$")		
        plt.plot(grd1,rhs0,color='m',label = r"$-(\Gamma_3-1) \ \overline{\rho} \ \overline{T} \ \overline{u'_r d''} / \ \widetilde{R}_{rr}$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"K cm$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hsse_temperature_eq_alternative.png')	
		
    def plot_tt_equation_3(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot temperature equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_tt

        rhs0 = self.minus_gamma3_minus_one_dd_tt_gg_o_gamma1_pp

        res = self.minus_resHSSTTequation3
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        #plt.title('alternative hsse temperature equation simp')
        plt.plot(grd1,lhs0,color='olive',label = r"$-\partial_r (\overline{T})$")		
        plt.plot(grd1,rhs0,color='m',label = r"$-(\Gamma_3-1) \ \overline{\rho} \ \overline{T} \ \overline{g}_r / \Gamma_1 \overline{P}$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"K cm$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':14})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hsse_temperature_eq_alternative_simplified.eps')			
		
		
