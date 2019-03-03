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
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	

        dd     = np.asarray(eht.item().get('dd')[intc])
        ux     = np.asarray(eht.item().get('ux')[intc])	
        tt     = np.asarray(eht.item().get('tt')[intc])
        cv     = np.asarray(eht.item().get('cv')[intc])
		
        ddux = np.asarray(eht.item().get('ddux')[intc])		
        ttux = np.asarray(eht.item().get('ttux')[intc])
		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        ttdivu = np.asarray(eht.item().get('ttdivu')[intc])

        enuc1_o_cv = np.asarray(eht.item().get('enuc1_o_cv')[intc])		
        enuc2_o_cv = np.asarray(eht.item().get('enuc2_o_cv')[intc])
		
        gamma1  = np.asarray(eht.item().get('gamma1')[intc])		
        gamma3  = np.asarray(eht.item().get('gamma3')[intc])				
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_tt      = np.asarray(eht.item().get('tt')) 
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd
        ftt    = ttux - tt*ux
		
        ##########################
        # HSS TEMPERATURE EQUATION 
        ##########################

        # LHS -gradx T
        self.minus_gradx_tt = -self.Grad(tt,xzn0) 		
		
        # RHS -dq/dt o ux 		
        self.minus_dt_tt_o_ux = -self.dt(t_tt,xzn0,t_timec,intc)/ux	

        # RHS -fht_ux grad T		
        self.minus_fht_ux_grad_tt_o_ux = -fht_ux*self.Grad(tt,xzn0)/ux
		
        # RHS -div ftt
        self.minus_div_ftt_o_ux = -self.Div(ftt,xzn0)/ux
		
        # RHS +(1-gamma3) T d = +(1-gamma3) tt Div eht_ux
        self.plus_one_minus_gamma3_tt_div_ux_o_ux = +(1.-gamma3)*tt*self.Div(ux,xzn0)/ux		
				
        # RHS +(2-gamma3) Wt = +(2-gamma3) eht_ttf_df
        self.plus_two_minus_gamma3_eht_ttf_df_o_ux = +(1.-gamma1)*(ttdivu - tt*divu)/ux
		
        # RHS source +enuc/cv
        self.plus_enuc_o_cv_o_ux = enuc1_o_cv/ux+enuc2_o_cv/ux		
		
        # RHS +dissipated turbulent kinetic energy _o_ cv (this is a guess)
        self.plus_disstke_o_cv_o_ux = +(tke_diss/(dd*cv))/ux  	
		
        # RHS +div ftt/dd cv (not included)	
        self.plus_div_ftt_o_dd_cv_o_ux = np.zeros(nx)/ux		

        # RHS +viscous tensor grad u / dd cv
        self.plus_tau_grad_u_o_dd_cv_o_ux = np.zeros(nx)/ux		
		
        # -res
        self.minus_resHSSTTequation = -(self.minus_gradx_tt+self.minus_dt_tt_o_ux+self.minus_fht_ux_grad_tt_o_ux+\
         self.minus_div_ftt_o_ux+self.plus_one_minus_gamma3_tt_div_ux_o_ux+\
         self.plus_two_minus_gamma3_eht_ttf_df_o_ux+self.plus_enuc_o_cv_o_ux+\
         self.plus_disstke_o_cv_o_ux + self.plus_div_ftt_o_dd_cv_o_ux + \
         self.plus_tau_grad_u_o_dd_cv_o_ux)
		
        ##########################
        # END TEMPERATURE EQUATION 
        ##########################			
		
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
        rhs1 = self.minus_fht_ux_grad_tt_o_ux
        rhs2 = self.minus_div_ftt_o_ux
        rhs3 = self.plus_one_minus_gamma3_tt_div_ux_o_ux 
        rhs4 = self.plus_two_minus_gamma3_eht_ttf_df_o_ux
        rhs5 = self.plus_enuc_o_cv_o_ux
        rhs6 = self.plus_disstke_o_cv_o_ux
        rhs7 = self.plus_div_ftt_o_dd_cv_o_ux 
        rhs8 = self.plus_tau_grad_u_o_dd_cv_o_ux
		
        res = self.minus_resHSSTTequation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,rhs8,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('hss temperature equation')
        plt.plot(grd1,lhs0,color='olive',label = r"$-\partial_r (\overline{T})$")		
        plt.plot(grd1,rhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{T})/ \overline{u}_r$")
        plt.plot(grd1,rhs1,color='k',label = r"$-\widetilde{u}_r \partial_r \overline{T}/ \overline{u}_r$")	
        plt.plot(grd1,rhs2,color='#FF8C00',label = r"$-\nabla_r f_T/ \overline{u}_r $")     
        plt.plot(grd1,rhs3,color='#802A2A',label = r"$+(1-\Gamma_3) \bar{T} \bar{d}/ \overline{u}_r$") 
        plt.plot(grd1,rhs4,color='r',label = r"$+(2-\Gamma_3) W_T/ \overline{u}_r$")
        plt.plot(grd1,rhs5,color='b',label = r"$+(\overline{\epsilon_{nuc} / cv}/ \overline{u}_r$")
        plt.plot(grd1,rhs6,color='g',label = r"$+(\overline{\varepsilon / cv})/ \overline{u}_r$")
        plt.plot(grd1,rhs7,color='m',label = r"+$(\nabla \cdot F_T/ \rho c_v)/ \overline{u}_r$")
        plt.plot(grd1,rhs8,color='pink',label = r"+$(\tau_{ij} \partial_i u_j / \rho c_v)/ \overline{u}_r$")		
		
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
        plt.savefig('RESULTS/'+self.data_prefix+'tt_eq.png')		
		
		
		
