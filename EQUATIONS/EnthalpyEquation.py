import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class EnthalpyEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(EnthalpyEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	
        nx     = np.asarray(eht.item().get('nx')) 
		
        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd     = np.asarray(eht.item().get('dd')[intc])
        ux     = np.asarray(eht.item().get('ux')[intc])	
        hh     = np.asarray(eht.item().get('hh')[intc])
        pp     = np.asarray(eht.item().get('pp')[intc])
		
        ddux   = np.asarray(eht.item().get('ddux')[intc])		
        ddhh   = np.asarray(eht.item().get('ddhh')[intc])
        ddhhux = np.asarray(eht.item().get('ddhhux')[intc])
		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])

        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])		
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])

        gamma1 = np.asarray(eht.item().get('gamma1')[intc])		
        gamma3 = np.asarray(eht.item().get('gamma3')[intc])
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddhh    = np.asarray(eht.item().get('ddhh')) 	
        t_fht_hh = t_ddhh/t_dd		
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd
        fht_hh = ddhh/dd
        fhh = ddhhux - ddux*ddhh/dd
		
        ###################
        # ENTHALPY EQUATION 
        ###################

        # LHS -dq/dt 		
        self.minus_dt_dd_fht_hh = -self.dt(t_dd*t_fht_hh,xzn0,t_timec,intc)	

        # LHS -div dd fht_ux fht_hh		
        self.minus_div_dd_fht_ux_fht_hh = -self.Div(dd*fht_ux*fht_hh,xzn0)
		
        # RHS -div fhh
        self.minus_div_fhh = -self.Div(fhh,xzn0)
		
        # RHS -gamma1 P d = - gamma1 pp Div ux
        self.minus_gamma1_pp_div_ux = -gamma1*pp*self.Div(ux,xzn0)		
				
        # RHS -gamma1 Wp = -gamma1 eht_ppf_df
        self.minus_gamma1_eht_ppf_df = -gamma1*(ppdivu - pp*divu)
		
        # RHS source + gamma3 dd enuc
        self.plus_gamma3_dd_fht_enuc = gamma3*(ddenuc1+ddenuc2)		
		
        # RHS gamma3 dissipated turbulent kinetic energy
        self.plus_gamma3_disstke = +gamma3*tke_diss  	

        # RHS gamma3 div ft
        self.plus_gamma3_div_ft = +np.zeros(nx)		
		
        # -res
        self.minus_resHHequation = -(self.minus_dt_dd_fht_hh+self.minus_div_dd_fht_ux_fht_hh+self.minus_div_fhh+\
         self.minus_gamma1_pp_div_ux+self.minus_gamma1_eht_ppf_df+self.plus_gamma3_dd_fht_enuc+self.plus_gamma3_disstke+\
         self.plus_gamma3_div_ft)
		
        #######################
        # END ENTHALPY EQUATION 
        #######################			
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.fht_hh      = fht_hh			
		
    def plot_hh(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian enthalpy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.fht_hh
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'enthalpy')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{h}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{h}$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_hh.png')
	

    def plot_hh_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot enthalpy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_fht_hh
        lhs1 = self.minus_div_dd_fht_ux_fht_hh
		
        rhs0 = self.minus_div_fhh
        rhs1 = self.minus_gamma1_pp_div_ux
        rhs2 = self.minus_gamma1_eht_ppf_df
        rhs3 = self.plus_gamma3_dd_fht_enuc
        rhs4 = self.plus_gamma3_disstke
        rhs5 = self.plus_gamma3_div_ft
		
        res = self.minus_resHHequation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('enthalpy equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{\rho} \widetilde{h})$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{h}$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_h $")     
        plt.plot(grd1,rhs1,color='#802A2A',label = r"$-\Gamma_1 \bar{P} \bar{d}$") 
        plt.plot(grd1,rhs2,color='r',label = r"$-\Gamma_1 W_P$")
        plt.plot(grd1,rhs3,color='b',label = r"$+\Gamma_3 \overline{\rho}\widetilde{\epsilon}_{nuc}$")
        plt.plot(grd1,rhs4,color='m',label = r"$+\Gamma_3 \varepsilon_k$")
        plt.plot(grd1,rhs5,color='c',label = r"$+\Gamma_3 \nabla_r f_T$ (not incl.)") 		
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_h$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hh_eq.png')		
		
		
		
