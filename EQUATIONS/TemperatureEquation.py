import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TemperatureEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,tke_diss,data_prefix):
        super(TemperatureEquation,self).__init__(ig) 
	
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
		
        ddux = np.asarray(eht.item().get('ddux')[intc])		
        ttux = np.asarray(eht.item().get('ttux')[intc])
		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        ttdivu = np.asarray(eht.item().get('ttdivu')[intc])

        enuc1_o_cv = np.asarray(eht.item().get('enuc1_o_cv')[intc])		
        enuc2_o_cv = np.asarray(eht.item().get('enuc2_o_cv')[intc])
		
        gamma1  = np.asarray(eht.item().get('gamma1')[intc])		
        gamma3  = np.asarray(eht.item().get('gamma3')[intc])				

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if(ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])   
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma1 = cp/cv   # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            gamma3 = gamma1
            
        
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_tt      = np.asarray(eht.item().get('tt')) 
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd
        ftt    = ttux - tt*ux
		
        ######################
        # TEMPERATURE EQUATION 
        ######################

        # LHS -dq/dt 		
        self.minus_dt_tt = -self.dt(t_tt,xzn0,t_timec,intc)	

        # LHS -ux grad T		
        self.minus_ux_grad_tt = -ux*self.Grad(tt,xzn0)
		
        # RHS -div ftt
        self.minus_div_ftt = -self.Div(ftt,xzn0)
		
        # RHS +(1-gamma3) T d = +(1-gamma3) tt Div eht_ux
        self.plus_one_minus_gamma3_tt_div_ux = +(1.-gamma3)*tt*self.Div(ux,xzn0)		
				
        # RHS +(2-gamma3) Wt = +(2-gamma3) eht_ttf_df
        self.plus_two_minus_gamma3_eht_ttf_df = +(2.-gamma3)*(ttdivu - tt*divu)
		
        # RHS source +enuc/cv
        self.plus_enuc_o_cv = enuc1_o_cv+enuc2_o_cv		
		
        # RHS +dissipated turbulent kinetic energy _o_ cv (this is a guess)
        self.plus_disstke_o_cv = +tke_diss/(dd*cv)  	
		
        # RHS +div ftt/dd cv (not included)	
        self.plus_div_ftt_o_dd_cv = np.zeros(nx)		

        # RHS +viscous tensor grad u / dd cv
        #self.plus_tau_grad_u_o_dd_cv = np.zeros(nx)		
		
        # -res
        self.minus_resTTequation = -(self.minus_dt_tt+self.minus_ux_grad_tt+\
         self.minus_div_ftt+self.plus_one_minus_gamma3_tt_div_ux+self.plus_two_minus_gamma3_eht_ttf_df+\
         self.plus_enuc_o_cv+self.plus_disstke_o_cv + self.plus_div_ftt_o_dd_cv)
		
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
        setylabel = r"$\overline{T} (K)$"
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

        lhs0 = self.minus_dt_tt
        lhs1 = self.minus_ux_grad_tt
		
        rhs0 = self.minus_div_ftt
        rhs1 = self.plus_one_minus_gamma3_tt_div_ux 
        rhs2 = self.plus_two_minus_gamma3_eht_ttf_df
        rhs3 = self.plus_enuc_o_cv
        rhs4 = self.plus_disstke_o_cv
        rhs5 = self.plus_div_ftt_o_dd_cv 
		
        res = self.minus_resTTequation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('temperature equation')
        if (self.ig == 1):
            plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{T})$")
            plt.plot(grd1,lhs1,color='k',label = r"$-\overline{u}_x \partial_x \overline{T}$")	
		
            plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_x f_T $")     
            plt.plot(grd1,rhs1,color='#802A2A',label = r"$+(1-\Gamma_3) \bar{T} \bar{d}$") 
            plt.plot(grd1,rhs2,color='r',label = r"$+(2-\Gamma_3) \overline{T'd'}$")
            plt.plot(grd1,rhs3,color='b',label = r"$+\overline{\epsilon_{nuc} / cv}$")
            plt.plot(grd1,rhs4,color='g',label = r"$+\overline{\varepsilon / cv}$")
            plt.plot(grd1,rhs5,color='m',label = r"+$\nabla \cdot F_T/ \rho c_v$ (not incl.)")	
		
            plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_T$")
            # define X label
            setxlabel = r"x (cm)"			
        elif(self.ig == 2):
            plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{T})$")
            plt.plot(grd1,lhs1,color='k',label = r"$-\overline{u}_r \partial_r \overline{T}$")	
		
            plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_T $")     
            plt.plot(grd1,rhs1,color='#802A2A',label = r"$+(1-\Gamma_3) \bar{T} \bar{d}$") 
            plt.plot(grd1,rhs2,color='r',label = r"$+(2-\Gamma_3) \overline{T'd'}$")
            plt.plot(grd1,rhs3,color='b',label = r"$+\overline{\epsilon_{nuc} / cv}$")
            plt.plot(grd1,rhs4,color='g',label = r"$+\overline{\varepsilon / cv}$")
            plt.plot(grd1,rhs5,color='m',label = r"+$\nabla \cdot F_T/ \rho c_v$ (not incl.)")	
		
            plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_T$")		
            # define X label
            setxlabel = r"r (cm)"
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()  
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"K s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'tt_eq.png')		
		
		
		
