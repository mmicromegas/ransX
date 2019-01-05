import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import CALCULUS as calc
import ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

# https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/

class EntropyEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(EntropyEquation,self).__init__(ig) 
	
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
        self.tt        = np.asarray(eht.item().get('tt')[intc])		
		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])		
        self.ddei      = np.asarray(eht.item().get('ddei')[intc])
        self.ddss      = np.asarray(eht.item().get('ddss')[intc])		
        self.ddssux    = np.asarray(eht.item().get('ddssux')[intc])

        self.ddenuc1_tt      = np.asarray(eht.item().get('ddenuc1_tt')[intc])		
        self.ddenuc2_tt      = np.asarray(eht.item().get('ddenuc2_tt')[intc])
		
        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddss    = np.asarray(eht.item().get('ddss')) 		

 		# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	
		
        dd = self.dd
        ux = self.ux
        pp = self.pp
        tt = self.tt
		
        ddux   = self.ddux
        ddei   = self.ddei
        ddss   = self.ddss
        ddssux = self.ddssux		
 
        ddenuc1_tt = self.ddenuc1_tt
        ddenuc2_tt = self.ddenuc2_tt
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd
        fht_ss = ddss/dd
        f_ss = ddssux - ddux*ddss/dd
		
        ##################
        # ENTROPY EQUATION 
        ##################
		
        # LHS -dq/dt 		
        self.minus_dt_eht_dd_fht_ss = -self.dt(t_ddss,xzn0,t_timec,intc)	

        # LHS -div eht_dd fht_ux fht_ss		
        self.minus_div_eht_dd_fht_ux_fht_ss = -self.Div(dd*fht_ux*fht_ss,xzn0)		
		
        # RHS -div fss
        self.minus_div_fss = -self.Div(f_ss,xzn0)		
		
        # RHS -div ftt / T (not included)
        self.minus_div_ftt_T = -np.zeros(self.nx)
		
        # RHS +rho enuc / T
        self.plus_eht_dd_enuc_T = ddenuc1_tt+ddenuc2_tt
		
        # RHS approx. +diss tke / T
        self.plus_disstke_T_approx = tke_diss/tt

        # -res		
        self.minus_resSequation = -(self.minus_dt_eht_dd_fht_ss + self.minus_div_eht_dd_fht_ux_fht_ss + \
         self.minus_div_fss + self.minus_div_ftt_T + self.plus_eht_dd_enuc_T + self.plus_disstke_T_approx)
		
        ######################
        # END ENTROPY EQUATION 
        ######################					
		
		
    def plot_ss(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian entropy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddss/self.dd
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title(r'entropy')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{s}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{s}$ (erg g$^{-1}$ K$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_ss.png')
		
    def plot_ss_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot entropy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_fht_ss
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_ss
		
        rhs0 = self.minus_div_fss
        rhs1 = self.minus_div_ftt_T		
        rhs2 = self.plus_eht_dd_enuc_T
        rhs3 = self.plus_disstke_T_approx
		
        res = self.minus_resSequation
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('entropy equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{\rho} \widetilde{s} )$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{s}$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_s $")     
        plt.plot(grd1,rhs1,color='c',label = r"$-\overline{\nabla_r f_T /T}$ (not incl.)") 
        plt.plot(grd1,rhs2,color='b',label = r"$+\overline{\rho\epsilon_{nuc}/T}$")
        plt.plot(grd1,rhs3,color='m',label = r"$+\varepsilon_k/T$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_s$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$ s$^{-1}$ K$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'ss_eq.png')		
		