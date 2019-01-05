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

class InternalEnergyEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(InternalEnergyEquation,self).__init__(ig) 
	
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
		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])		
        self.ddei      = np.asarray(eht.item().get('ddei')[intc])
        self.ddeiux      = np.asarray(eht.item().get('ddeiux')[intc])
		
        self.divu        = np.asarray(eht.item().get('divu')[intc])		
        self.ppdivu      = np.asarray(eht.item().get('ppdivu')[intc])

        self.ddenuc1      = np.asarray(eht.item().get('ddenuc1')[intc])		
        self.ddenuc2      = np.asarray(eht.item().get('ddenuc2')[intc])
		
        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddei    = np.asarray(eht.item().get('ddei')) 		

 	# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	
		
        dd = self.dd
        ux = self.ux
        pp = self.pp
        ddux = self.ddux
        ddei = self.ddei
        ddeiux = self.ddeiux		
        divu = self.divu
        ppdivu = self.ppdivu
 
        ddenuc1 = self.ddenuc1
        ddenuc2 = self.ddenuc2
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd
        fht_ei = ddei/dd
        f_ei = ddeiux - ddux*ddei/dd
		
        ##########################
        # INTERNAL ENERGY EQUATION 
        ##########################

        # LHS -dq/dt 		
        self.minus_dt_eht_dd_fht_ei = -self.dt(t_ddei,xzn0,t_timec,intc)	

        # LHS -div eht_dd fht_ux fht_ei		
        self.minus_div_eht_dd_fht_ux_fht_ei = -self.Div(dd*fht_ux*fht_ei,xzn0)
		
        # RHS -div fei
        self.minus_div_fei = -self.Div(f_ei,xzn0)
		
        # RHS -div ftt (not included) heat flux
        self.minus_div_ftt = -np.zeros(self.nx)
		
        # RHS -P d = - eht_pp Div eht_ux
        self.minus_eht_pp_div_eht_ux = -pp*self.Div(ux,xzn0)		
				
        # RHS -Wp = -eht_ppf_df
        self.minus_eht_ppf_df = -(ppdivu - pp*divu)
		
        # RHS source + dd enuc
        self.plus_eht_dd_fht_enuc = ddenuc1+ddenuc2		
		
        # RHS dissipated turbulent kinetic energy
        self.plus_disstke = +tke_diss  	

	# -res
        self.minus_resEiEquation = -(self.minus_dt_eht_dd_fht_ei + self.minus_div_eht_dd_fht_ux_fht_ei + \
         self.minus_div_fei + self.minus_div_ftt + self.minus_eht_pp_div_eht_ux + self.minus_eht_ppf_df + \
         self.plus_eht_dd_fht_enuc + self.plus_disstke)
		
        ##############################
        # END INTERNAL ENERGY EQUATION 
        ##############################			
		
    def plot_ei(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian internal energy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddei/self.dd
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'internal energy')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{\varepsilon}_I$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{\varepsilon}_I$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_ei.png')
	

    def plot_ei_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot internal energy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_fht_ei
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_ei
		
        rhs0 = self.minus_div_fei
        rhs1 = self.minus_div_ftt	
        rhs2 = self.minus_eht_pp_div_eht_ux
        rhs3 = self.minus_eht_ppf_df
        rhs4 = self.plus_eht_dd_fht_enuc
        rhs5 = self.plus_disstke
		
        res = self.minus_resEiEquation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('internal energy equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{\rho} \widetilde{\epsilon}_I )$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{\epsilon}_I$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_I $")     
        plt.plot(grd1,rhs1,color='c',label = r"$-\nabla_r f_T$ (not incl.)") 
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-\bar{P} \bar{d}$") 
        plt.plot(grd1,rhs3,color='r',label = r"$-W_P$")
        plt.plot(grd1,rhs4,color='b',label = r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}$")
        plt.plot(grd1,rhs5,color='m',label = r"$+\varepsilon_k$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_\epsilon$")
 
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
        plt.savefig('RESULTS/'+self.data_prefix+'ei_eq.png')		
		
		
		
