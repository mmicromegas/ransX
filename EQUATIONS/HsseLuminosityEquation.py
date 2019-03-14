import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class HsseLuminosityEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(HsseLuminosityEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0')) 	
        nx   = np.asarray(eht.item().get('nx')) 
		
        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])	
        pp = np.asarray(eht.item().get('pp')[intc])
        tt = np.asarray(eht.item().get('tt')[intc])
        cp = np.asarray(eht.item().get('cp')[intc])
		
        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])
        dduxuy = np.asarray(eht.item().get('dduxuy')[intc])
        dduxuz = np.asarray(eht.item().get('dduxuz')[intc])
		
        ddekux = np.asarray(eht.item().get('ddekux')[intc])	
        ddek   = np.asarray(eht.item().get('ddek')[intc])
		
        ddei = np.asarray(eht.item().get('ddei')[intc])
        ddeiux = np.asarray(eht.item().get('ddeiux')[intc])
		
        divu   = np.asarray(eht.item().get('divu')[intc])		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])
        ppux   = np.asarray(eht.item().get('ppux')[intc])			

        ddenuc1 = np.asarray(eht.item().get('ddenuc1')[intc])		
        ddenuc2 = np.asarray(eht.item().get('ddenuc2')[intc])
		
        chim = np.asarray(eht.item().get('chim')[intc]) 
        chit = np.asarray(eht.item().get('chit')[intc]) 
        chid = np.asarray(eht.item().get('chid')[intc])		
		
        ###########################
        # HSSE LUMINOSITY EQUATION 
        ##########################  		
						
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 
        t_dd      = np.asarray(eht.item().get('dd'))
        t_tt      = np.asarray(eht.item().get('tt'))
        t_pp      = np.asarray(eht.item().get('pp'))		
		
        t_ddei    = np.asarray(eht.item().get('ddei')) 		
		
        t_ddux    = np.asarray(eht.item().get('ddux')) 
        t_dduy    = np.asarray(eht.item().get('dduy')) 
        t_dduz    = np.asarray(eht.item().get('dduz')) 		
		
        t_dduxux = np.asarray(eht.item().get('dduxux'))
        t_dduyuy = np.asarray(eht.item().get('dduyuy'))
        t_dduzuz = np.asarray(eht.item().get('dduzuz'))

        t_uxux = np.asarray(eht.item().get('uxux'))
        t_uyuy = np.asarray(eht.item().get('uyuy'))
        t_uzuz = np.asarray(eht.item().get('uzuz'))
		
        t_fht_ek = 0.5*(t_dduxux+t_dduyuy+t_dduzuz)/t_dd
        t_fht_ei = t_ddei/t_dd			
		
        t_fht_ux = t_ddux/t_dd 
        t_fht_uy = t_dduy/t_dd
        t_fht_uz = t_dduz/t_dd
		
        t_fht_ui_fht_ui = t_fht_ux*t_fht_ux+t_fht_uy*t_fht_uy+t_fht_uz*t_fht_uz 		
		
        # construct equation-specific mean fields			
        #fht_ek = 0.5*(dduxux + dduyuy + dduzuz)/dd	
        fht_ek = ddek/dd		
        fht_ux = ddux/dd
        fht_uy = dduy/dd
        fht_uz = dduz/dd		
        fht_ei = ddei/dd
        fht_et = fht_ek + fht_ei
        fht_enuc = (ddenuc1+ddenuc2)/dd		
		
        fei   = ddeiux - ddux*ddei/dd
        fekx  = ddekux - fht_ux*fht_ek
        fpx   = ppux - pp*ux		
				
        fht_lum = 4.*np.pi*(xzn0**2)*dd*fht_ux*fht_et 				
				
        fht_ui_fht_ui = fht_ux*fht_ux+fht_uy*fht_uy+fht_uz*fht_uz 				
				
        alpha = 1./chid
        delta = -chit/chid
        phi   = chid/chim				
		
        # sphere surface		
        sps = +4.*np.pi*(xzn0**2.)
		
        # LHS -grad fht_lum 			
        self.minus_gradx_fht_lum = -self.Grad(fht_lum,xzn0)             
	  
        # RHS +4 pi r^2 dd fht_enuc
        self.plus_four_pi_rsq_dd_fht_enuc = sps*dd*fht_enuc		
	  
        # RHS -4 pi r^2 dd cp dt tt
        self.minus_four_pi_rsq_dd_cp_dt_tt = -sps*dd*cp*self.dt(t_tt,xzn0,t_timec,intc)		
	  
        # RHS +4 pi r^2 delta dt pp
        self.plus_four_pi_rsq_delta_dt_pp = +sps*delta*self.dt(t_pp,xzn0,t_timec,intc)		
	  
        # RHS +4 pi r^2 dd fht_ux gradx fht_et
        self.plus_four_pi_rsq_dd_fht_ux_gradx_fht_et = +sps*dd*fht_ux*self.Grad(fht_et,xzn0)

        # RHS +4 pi r^2 dd Dt fht_ui fht_ui/2 
        self.plus_four_pi_rsq_dd_Dt_fht_ui_fht_ui_o_two	= +sps*dd*(self.dt(t_fht_ui_fht_ui,xzn0,t_timec,intc) + fht_ux*self.Grad(fht_ui_fht_ui,xzn0))/2.
		
        # RHS -4 pi r^2 delta dt pp 
        self.minus_four_pi_rsq_delta_dt_pp = -sps*delta*self.dt(t_pp,xzn0,t_timec,intc)			  
	  
        # RHS -4 pi r^2 div fei
        self.minus_four_pi_rsq_div_fei = -sps*self.Div(fei,xzn0)
		
        # RHS -4 pi r^2 div ftt (not included) heat flux
        self.minus_four_pi_rsq_div_ftt = -sps*np.zeros(nx)		
											  		
        # RHS -4 pi r^2 P d = - 4 pi r^2 eht_pp Div eht_ux
        self.minus_four_pi_rsq_pp_div_ux = -sps*pp*self.Div(ux,xzn0)

        # RHS +4 pi r^2 dd tke_diss
        self.plus_four_pi_rsq_dd_tke_diss = -sps*tke_diss        
		
        # RHS +fht_et gradx dd fht_ux +4 pi r^2 		
        self.plus_fht_et_gradx_dd_fht_ux_four_pi_rsq = +fht_et*self.Grad(dd*fht_ux*sps,xzn0)		
		
		
        # -res		
        self.minus_resTeEquation = -(self.minus_gradx_fht_lum+self.plus_four_pi_rsq_dd_fht_enuc+\
         self.minus_four_pi_rsq_dd_cp_dt_tt+self.plus_four_pi_rsq_delta_dt_pp+\
         self.plus_four_pi_rsq_dd_fht_ux_gradx_fht_et+self.plus_four_pi_rsq_dd_Dt_fht_ui_fht_ui_o_two+\
         self.minus_four_pi_rsq_delta_dt_pp+self.minus_four_pi_rsq_div_fei+\
         self.minus_four_pi_rsq_div_ftt+self.minus_four_pi_rsq_pp_div_ux+self.minus_four_pi_rsq_pp_div_ux+\
         self.plus_four_pi_rsq_dd_tke_diss+self.plus_fht_et_gradx_dd_fht_ux_four_pi_rsq)

        #############################
        # END HSS LUMINOSITY EQUATION 
        #############################  

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.fht_et      = fht_ei + fht_ek			
		
    def plot_et(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean total energy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.fht_et
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'total energy')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{\varepsilon}_t$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{\varepsilon}_t$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_et.png')
	

    def plot_luminosity_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot luminosity equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_gradx_fht_lum
		
        rhs0 = self.plus_four_pi_rsq_dd_fht_enuc
        rhs1 = self.minus_four_pi_rsq_dd_cp_dt_tt
        rhs2 = self.plus_four_pi_rsq_delta_dt_pp
        rhs3 = self.plus_four_pi_rsq_dd_fht_ux_gradx_fht_et
        rhs4 = self.plus_four_pi_rsq_dd_Dt_fht_ui_fht_ui_o_two
        rhs5 = self.minus_four_pi_rsq_delta_dt_pp
        rhs6 = self.minus_four_pi_rsq_div_fei
        rhs7 = self.minus_four_pi_rsq_div_ftt 
        rhs8 = self.minus_four_pi_rsq_pp_div_ux
        rhs9 = self.plus_four_pi_rsq_dd_tke_diss
        rhs10 = self.plus_fht_et_gradx_dd_fht_ux_four_pi_rsq    
 	
        res = self.minus_resTeEquation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,rhs8,rhs9,rhs10,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('hss luminosity equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_r \widetilde{L}$")
 
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{\epsilon}_{nuc}$")     
        plt.plot(grd1,rhs1,color='y',label = r"$-4 \pi r^2 \rho c_p \partial_t \overline{T}$") 
        plt.plot(grd1,rhs2,color='silver',label = r"$+4 \pi r^2 \delta \partial_t \overline{P}$")     
        plt.plot(grd1,rhs3,color='c',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{u}_r \partial_r \widetilde{\epsilon}_t$") 		
        plt.plot(grd1,rhs4,color='m',label = r"$+4 \pi r^2 \overline{\rho} \widetilde{D}_t \widetilde{u}_i \widetilde{u}_i /2$")		
        plt.plot(grd1,rhs5,color='#802A2A',label = r"$-4 \pi r^2 \delta \partial_t \overline{P}$") 		
        plt.plot(grd1,rhs6,color='r',label = r"$-4 \pi r^2 \nabla_r f_I$") 
        plt.plot(grd1,rhs7,color='gray',label = r"$-4 \pi r^2 \nabla_r f_{th}$ (not incl.)") 		
        plt.plot(grd1,rhs8,color='b',label = r"$-4 \pi r^2 \overline{P} \ \overline{d}$")
        plt.plot(grd1,rhs9,color='green',label = r"$+4 \pi r^2 \overline{\rho} \varepsilon_k$")		
        plt.plot(grd1,rhs10,color='pink',label = r"$+4 \pi r^2 \widetilde{\epsilon}_t \partial_r \overline{\rho} \widetilde{u}_r 4 \pi r^2 $")	
	
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r""
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'hsse_luminosity_eq.png')		
		
		
		
