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

class TotalEnergyEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(TotalEnergyEquation,self).__init__(ig) 
	
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
        self.dduy      = np.asarray(eht.item().get('dduy')[intc])
        self.dduz      = np.asarray(eht.item().get('dduz')[intc])

        self.dduxux    = np.asarray(eht.item().get('dduxux')[intc])
        self.dduyuy    = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz    = np.asarray(eht.item().get('dduzuz')[intc])
        self.dduxuy    = np.asarray(eht.item().get('dduxuy')[intc])
        self.dduxuz    = np.asarray(eht.item().get('dduxuz')[intc])
		
        self.ddekux	   = np.asarray(eht.item().get('ddekux')[intc])	
        self.ddek      = np.asarray(eht.item().get('ddek')[intc])
		
        self.ddei      = np.asarray(eht.item().get('ddei')[intc])
        self.ddeiux      = np.asarray(eht.item().get('ddeiux')[intc])
		
        self.divu        = np.asarray(eht.item().get('divu')[intc])		
        self.ppdivu      = np.asarray(eht.item().get('ppdivu')[intc])
        self.ppux      = np.asarray(eht.item().get('ppux')[intc])			

        self.ddenuc1      = np.asarray(eht.item().get('ddenuc1')[intc])		
        self.ddenuc2      = np.asarray(eht.item().get('ddenuc2')[intc])
		
        xzn0 = self.xzn0

        #######################
        # TOTAL ENERGY EQUATION 
        #######################  		
		
 	    # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	
		
        dd = self.dd
        ux = self.ux
        pp = self.pp
		
        ddux = self.ddux
        dduy = self.dduy
        dduz = self.dduz

        dduxux = self.dduxux
        dduyuy = self.dduyuy
        dduzuz = self.dduzuz		
        dduxuy = self.dduxuy
        dduxuz = self.dduxuz
		
        ddek   = self.ddek
        ddekux = self.ddekux
        ppux   = self.ppux
        ppdivu = self.ppdivu
        divu   = self.divu
		
        uxffuxff = (dduxux/dd - ddux*ddux/(dd*dd)) 
        uyffuyff = (dduyuy/dd - dduy*dduy/(dd*dd)) 
        uzffuzff = (dduzuz/dd - dduz*dduz/(dd*dd)) 

        ddei = self.ddei
        ddeiux = self.ddeiux

        ddenuc1 = self.ddenuc1
        ddenuc2 = self.ddenuc2		

        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 
        t_dd      = np.asarray(eht.item().get('dd'))

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
		
        t_fht_ke = 0.5*(t_dduxux+t_dduyuy+t_dduzuz)/t_dd
        t_fht_ei = t_ddei/t_dd			
		
        # construct equation-specific mean fields		
        fht_ke = 0.5*(dduxux + dduyuy + dduzuz)/dd		
        fht_ux = ddux/dd
        fht_ei = ddei/dd
        f_ei = ddeiux - ddux*ddei/dd		
		
        self.fht_et = fht_ke + fht_ei
		
        # LHS -dq/dt 			
        self.minus_dt_eht_dd_fht_ke = -self.dt(t_dd*t_fht_ke,xzn0,t_timec,intc)
        self.minus_dt_eht_dd_fht_ei = -self.dt(t_dd*t_fht_ei,xzn0,t_timec,intc)
        self.minus_dt_eht_dd_fht_et = self.minus_dt_eht_dd_fht_ke + \
                                      self.minus_dt_eht_dd_fht_ei
		
        # LHS -div dd ux te
        self.minus_div_eht_dd_fht_ux_fht_ke = -self.Div(dd*fht_ux*fht_ke,xzn0)
        self.minus_div_eht_dd_fht_ux_fht_ei = -self.Div(dd*fht_ux*fht_ei,xzn0)
        self.minus_div_eht_dd_fht_ux_fht_et = self.minus_div_eht_dd_fht_ux_fht_ke + \
                                              self.minus_div_eht_dd_fht_ux_fht_ei		

        # RHS -div fei
        self.minus_div_fei = -self.Div(f_ei,xzn0)
		
        # RHS -div ftt (not included) heat flux
        self.minus_div_ftt = -np.zeros(self.nx)		
											  
        # -div kinetic energy flux
        self.minus_div_fekx  = -self.Div(dd*(ddekux/dd - (ddux/dd)*(ddek/dd)),xzn0)

        # -div acoustic flux		
        self.minus_div_fpx = -self.Div(ppux - pp*ux,xzn0)		
		
        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = -ux + ddux/dd		
		
        # +buoyancy work
        self.plus_wb = self.plus_ax*self.Grad(pp,xzn0)
		
        # RHS -P d = - eht_pp Div eht_ux
        self.minus_eht_pp_div_eht_ux = -pp*self.Div(ux,xzn0)
				
        # -R grad u
		
        rxx = dduxux - ddux*ddux/dd
        rxy = dduxuy - ddux*dduy/dd
        rxz = dduxuz - ddux*dduz/dd
		
        self.minus_r_grad_u = -(rxx*self.Grad(ddux/dd,xzn0) + \
                                rxy*self.Grad(dduy/dd,xzn0) + \
                                rxz*self.Grad(dduz/dd,xzn0))
		
        # -dd Dt ke
        t_fht_ux = t_ddux/t_dd		
        t_fht_uy = t_dduy/t_dd	
        t_fht_uz = t_dduz/t_dd	

        fht_ux = ddux/dd		
        fht_uy = dduy/dd	
        fht_uz = dduz/dd
		
        self.minus_dd_Dt_fht_ui_fht_ui = \
            -self.dt(t_dd*(t_fht_ux**2.+t_fht_uy**2.+t_fht_uz**2.),xzn0,t_timec,intc) - \
             self.Div(dd*fht_ux*(fht_ux**2.+fht_uy**2.+fht_uz**2.),xzn0)
		
        # RHS source + dd enuc
        self.plus_eht_dd_fht_enuc = ddenuc1+ddenuc2				
		
        # -res		
        self.minus_resTeEquation = - (self.minus_dt_eht_dd_fht_et + self.minus_div_eht_dd_fht_ux_fht_et + \
                                      self.minus_div_fei + self.minus_div_ftt + self.minus_div_fekx + \
                                      self.minus_div_fpx + self.minus_r_grad_u + self.minus_eht_pp_div_eht_ux + \
                                      self.plus_wb + self.plus_eht_dd_fht_enuc + self.minus_dd_Dt_fht_ui_fht_ui)

        ###########################
        # END TOTAL ENERGY EQUATION 
        ###########################  
		
		
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
	

    def plot_et_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot total energy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_eht_dd_fht_et
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_et
		
        rhs0 = self.minus_div_fei
        rhs1 = self.minus_div_ftt
        rhs2 = self.minus_div_fekx
        rhs3 = self.minus_div_fpx
        rhs4 = self.minus_r_grad_u		
        rhs5 = self.minus_eht_pp_div_eht_ux
        rhs6 = self.plus_wb
        rhs7 = self.plus_eht_dd_fht_enuc

        res = self.minus_resTeEquation
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('total energy equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{\rho} \widetilde{\epsilon}_t )$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{\epsilon}_t$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_I $")     
        plt.plot(grd1,rhs1,color='y',label = r"$-\nabla_r f_T$ (not incl.)") 
        plt.plot(grd1,rhs2,color='silver',label = r"$-\nabla_r f_k$")     
        plt.plot(grd1,rhs3,color='c',label = r"$-\nabla_r f_p$") 		
        plt.plot(grd1,rhs4,color='m',label = r"$-\widetilde{R}_{ri}\partial_r \widetilde{u_i}$")		
        plt.plot(grd1,rhs5,color='#802A2A',label = r"$-\bar{P} \bar{d}$") 		
        plt.plot(grd1,rhs6,color='r',label = r'$+W_b$')  
        plt.plot(grd1,rhs7,color='b',label = r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_{\epsilon_t}$")
 
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
        plt.savefig('RESULTS/'+self.data_prefix+'et_eq.png')		
		
		
		
