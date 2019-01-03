import numpy as np
import matplotlib.pyplot as plt
import CALCULUS as calc
import ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

# https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf

class TurbulentKineticEnergyEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,minus_kolmrate,data_prefix):
        super(TurbulentKineticEnergyEquation,self).__init__(ig) 
	
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

        self.dduxux    = np.asarray(eht.item().get('dduxux')[intc])
        self.dduxuy    = np.asarray(eht.item().get('dduxuy')[intc])
        self.dduxuz    = np.asarray(eht.item().get('dduxuz')[intc])
		
        self.ddekux	   = np.asarray(eht.item().get('ddekux')[intc])	
        self.ddek      = np.asarray(eht.item().get('ddek')[intc])		
		
        self.ppdivu    = np.asarray(eht.item().get('ppdivu')[intc])
        self.divu      = np.asarray(eht.item().get('divu')[intc])
        self.ppux      = np.asarray(eht.item().get('ppux')[intc])		

        ###################################
        # TURBULENT KINETIC ENERGY EQUATION 
        ###################################   		
		
 		# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf	
		
        dd = self.dd
        ux = self.ux
        pp = self.pp
		
        ddux = self.ddux
        dduy = self.dduy
        dduz = self.dduz

        dduxux = self.dduxux
        dduyuy = self.dduyuy
        dduzuz = self.dduzuz

        dduxux = self.dduxux
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

        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 
        t_dd      = np.asarray(eht.item().get('dd'))
		
        t_ddux    = np.asarray(eht.item().get('ddux')) 
        t_dduy    = np.asarray(eht.item().get('dduy')) 
        t_dduz    = np.asarray(eht.item().get('dduz')) 		
		
        t_dduxux = np.asarray(eht.item().get('dduxux'))
        t_dduyuy = np.asarray(eht.item().get('dduyuy'))
        t_dduzuz = np.asarray(eht.item().get('dduzuz'))
		
        t_uxffuxff = t_dduxux/t_dd - t_ddux*t_ddux/(t_dd*t_dd)
        t_uyffuyff = t_dduyuy/t_dd - t_dduy*t_dduy/(t_dd*t_dd)
        t_uzffuzff = t_dduzuz/t_dd - t_dduz*t_dduz/(t_dd*t_dd)
		
        t_tke = 0.5*(t_uxffuxff+t_uyffuyff+t_uzffuzff)		
		
        # construct equation-specific mean fields		
        tke = 0.5*(uxffuxff + uyffuyff + uzffuzff)
        self.tke = tke
		
        # LHS -dq/dt 			
        self.minus_dt_dd_tke = -self.dt(t_dd*t_tke,xzn0,t_timec,intc)

        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = -self.Div(ddux*tke,xzn0)
		
        # -div kinetic energy flux
        self.minus_div_fekx  = -self.Div(dd*(ddekux/dd - (ddux/dd)*(ddek/dd)),xzn0)

        # -div acoustic flux		
        self.minus_div_fpx = -self.Div(ppux - pp*ux,xzn0)		
		
        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = -ux + ddux/dd		
		
        # +buoyancy work
        self.plus_wb = self.plus_ax*self.Grad(pp,xzn0)
		
        # +pressure dilatation
        self.plus_wp = ppdivu-pp*divu
				
        # -R grad u
		
        rxx = dduxux - ddux*ddux/dd
        rxy = dduxuy - ddux*dduy/dd
        rxz = dduxuz - ddux*dduz/dd
		
        self.minus_r_grad_u = -(rxx*self.Grad(ddux/dd,xzn0) + \
                                rxy*self.Grad(dduy/dd,xzn0) + \
                                rxz*self.Grad(dduz/dd,xzn0))
		

        # -res		
        self.minus_resTkeEquation = - (self.minus_dt_dd_tke + self.minus_div_eht_dd_fht_ux_tke + \
                                       self.plus_wb + self.plus_wp + self.minus_div_fekx + \
	                                   self.minus_div_fpx + self.minus_r_grad_u)

									   				
        # - kolm_rate u'3/lc
        self.minus_kolmrate = minus_kolmrate 		
													
        #######################################
        # END TURBULENT KINETIC ENERGY EQUATION 
        #######################################  		
			
    def plot_tke(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot turbulent kinetic energy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot 		
        plt1 = self.tke
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
				
        # plot DATA 
        plt.title('turbulent kinetic energy')
        plt.plot(grd1,plt1,color='brown',label = r'$\frac{1}{2} \widetilde{u''}_i \widetilde{u''}_i$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{k}$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_tke.png')		

    def plot_tke_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot turbulent kinetic energy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_dd_tke
        lhs1 = self.minus_div_eht_dd_fht_ux_tke
		
        rhs0 = self.plus_wb
        rhs1 = self.plus_wp		
        rhs2 = self.minus_div_fekx
        rhs3 = self.minus_div_fpx
        rhs4 = self.minus_r_grad_u
		
        res = self.minus_resTkeEquation
		
        rhs5 = self.minus_kolmrate*self.dd		
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # plot DATA 
        plt.title('turbulent kinetic energy equation')
        plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t (\widetilde{k})$')
        plt.plot(grd1,-lhs1,color='k',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{k})$")	
		
        plt.plot(grd1,rhs0,color='r',label = r'$+W_b$')     
        plt.plot(grd1,rhs1,color='c',label = r'$+W_p$') 
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-\nabla_r f_k$") 
        plt.plot(grd1,rhs3,color='m',label = r"$-\nabla_r f_P$")
        plt.plot(grd1,rhs4,color='b',label = r"$-\widetilde{R}_{ri}\partial_r \widetilde{u_i}$")		
        plt.plot(grd1,rhs5,color='k',linewidth=0.7,label = r"$-\overline{\rho} u^{'3}_{rms}/l_c$")		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_k$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'tke_eq.png')	
	
    def tke_dissipation(self):
        return self.minus_resTkeEquation		

    def tke(self):
        return self.tke		
		
