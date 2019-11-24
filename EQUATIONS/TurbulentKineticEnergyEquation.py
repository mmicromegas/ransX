import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al
import EQUATIONS.Properties as prop

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class TurbulentKineticEnergyEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,minus_kolmrate,bconv,tconv,data_prefix):
        super(TurbulentKineticEnergyEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	
        xznl   = np.asarray(eht.item().get('xznl'))
        xznr   = np.asarray(eht.item().get('xznr'))		
		
        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf			
		
        dd    = np.asarray(eht.item().get('dd')[intc])
        ux    = np.asarray(eht.item().get('ux')[intc])	
        uy    = np.asarray(eht.item().get('uy')[intc])
        uz    = np.asarray(eht.item().get('uz')[intc])		
        pp    = np.asarray(eht.item().get('pp')[intc])		
		
        ddux  = np.asarray(eht.item().get('ddux')[intc])
        dduy  = np.asarray(eht.item().get('dduy')[intc])
        dduz  = np.asarray(eht.item().get('dduz')[intc])		

        uxux = np.asarray(eht.item().get('uxux')[intc])
        uyuy = np.asarray(eht.item().get('uyuy')[intc])
        uzuz = np.asarray(eht.item().get('uzuz')[intc])
		
        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduxuy = np.asarray(eht.item().get('dduxuy')[intc])
        dduxuz = np.asarray(eht.item().get('dduxuz')[intc])
		
        ddekux = np.asarray(eht.item().get('ddekux')[intc])	
        ddek   = np.asarray(eht.item().get('ddek')[intc])		
		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])
        divu   = np.asarray(eht.item().get('divu')[intc])
        ppux   = np.asarray(eht.item().get('ppux')[intc])		

        ###################################
        # TURBULENT KINETIC ENERGY EQUATION 
        ###################################   		
				
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
        fht_ux = ddux/dd
        fht_ek = ddek/dd		
		
        uxffuxff = (dduxux/dd - ddux*ddux/(dd*dd)) 
        uyffuyff = (dduyuy/dd - dduy*dduy/(dd*dd)) 
        uzffuzff = (dduzuz/dd - dduz*dduz/(dd*dd)) 

        uxfuxf = (uxux - ux*ux) 
        uyfuyf = (uyuy - uy*uy) 
        uzfuzf = (uzuz - uz*uz)
		
        tke = 0.5*(uxffuxff + uyffuyff + uzffuzff)
        eht_tke = 0.5*(uxfuxf + uyfuyf + uzfuzf)
		
        fekx = ddekux - fht_ek*fht_ux
        fpx  = ppux - pp*ux 

        # LHS -dq/dt 			
        self.minus_dt_dd_tke = -self.dt(t_dd*t_tke,xzn0,t_timec,intc)

        # LHS -div dd ux tke
        self.minus_div_eht_dd_fht_ux_tke = -self.Div(dd*fht_ux*tke,xzn0)
		
        # -div kinetic energy flux
        self.minus_div_fekx  = -self.Div(fekx,xzn0)

        # -div acoustic flux		
        self.minus_div_fpx = -self.Div(fpx,xzn0)		
		
        # RHS warning ax = overline{+u''_x} 
        self.plus_ax = -ux + fht_ux		
		
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

        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0    = xzn0
        self.xznl    = xznl
        self.xznr    = xznr		
        self.dd      = dd
        self.tke     = tke
        self.eht_tke = eht_tke
        self.t_timec = t_timec
        self.t_tke 	 = t_tke
        self.t_dd 	 = t_dd
        self.bconv   = bconv
        self.tconv	 = tconv
        self.ig      = ig 		
		
		
    def plot_tke(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot turbulent kinetic energy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot 		
        plt1 = self.tke
        plt2 = self.eht_tke
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
				
        # plot DATA 
        plt.title('turbulent kinetic energy')
        plt.plot(grd1,plt1,color='brown',label = r"$\frac{1}{2} \widetilde{u''_i u''_i}$")
        plt.plot(grd1,plt2,color='r',linestyle='--',label = r"$\frac{1}{2} \overline{u'_i u'_i}$")
		

        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()
			
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
        if (self.ig == 1):			
            plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t (\overline{\rho} \widetilde{k})$')
            plt.plot(grd1,-lhs1,color='k',label = r"$-\nabla_x (\overline{\rho} \widetilde{u}_x \widetilde{k})$")
            plt.plot(grd1,rhs0,color='r',label = r'$+W_b$')     
            plt.plot(grd1,rhs1,color='c',label = r'$+W_p$') 
            plt.plot(grd1,rhs2,color='#802A2A',label = r"$-\nabla_x f_k$") 
            plt.plot(grd1,rhs3,color='m',label = r"$-\nabla_x f_P$")
            plt.plot(grd1,rhs4,color='b',label = r"$-\widetilde{R}_{xi}\partial_x \widetilde{u_i}$")		
            plt.plot(grd1,rhs5,color='k',linewidth=0.7,label = r"$-\overline{\rho} u^{'3}_{rms}/l_c$")		
            plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_k$")
        elif (self.ig == 2): 
            plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t (\overline{\rho} \widetilde{k})$')
            plt.plot(grd1,-lhs1,color='k',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{k})$")
            plt.plot(grd1,rhs0,color='r',label = r'$+W_b$')     
            plt.plot(grd1,rhs1,color='c',label = r'$+W_p$') 
            plt.plot(grd1,rhs2,color='#802A2A',label = r"$-\nabla_r f_k$") 
            plt.plot(grd1,rhs3,color='m',label = r"$-\nabla_r f_P$")
            plt.plot(grd1,rhs4,color='b',label = r"$-\widetilde{R}_{ri}\partial_r \widetilde{u_i}$")		
            plt.plot(grd1,rhs5,color='k',linewidth=0.7,label = r"$-\overline{\rho} u^{'3}_{rms}/l_c$")		
            plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_k$") 
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()
 
        # convective boundary markers
        plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')		
        plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')	 
  
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()
			
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
		  