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

class InternalEnergyFluxEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,tke_diss,data_prefix):
        super(InternalEnergyFluxEquation,self).__init__(ig) 
	
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
        self.ei        = np.asarray(eht.item().get('ei')[intc])	
        self.tt        = np.asarray(eht.item().get('tt')[intc])
		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])		
        self.dduy      = np.asarray(eht.item().get('dduy')[intc])
        self.dduz      = np.asarray(eht.item().get('dduz')[intc])		
        self.ddei      = np.asarray(eht.item().get('ddei')[intc])
		
        self.dduxux      = np.asarray(eht.item().get('dduxux')[intc])		
        self.dduyuy      = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz      = np.asarray(eht.item().get('dduzuz')[intc])	
		
        self.ddeiux      = np.asarray(eht.item().get('ddeiux')[intc])
        self.ddeiuy      = np.asarray(eht.item().get('ddeiuy')[intc])
        self.ddeiuz      = np.asarray(eht.item().get('ddeiuz')[intc])		

        self.ddeiuxux      = np.asarray(eht.item().get('ddeiuxux')[intc])
        self.ddeiuyuy      = np.asarray(eht.item().get('ddeiuyuy')[intc])
        self.ddeiuzuz      = np.asarray(eht.item().get('ddeiuzuz')[intc])
		
        self.divu        = np.asarray(eht.item().get('divu')[intc])		
        self.ppdivu      = np.asarray(eht.item().get('ppdivu')[intc])

        self.ddenuc1      = np.asarray(eht.item().get('ddenuc1')[intc])		
        self.ddenuc2      = np.asarray(eht.item().get('ddenuc2')[intc])

        self.dduxenuc1      = np.asarray(eht.item().get('dduxenuc1')[intc])		
        self.dduxenuc2      = np.asarray(eht.item().get('dduxenuc2')[intc])
		
        self.eigradxpp      = np.asarray(eht.item().get('eigradxpp')[intc])				
		
        self.ppdivu      = np.asarray(eht.item().get('ppdivu')[intc])		
        self.uxppdivu    = np.asarray(eht.item().get('uxppdivu')[intc])		
		
        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd = np.asarray(eht.item().get('dd'))
        t_ddux = np.asarray(eht.item().get('ddux')) 
        t_ddei = np.asarray(eht.item().get('ddei'))
        t_ddeiux = np.asarray(eht.item().get('ddeiux')) 		

 	# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	

        ux = self.ux
        dd = self.dd
        pp = self.pp
        ei = self.ei
        tt = self.tt
		
        ddux = self.ddux
        dduy = self.dduy
        dduz = self.dduz		
        ddei = self.ddei

        dduxux = self.dduxux
        dduyuy = self.dduyuy
        dduzuz = self.dduzuz		
        ddeiux = self.ddeiux
		
        ddeiux   = self.ddeiux
        ddeiuy   = self.ddeiuy
        ddeiuz   = self.ddeiuz		
		
        ddeiuxux   = self.ddeiuxux
        ddeiuyuy   = self.ddeiuyuy
        ddeiuzuz   = self.ddeiuzuz
        ddenuc1    = self.ddenuc1
        ddenuc2    = self.ddenuc2
        dduxenuc1  = self.dduxenuc1
        dduxenuc2  = self.dduxenuc2
        eigradxpp  = self.eigradxpp

        ddei     = self.ddei		
        ppdivu   = self.ppdivu
        uxppdivu = self.uxppdivu
		
        # construct equation-specific mean fields		
        fht_ux   = ddux/dd
        fht_ei   = ddei/dd
        rxx  = dduxux - ddux*ddux/dd 		

        f_ei       = ddeiux - ddux*ddei/dd
        fr_ei      = ddeiuxux - ddei*dduxux/dd - 2.*fht_ux*ddeiux + 2.*dd*fht_ux*fht_ei*fht_ux

        eiff = ei - ddei/dd				
        eiff_gradx_ppf = eigradxpp - ei*self.Grad(pp,xzn0)
		
        uxff_dd_enuc = (dduxenuc1 + dduxenuc2) - fht_ux*(ddenuc1 + ddenuc2)
		
        uxff_epsilonk_approx = (ux - ddux/dd)*tke_diss
		
        Grei = -(ddeiuyuy-ddei*dduyuy/dd-2.*(dduy/dd)*(ddeiuy/dd)+2.*ddei*dduy*dduy/(dd*dd*dd))/xzn0- \
                (ddeiuzuz-ddei*dduzuz/dd-2.*(dduz/dd)*(ddeiuz/dd)+2.*ddei*dduz*dduz/(dd*dd*dd))/xzn0
		
        eiff_GrM = -(ddeiuyuy - (ddei/dd)*dduyuy)/xzn0 - (ddeiuzuz - (ddei/dd)*dduzuz)/xzn0		
		
        ###############################		
	# INTERNAL ENERGY FLUX EQUATION
        ###############################
					   
        # time-series of internal energy flux 
        t_f_ei = t_ddeiux/t_dd - t_ddei*t_ddux/(t_dd*t_dd)
		
        # LHS -dq/dt 		
        self.minus_dt_f_ei = -self.dt(t_f_ei,xzn0,t_timec,intc)
     
        # LHS -div fht_ux f_ei
        self.minus_div_fht_ux_f_ei = -self.Div(fht_ux*f_ei,xzn0)	 
		
        # RHS -div flux internal energy flux
        self.minus_div_fr_ei = -self.Div(fr_ei,xzn0)
        
	# RHS -f_ei_gradx_fht_ux
        self.minus_f_ei_gradx_fht_ux = -f_ei*self.Grad(fht_ux,xzn0)
		
	# RHS -rxx_gradx_fht_ei
        self.minus_rxx_gradx_fht_ei = -rxx*self.Grad(fht_ei,xzn0)	

	# RHS -eht_eiff_gradx_eht_pp
        self.minus_eht_eiff_gradx_eht_pp = -(ei - ddei/dd)*self.Grad(pp,xzn0)
		
        # RHS -eht_eiff_gradx_ppf
        self.minus_eht_eiff_gradx_ppf = -(eigradxpp - (ddei/dd)*self.Grad(pp,xzn0))
		
	# RHS -eht_uxff_pp_divu
        self.minus_eht_uxff_pp_divu = -(uxppdivu - (ddux/dd)*ppdivu)

        # RHS eht_uxff_dd_nuc	
        self.plus_eht_uxff_dd_nuc =  (dduxenuc1 + dduxenuc2) - fht_ux*(ddenuc1+ddenuc2) 		
	# RHS eht_uxff_div_ftt (not calculated)
        eht_uxff_div_f_tt = np.zeros(self.nx)  		
        self.plus_eht_uxff_div_ftt = eht_uxff_div_f_tt
		
        # RHS eht_uxff_epsilonk_approx	
        self.plus_eht_uxff_epsilonk_approx =  (ux - fht_ux)*tke_diss 		

        # RHS Gei
        self.plus_Gei = -Grei-eiff_GrM	

        # -res  
        self.minus_resEiFluxEquation = -(self.minus_dt_f_ei + self.minus_div_fht_ux_f_ei + \
          self.minus_div_fr_ei + self.minus_f_ei_gradx_fht_ux + self.minus_rxx_gradx_fht_ei + \
          self.minus_eht_eiff_gradx_eht_pp + self.minus_eht_eiff_gradx_ppf + self.minus_eht_uxff_pp_divu + \
          self.plus_eht_uxff_dd_nuc + self.plus_eht_uxff_div_ftt + self.plus_eht_uxff_epsilonk_approx + \
          self.plus_Gei)
                                       
        ###################################		
	# END INTERNAL ENERGY FLUX EQUATION
        ###################################
									   
    def plot_fei(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot mean Favrian internal energy flux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddeiux - self.ddux*self.ddei/self.dd
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title(r'internal energy flux')
        plt.plot(grd1,plt1,color='brown',label = r'f$_I$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$f_I$ (erg cm$^{-2}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_fei.png')

									   
    def plot_fei_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot internal energy flux equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_f_ei
        lhs1 = self.minus_div_fht_ux_f_ei
		
        rhs0 = self.minus_div_fr_ei
        rhs1 = self.minus_f_ei_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_ei 
        rhs3 = self.minus_eht_uxff_pp_divu
        rhs4 = self.minus_eht_eiff_gradx_eht_pp 
        rhs5 = self.minus_eht_eiff_gradx_ppf
        rhs6 = self.plus_eht_uxff_dd_nuc
        rhs7 = self.plus_eht_uxff_div_ftt
        rhs8 = self.plus_eht_uxff_epsilonk_approx
        rhs9 = self.plus_Gei
	  
        res = self.minus_resEiFluxEquation
	
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,rhs6,rhs7,rhs8,rhs9,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
		
        # plot DATA 
        plt.title('internal energy flux equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t f_I$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\widetilde{u}_r f_I$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_i^r $")     
        plt.plot(grd1,rhs1,color='#802A2A',label = r"$-f_i \partial_r \widetilde{u}_r$") 
        plt.plot(grd1,rhs2,color='r',label = r"$-\widetilde{R}_{rr} \partial_r \widetilde{\epsilon_I}$") 
        plt.plot(grd1,rhs3,color='firebrick',label = r"$-\overline{u''_r P d}$") 
        plt.plot(grd1,rhs4,color='c',label = r"$-\overline{\epsilon''_I}\partial_r \overline{P}$")
        plt.plot(grd1,rhs5,color='mediumseagreen',label = r"$-\overline{\epsilon''_I \partial_r P'}$")
        plt.plot(grd1,rhs6,color='b',label = r"$+\overline{u''_r \rho \varepsilon_{nuc}}$")
        plt.plot(grd1,rhs7,color='m',label = r"$+\overline{u''_r \nabla \cdot T}$")
        plt.plot(grd1,rhs8,color='g',label = r"$+\overline{u''_r \varepsilon_k }$")
        plt.plot(grd1,rhs9,color='y',label = r"$+G_{\epsilon}$")

		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_\epsilon$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-2}$ s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'fei_eq.png')	
