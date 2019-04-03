import numpy as np
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class ReynoldsStressXXequation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,minus_kolmrate,data_prefix):
        super(ReynoldsStressXXequation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf			
		
        dd     = np.asarray(eht.item().get('dd')[intc])
        ux     = np.asarray(eht.item().get('ux')[intc])	
        pp     = np.asarray(eht.item().get('pp')[intc])		
		
        ddux   = np.asarray(eht.item().get('ddux')[intc])
        dduy   = np.asarray(eht.item().get('dduy')[intc])
        dduz   = np.asarray(eht.item().get('dduz')[intc])		
		
        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduxuy = np.asarray(eht.item().get('dduxuy')[intc])
        dduxuz = np.asarray(eht.item().get('dduxuz')[intc])

        dduxuxux = np.asarray(eht.item().get('dduxuxux')[intc])
        dduxuyuy = np.asarray(eht.item().get('dduxuyuy')[intc])
        dduxuzuz = np.asarray(eht.item().get('dduxuzuz')[intc])
		
        ddekux = np.asarray(eht.item().get('ddekux')[intc])	
        ddek   = np.asarray(eht.item().get('ddek')[intc])		
		
        ppdivux = np.asarray(eht.item().get('ppdivux')[intc])
        divux   = np.asarray(eht.item().get('divux')[intc])
        ppux    = np.asarray(eht.item().get('ppux')[intc])		

        #############################
        # REYNOLDS STRESS XX EQUATION 
        #############################   				
		
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
		
        t_rxx = t_dd*t_uxffuxff		
		
        # construct equation-specific mean fields
        fht_ux = ddux/dd
        fht_uy = dduy/dd
        fht_uz = dduz/dd

        uxffuxff = (dduxux/dd - ddux*ddux/(dd*dd)) 
			
        rxx = dd*uxffuxff
		
        fkr = dduxuxux - 3.*fht_ux*dduxux + 2.*fht_ux*fht_ux*fht_ux*dd
        fpx = ppux - pp*ux
		
        # LHS -dq/dt 			
        self.minus_dt_rxx = -self.dt(t_rxx,xzn0,t_timec,intc)

        # LHS -div ux rxx
        self.minus_div_fht_ux_rxx = -self.Div(fht_ux*rxx,xzn0)
		
        # -div 2 fkr 
        self.minus_div_two_fkr  = -self.Div(2.*fkr,xzn0)

        # -div 2 acoustic flux		
        self.minus_div_two_fpx = -self.Div(2.*fpx,xzn0)		
		
        # warning ax = overline{+u''_x} 
        self.plus_ax = -ux + fht_ux		
		
        # +2 buoyancy work
        self.plus_two_wb = 2.*self.plus_ax*self.Grad(pp,xzn0)
		
        # +2 pressure rr dilatation
        self.plus_two_ppf_divuxff = 2.*(ppdivux - pp*divux)
				
        # -2 R grad u	
        self.minus_two_rxx_grad_fht_ux = -2.*rxx*self.Grad(fht_ux,xzn0)
		
		# +GrrR
        GrrR = - 2.*(dduxuyuy - 2.*dduy*dduxuy/dd - fht_ux*dduyuy + 2.*fht_uy*fht_uy*fht_ux*dd)/xzn0 - \
               	 2.*(dduxuzuz - 2.*dduz*dduxuz/dd - fht_ux*dduzuz + 2.*fht_uz*fht_uz*fht_ux*dd)/xzn0 	
        uxff_GrM = (-dduxuyuy-dduxuzuz)/xzn0 - fht_ux*(-dduyuy-dduzuz)/xzn0

        # +2 Gkr		
        self.plus_two_Gkr = 2.*((1./2.)*GrrR - uxff_GrM) 	   
		
        # -res		
        self.minus_resRxxEquation = -(self.minus_dt_rxx + self.minus_div_fht_ux_rxx + self.minus_div_two_fkr +\
                                      self.minus_div_two_fpx + self.plus_two_wb + self.plus_two_ppf_divuxff +\
                                      self.minus_two_rxx_grad_fht_ux + self.plus_two_Gkr)

									   				
        # - kolm_rate 1/3 u'3/lc
        self.minus_onethrd_kolmrate = (1./3.)*minus_kolmrate 		
													
        #################################
        # END REYNOLDS STRESS XX EQUATION 
        #################################  		
			
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0        = xzn0
        self.dd         = dd			
        self.rxx         = rxx				
			
    def plot_rxx(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Reynolds stress xx in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot 		
        plt1 = self.rxx
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
				
        # plot DATA 
        plt.title('rxx')
        plt.plot(grd1,plt1,color='brown',label = r"$\overline{\rho} \widetilde{u''_r u''_r}$")

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$R_{xx}$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_rxx.png')		

    def plot_rxx_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Reynolds stress rxx equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_rxx
        lhs1 = self.minus_div_fht_ux_rxx 
		
        rhs0 = self.plus_two_wb
        rhs1 = self.plus_two_ppf_divuxff		
        rhs2 = self.minus_div_two_fkr
        rhs3 = self.minus_div_two_fpx
        rhs4 = self.minus_two_rxx_grad_fht_ux
        rhs5 = self.plus_two_Gkr
		
        res = self.minus_resRxxEquation
		
        #rhs6 = self.minus_onethrd_kolmrate*self.dd		
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # plot DATA 
        plt.title('reynolds stress xx equation')
        plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t R_{rr}$')
        plt.plot(grd1,-lhs1,color='k',label = r"$-\nabla_r (\widetilde{u}_r R_{rr})$")	
		
        plt.plot(grd1,rhs0,color='r',label = r"$+2 W_b$")     
        plt.plot(grd1,rhs1,color='c',label = r"$+2 \overline{P' \nabla u''_r }$") 
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-\nabla_r 2 f_k^r$") 
        plt.plot(grd1,rhs3,color='m',label = r"$-\nabla_r 2 f_P$")
        plt.plot(grd1,rhs4,color='b',label = r"$-\widetilde{R}_{rr}\partial_r \widetilde{u_r}$")		
        plt.plot(grd1,rhs5,color='y',label=r"$2 \mathcal{G}_k^r$")
        #plt.plot(grd1,rhs6,color='k',linewidth=0.7,label = r"$-\overline{\rho} 1/3 u^{'3}_{rms}/l_c$")		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_{Rrr}$")
 
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
        plt.savefig('RESULTS/'+self.data_prefix+'rxx_eq.png')	
	
    def tke_dissipation(self):
        return self.minus_resTkeEquation		

    def tke(self):
        return self.tke		
		
