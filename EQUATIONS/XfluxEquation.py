import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XfluxEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,bconv,tconv,intc,data_prefix):
        super(XfluxEquation,self).__init__(ig) 
					
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	

        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])	
        pp = np.asarray(eht.item().get('pp')[intc])
        xi = np.asarray(eht.item().get('x'+inuc)[intc])	
		
        ddux = np.asarray(eht.item().get('ddux')[intc])
        dduy = np.asarray(eht.item().get('dduy')[intc])
        dduz = np.asarray(eht.item().get('dduz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduyuy = np.asarray(eht.item().get('dduyuy')[intc])
        dduzuz = np.asarray(eht.item().get('dduzuz')[intc])
	
        ddxi    = np.asarray(eht.item().get('ddx'+inuc)[intc])
        ddxiux  = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        ddxidot = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])	
	 	
        ddxidotux = np.asarray(eht.item().get('ddx'+inuc+'dotux')[intc]) 	
        ddxiuxux  = np.asarray(eht.item().get('ddx'+inuc+'uxux')[intc])		
        ddxiuyuy  = np.asarray(eht.item().get('ddx'+inuc+'uyuy')[intc])
        ddxiuzuz  = np.asarray(eht.item().get('ddx'+inuc+'uzuz')[intc])		

        xigradxpp = np.asarray(eht.item().get('x'+inuc+'gradxpp')[intc]) 

        # Reynolds-averaged mean fields for flux modelling:
        ddcp = np.asarray(eht.item().get('ddcp')[intc])
        ddtt = np.asarray(eht.item().get('ddtt')[intc])
        ddhh = np.asarray(eht.item().get('ddhh')[intc])
        ddhhux = np.asarray(eht.item().get('ddhhux')[intc])
        ddttsq = np.asarray(eht.item().get('ddttsq')[intc])

        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])
		
        divu = np.asarray(eht.item().get('divu')[intc])
        gamma1 = np.asarray(eht.item().get('gamma1')[intc])
        gamma3 = np.asarray(eht.item().get('gamma3')[intc])
		
        gamma1 = np.asarray(eht.item().get('ux')[intc])
        gamma3 = np.asarray(eht.item().get('ux')[intc])		
		
        fht_rxx = dduxux - ddux*ddux/dd
        fdil = (uxdivu - ux*divu) 
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddux    = np.asarray(eht.item().get('ddux')) 
        t_ddxi    = np.asarray(eht.item().get('ddx'+inuc))		
        t_ddxiux  = np.asarray(eht.item().get('ddx'+inuc+'ux'))
				
        ##################
        # Xi FLUX EQUATION 
        ##################		
   
        # construct equation-specific mean fields
        t_fxi  = t_ddxiux - t_ddxi*t_ddux/t_dd
			
        fht_ux = ddux/dd
        fht_xi = ddxi/dd
		
        rxx    = dduxux - ddux*ddux/dd		
        fxi    = ddxiux - ddxi*ddux/dd
        frxi   = ddxiuxux - (ddxi/dd)*dduxux - 2.*(ddux/dd)*ddxiux + 2.*ddxi*ddux*ddux/(dd*dd) 		
			  
        # LHS -dq/dt 
        self.minus_dt_fxi = -self.dt(t_fxi,xzn0,t_timec,intc)
		
        # LHS -div(dduxfxi)
        self.minus_div_fht_ux_fxi = -self.Div(fht_ux*fxi,xzn0)		

        # RHS -div frxi  
        self.minus_div_frxi = -self.Div(frxi,xzn0)

        # RHS -fxi d_r fu_r
        self.minus_fxi_gradx_fht_ux = -fxi*self.Grad(fht_ux,xzn0)
		
        # RHS -rxx d_r xi
        self.minus_rxx_gradx_fht_xi = -rxx*self.Grad(fht_xi,xzn0)

        # RHS - X''i gradx P - X''_i gradx P'
        self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff = \
          -(xi*self.Grad(pp,xzn0) - fht_xi*self.Grad(pp,xzn0)) - (xigradxpp - xi*self.Grad(pp,xzn0)) 		

        # RHS +uxff_eht_dd_xidot
        self.plus_uxff_eht_dd_xidot = +(ddxidotux - (ddux/dd)*ddxidot)  		
		
        # RHS +gi 
        self.plus_gi = \
                  -(ddxiuyuy - (ddxi/dd)*dduyuy - 2.*(dduy/dd) + 2.*ddxi*dduy*dduy/(dd*dd))/xzn0 - \
                   (ddxiuzuz - (ddxi/dd)*dduzuz - 2.*(dduz/dd) + 2.*ddxi*dduz*dduz/(dd*dd))/xzn0 + \
                   (ddxiuyuy - (ddxi/dd)*dduyuy)/xzn0 + \
                   (ddxiuzuz - (ddxi/dd)*dduzuz)/xzn0

        # -res				   
        self.minus_resXiFlux = -(self.minus_dt_fxi + self.minus_div_fht_ux_fxi + self.minus_div_frxi + \
                         self.minus_fxi_gradx_fht_ux + self.minus_rxx_gradx_fht_xi + \
                         self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff + \
                         self.plus_uxff_eht_dd_xidot + self.plus_gi)     
		
        ######################
        # END Xi FLUX EQUATION 
        ######################	

        # variance of temperature fluctuations		
        sigmatt = (ddttsq-ddtt*ddtt/dd)/dd        		
				
        # enthalpy flux 
        fhh = ddhhux - ddhh*ddux/dd				
				
        # heat capacity		
        fht_cp = ddcp/dd 		
		
        # mlt velocity		
        alphae = 1.		
        u_mlt = fhh/(alphae*fht_cp*sigmatt)
		
        self.minus_ddxiumlt = -dd*xi*u_mlt	
		
        # models
        self.model_1 = self.minus_rxx_gradx_fht_xi	
        self.model_2 = self.minus_ddxiumlt		

        # grad models		
        self.plus_gradx_fxi = +self.Grad(fxi,xzn0)
        cnst = gamma1
        self.minus_cnst_dd_fxi_fdil_o_fht_rxx = -cnst*dd*fxi*fdil/fht_rxx		
        
		
        hp = 2.5e8		
		
        # Dgauss gradient model	
        dir = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\INIMODEL\\'		
        data = np.loadtxt(dir+'imodel.tycho',skiprows=26)		
        nxmax = 500
		
        rr = data[1:nxmax,2]
        vmlt = data[1:nxmax,8]		
        u_mlt = np.interp(xzn0,rr,vmlt)		
		
        #Dumlt1     = (1./3.)*u_mlt*lc		
		
        alpha = 1.5
        Dumlt2 = (1./3.)*u_mlt*alpha*hp        

        alpha = 1.6
        Dumlt3 = (1./3.)*u_mlt*alpha*hp  		
		
        def gauss(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*(sigma**2)))		
		
        Dmlt = Dumlt3		
		
        ampl = max(Dmlt)
        xx0 = (bconv+0.46e8+tconv)/2.
        width = 4.e7

        Dgauss = gauss(xzn0,ampl,xx0,width)		
		
        self.model_3 = -Dgauss*dd*self.Grad(fht_xi,xzn0)		
        self.model_4 = -Dumlt3*dd*self.Grad(fht_xi,xzn0)
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0    = xzn0		
        self.inuc    = inuc
        self.element = element
        self.fxi     = fxi		
        self.bconv   = bconv
        self.tconv	 = tconv 	
		
		
    def plot_Xflux(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xflux stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0		
		
        # load and calculate DATA to plot
        plt1 = self.fxi
        plt2 = self.model_1
        plt3 = self.model_2
        plt4 = self.model_3
        plt5 = self.model_4		
		
        # create FIGURE
        plt.figure(figsize=(7,6))	

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))			
		
        # set plot boundaries   
        to_plot = [plt1,plt2,plt3,plt4,plt5]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
					
        # plot DATA 
        plt.title('Xflux for '+self.element)
        plt.plot(grd1,plt1,color='k',label = r'f')
        #plt.plot(grd1,plt2,color='c',label=r"$-\widetilde{R}_{rr} \partial_r \widetilde{X}$")
        #plt.plot(grd1,plt3,color='g',label=r"$-\overline{\rho} \ \overline{X} \ u_{mlt}$")
        plt.plot(grd1,plt4,color='r',label=r"$-D_{gauss} \ \partial_r \widetilde{X}$")
        plt.plot(grd1,plt5,color='b',label=r"$-D_{mlt} \ \partial_r \widetilde{X}$",linewidth=0.7)
		

        # convective boundary markers
        plt.axvline(self.bconv+0.46e8,linestyle='--',linewidth=0.7,color='k')		
        plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')		
		
        # convective boundary markers		
        #plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')		
        #plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')  				
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()
			
        setylabel = r"$\overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_Xflux_'+element+'.png')
		
    def plot_Xflux_gradient(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xflux stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0		
		
        # load and calculate DATA to plot
        plt1 = self.plus_gradx_fxi
        plt2 = self.minus_cnst_dd_fxi_fdil_o_fht_rxx

        # create FIGURE
        plt.figure(figsize=(7,6))	

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))			
		
        # set plot boundaries   
        to_plot = [plt1,plt2]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
					
        # plot DATA 
        plt.title('grad Xflux for '+self.element)
        plt.plot(grd1,plt1,color='k',label = r"$+\partial_r f$")
        plt.plot(grd1,plt2,color='r',label=r"$.$")


        # convective boundary markers
        plt.axvline(self.bconv+0.46e8,linestyle='--',linewidth=0.7,color='k')		
        plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')		
		
        # convective boundary markers		
        #plt.axvline(self.bconv,linestyle='--',linewidth=0.7,color='k')		
        #plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')  		
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()
			
        setylabel = r"$\partial_r \overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-3}$ s$^{-1}$)"
		
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_model_Xflux_'+element+'.png')		
		
    def plot_Xflux_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xi flux equation in the model""" 
 
        # convert nuc ID to string
        xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
		
        lhs0 = self.minus_dt_fxi
        lhs1 = self.minus_div_fht_ux_fxi
		
        rhs0 = self.minus_div_frxi
        rhs1 = self.minus_fxi_gradx_fht_ux
        rhs2 = self.minus_rxx_gradx_fht_xi
        rhs3 = self.minus_xiff_gradx_pp_minus_xiff_gradx_ppff
        rhs4 = self.plus_uxff_eht_dd_xidot 
        rhs5 = self.plus_gi
		
        res =  self.minus_resXiFlux
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,rhs3,rhs4,rhs5,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
				
        # plot DATA 
        plt.title('Xflux equation for '+self.element)
        if (self.ig == 1):
            plt.plot(grd1,lhs0,color='#8B3626',label = r'$-\partial_t f_i$')
            plt.plot(grd1,lhs1,color='#FF7256',label = r'$-\nabla_x (\widetilde{u}_x f)$')		
            plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_x f^r_i$')
            plt.plot(grd1,rhs1,color='g',label=r'$-f_i \partial_x \widetilde{u}_x$')
            plt.plot(grd1,rhs2,color='r',label=r'$-R_{rr} \partial_x \widetilde{X}$')	
            plt.plot(grd1,rhs3,color='cyan',label=r"$-\overline{X''} \partial_x \overline{P} - \overline{X'' \partial_x P'}$")
            plt.plot(grd1,rhs4,color='purple',label=r"$+\overline{u''_x \rho \dot{X}}$")
            #plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')		
            plt.plot(grd1,res,color='k',linestyle='--',label='res')
        elif (self.ig == 2):
            plt.plot(grd1,lhs0,color='#8B3626',label = r'$-\partial_t f_i$')
            plt.plot(grd1,lhs1,color='#FF7256',label = r'$-\nabla_r (\widetilde{u}_r f)$')		
            plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f^r_i$')
            plt.plot(grd1,rhs1,color='g',label=r'$-f_i \partial_r \widetilde{u}_r$')
            plt.plot(grd1,rhs2,color='r',label=r'$-R_{rr} \partial_r \widetilde{X}$')	
            plt.plot(grd1,rhs3,color='cyan',label=r"$-\overline{X''} \partial_r \overline{P} - \overline{X'' \partial_r P'}$")
            plt.plot(grd1,rhs4,color='purple',label=r"$+\overline{u''_r \rho \dot{X}}$")
            plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')		
            plt.plot(grd1,res,color='k',linestyle='--',label='res')		
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
			
        setylabel = r"g cm$^{-2}$ s$^{-2}$"
		
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':10})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_XfluxEquation_'+element+'.png')			