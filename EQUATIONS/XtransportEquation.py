import numpy as np
import matplotlib.pyplot as plt
import CALCULUS as calc
import ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XtransportEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,intc,data_prefix):
        super(XtransportEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.data_prefix = data_prefix
        self.inuc  = inuc
        self.element = element
		
        # assign global data to be shared across whole class	
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('xzn0')) 

        self.dd        = np.asarray(eht.item().get('dd')[intc])
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])	
        self.ddxi      = np.asarray(eht.item().get('ddx'+inuc)[intc])
        self.ddxiux    = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        self.ddxidot   = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])	
		
        #######################
        # Xi TRANSPORT EQUATION 
        #######################
		
 		# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf		
		
        dd      = self.dd
        ddux    = self.ddux
        ddxi    = self.ddxi
        ddxiux  = self.ddxiux
        ddxidot = self.ddxidot
        xzn0    = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 	
        t_ddxi    = np.asarray(eht.item().get('ddx'+inuc))			
		
        # construct equation-specific mean fields
        fht_ux = ddux/dd
        fht_xi = ddxi/dd
        fxi    = ddxiux - ddxi*ddux/dd
		
        # LHS -dq/dt 		
        self.minus_dt_eht_dd_fht_xi = -self.dt(t_ddxi,xzn0,t_timec,intc)
        # LHS -div(ddXiux)
        self.minus_div_eht_dd_fht_ux_fht_xi = -self.Div(dd*fht_ux*fht_xi,xzn0)
		
        # RHS -div fxi 
        self.minus_div_fxi = -self.Div(fxi,self.xzn0) 
        # RHS +ddXidot 
        self.plus_ddxidot = +ddxidot 
        # -res
        self.minus_resXiTransport = -(self.minus_dt_eht_dd_fht_xi + self.minus_div_eht_dd_fht_ux_fht_xi + \
                               self.minus_div_fxi + self.plus_ddxidot)
		
        ###########################		
        # END Xi TRANSPORT EQUATION
        ###########################
		
        #print('#----------------------------------------------------#')
        #print('Loading RA-ILES COMPOSITION TRANSPORT EQUATION terms')	
        #print('Central time (in s): ',round(self.timec,1))
        #print('Averaging windows (in s): ',self.tavg.item(0))
        #print('Time range (in s from-to): ',round(self.trange[0],1),round(self.trange[1],1))

				
    def plot_Xrho(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xrho stratification in the model""" 

        # convert nuc ID to string
        #xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddxi
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
				
        # plot DATA 
        plt.title('rhoX for '+element)
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho} \widetilde{X}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_rhoX_'+'.png')
	
    def plot_Xtransport_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xrho transport equation in the model""" 

        # convert nuc ID to string
        #xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
				
        lhs0 = self.minus_dt_eht_dd_fht_xi
        lhs1 = self.minus_div_eht_dd_fht_ux_fht_xi 
		
        rhs0 = self.minus_div_fxi
        rhs1 = self.plus_ddxidot
		
        res = self.minus_resXiTransport
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)		
				
        # plot DATA 
        plt.title('rhoX transport for '+element)
        plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t (\overline{\rho} \widetilde{X})$')
        plt.plot(grd1,lhs1,color='cyan',label = r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f$')
        plt.plot(grd1,rhs1,color='g',label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-3}$ s$^{-1}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_Xtransport_'+element+'.png')

			

        #for i in range(1,self.nx-1):
        #    dr = self.xznr[i]-self.xznl[i]
        #    self.t_mm[:,i] = self.t_dd[:,i]*(4./3.)*np.pi*dr**3
            #print(i,self.t_mm[:,i])			
		
        #self.dmdt = self.dt(self.t_mm,self.xzn0,self.t_timec,intc) 
        #print(self.dmdt)
        #self.vexp = -self.dmdt/(4.*np.pi*self.xzn0*self.xzn0*self.dd)			
			