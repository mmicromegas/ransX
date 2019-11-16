import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class XtransportEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,bconv,tconv,intc,data_prefix):
        super(XtransportEquation,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0   = np.asarray(eht.item().get('xzn0')) 	
        nx   = np.asarray(eht.item().get('nx'))
		
        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf

        dd      = np.asarray(eht.item().get('dd')[intc])
        ux      = np.asarray(eht.item().get('ux')[intc])
        ddux    = np.asarray(eht.item().get('ddux')[intc])	
        dduxux  = np.asarray(eht.item().get('dduxux')[intc])
        ddxi    = np.asarray(eht.item().get('ddx'+inuc)[intc])
        ddxiux  = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        ddxidot = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])	
		
        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])		
        divu = np.asarray(eht.item().get('divu')[intc])
        gamma1 = np.asarray(eht.item().get('gamma1')[intc])
        gamma3 = np.asarray(eht.item().get('gamma3')[intc])
		
        uxdivu = np.asarray(eht.item().get('ux')[intc])		
        gamma1 = np.asarray(eht.item().get('ux')[intc])
        gamma3 = np.asarray(eht.item().get('ux')[intc])		
		
        fht_rxx = dduxux - ddux*ddux/dd
        fdil = (uxdivu - ux*divu) 		
		
        #######################
        # Xi TRANSPORT EQUATION 
        #######################
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec')) 	
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_ddxi    = np.asarray(eht.item().get('ddx'+inuc))	
        t_fht_xi  = t_ddxi/t_dd		
		
        # construct equation-specific mean fields
        fht_ux = ddux/dd
        fht_xi = ddxi/dd
        fxi    = ddxiux - ddxi*ddux/dd	 
		
        # LHS -dq/dt 		
        self.minus_dt_dd_fht_xi = -self.dt(t_dd*t_fht_xi,xzn0,t_timec,intc)
		
        # LHS -div(ddXiux)
        self.minus_div_eht_dd_fht_ux_fht_xi = -self.Div(dd*fht_ux*fht_xi,xzn0)
		
        # RHS -div fxi 
        self.minus_div_fxi = -self.Div(fxi,xzn0) 
		
        # RHS +ddXidot 
        self.plus_ddxidot = +ddxidot 
		
        # -res
        self.minus_resXiTransport = -(self.minus_dt_dd_fht_xi + self.minus_div_eht_dd_fht_ux_fht_xi + \
                               self.minus_div_fxi + self.plus_ddxidot)
		
        ###########################		
        # END Xi TRANSPORT EQUATION
        ###########################
		
        # grad models		
        self.plus_gradx_fht_xi = +self.Grad(fht_xi,xzn0)
        cnst = gamma1
        self.minus_cnst_dd_fht_xi_fdil_o_fht_rxx = -cnst*dd*fht_xi*fdil/fht_rxx			
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0    = xzn0
        self.nx      = nx		
        self.inuc    = inuc
        self.element = element
        self.ddxi    = ddxi	
        self.fht_xi  = fht_xi
		
        self.bconv   = bconv
        self.tconv	 = tconv 		
        self.ig      = ig		
		
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
			
        setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_rhoX_'+element+'.png')
	
    def plot_X(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot X stratification in the model""" 

        # convert nuc ID to string
        #xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.fht_xi
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
	
        # plot DATA 
        plt.title('X for '+element)
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{X}$')

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
			
        setylabel = r"$\widetilde{X}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_X_'+element+'.png')	
	
    def plot_gradX(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot grad X stratification in the model""" 

        # convert nuc ID to string
        #xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.plus_gradx_fht_xi
        plt2 = self.minus_cnst_dd_fht_xi_fdil_o_fht_rxx		
				
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [plt1,plt2]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
	
        # plot DATA 
        plt.title('X for '+element)
        plt.plot(grd1,plt1,color='brown',label = r'$\partial_r \widetilde{X}$')
        plt.plot(grd1,plt2,color='r',label = r'$.$')
		
        # convective boundary markers
        plt.axvline(self.bconv+0.46e8,linestyle='--',linewidth=0.7,color='k')		
        plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k')			
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()
			
        setylabel = r"$\partial_r \widetilde{X}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_gradX_'+element+'.png')	
	
    def plot_Xtransport_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot Xrho transport equation in the model""" 

        # convert nuc ID to string
        #xnucid = str(self.inuc)
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
				
        lhs0 = self.minus_dt_dd_fht_xi
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
        if (self.ig == 1):
            plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t (\overline{\rho} \widetilde{X})$')
            plt.plot(grd1,lhs1,color='cyan',label = r'$-\nabla_x (\overline{\rho} \widetilde{X} \widetilde{u}_x)$')		
            plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_x f$')
            plt.plot(grd1,rhs1,color='g',label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
            plt.plot(grd1,res,color='k',linestyle='--',label='res')			
        elif (self.ig == 2):
            plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t (\overline{\rho} \widetilde{X})$')
            plt.plot(grd1,lhs1,color='cyan',label = r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$')		
            plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f$')
            plt.plot(grd1,rhs1,color='g',label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
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
			
        setylabel = r"g cm$^{-3}$ s$^{-1}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_Xtransport_'+element+'.png')
		
    def plot_Xtransport_equation_integral_budget(self,laxis,xbl,xbr,ybu,ybd):
        """Plot integral budgets of composition transport equation in the model""" 

        element = self.element
		
        # load x GRID
        grd1 = self.xzn0
        nx = self.nx		

        term1 = self.minus_dt_dd_fht_xi 
        term2 = self.minus_div_eht_dd_fht_ux_fht_xi
        term3 = self.minus_div_fxi
        term4 = self.plus_ddxidot
        term5 = self.minus_resXiTransport
		
        # calculate INDICES for grid boundaries 
        if laxis == 1 or laxis == 2:
            idxl, idxr = self.idx_bndry(xbl,xbr)
        else:
            idxl = 0
            idxr = self.nx-1
		
        term1_sel = term1[idxl:idxr]
        term2_sel = term2[idxl:idxr]
        term3_sel = term3[idxl:idxr]
        term4_sel = term4[idxl:idxr]
        term5_sel = term5[idxl:idxr]
		
        rc = self.xzn0[idxl:idxr]

        Sr = 4.*np.pi*rc**2

        int_term1 = integrate.simps(term1_sel*Sr,rc)
        int_term2 = integrate.simps(term2_sel*Sr,rc)
        int_term3 = integrate.simps(term3_sel*Sr,rc) 
        int_term4 = integrate.simps(term4_sel*Sr,rc)     
        int_term5 = integrate.simps(term5_sel*Sr,rc)
		
        fig = plt.figure(figsize=(7,6))
    
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        if laxis == 2:		
            plt.ylim([ybd,ybu])	 
	 
        fc = 1.
    
        # note the change: I'm only supplying y data.
        y = [int_term1/fc,int_term2/fc,int_term3/fc,int_term4/fc,int_term5/fc]

        # Calculate how many bars there will be
        N = len(y)
 
        # Generate a list of numbers, from 0 to N
        # This will serve as the (arbitrary) x-axis, which
        # we will then re-label manually.
        ind = range(N)
 
        # See note below on the breakdown of this command
        ax.bar(ind, y, facecolor='#0000FF',
               align='center', ecolor='black')
 
        #Create a y label
        ax.set_ylabel(r'g s$^{-1}$')
 
        # Create a title, in italics
        ax.set_title('rhoX transport budget for '+element)
 
        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)
 
        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        group_labels = [r'$-\partial_t (\overline{\rho} \widetilde{X})$',\
                        r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$',\
                        r'$-\nabla_r f$',r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$','res']
                         
        # Set the x tick labels to the group_labels defined above.
        ax.set_xticklabels(group_labels,fontsize=16)
 
        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        fig.autofmt_xdate()
        
        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'xtransport_'+element+'_eq_bar.png')				
		