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

class DensitySpecificVolumeCovarianceEquation(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(DensitySpecificVolumeCovarianceEquation,self).__init__(ig) 
	
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
        self.sv        = np.asarray(eht.item().get('sv')[intc])
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])		
        self.svux      = np.asarray(eht.item().get('svux')[intc])	
        self.svdivu    = np.asarray(eht.item().get('svdivu')[intc])		
        self.divu      = np.asarray(eht.item().get('divu')[intc])	
	
        xzn0 = self.xzn0
		
        # store time series for time derivatives
        t_timec   = np.asarray(eht.item().get('timec'))		
        t_dd      = np.asarray(eht.item().get('dd')) 
        t_sv      = np.asarray(eht.item().get('sv')) 		

 		# pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/ransXtoPROMPI.pdf/	
		
        dd   = self.dd
        ux   = self.ux
        sv   = self.sv
        ddux = self.ddux
        svux = self.svux
        divu = self.divu
        svdivu = self.svdivu
		
        # construct equation-specific mean fields		
        fht_ux = ddux/dd	
        b = 1.-sv*dd

        t_b = 1.-t_sv*t_dd
		
        ##################################################
        # DENSITY-SPECIFIC VOLUME COVARIANCE or B EQUATION 
        ##################################################
        
        # LHS -db/dt 		
        self.minus_dt_b = self.dt(t_b,xzn0,t_timec,intc)

        # LHS -fht_ux Grad b
        self.minus_fht_ux_gradx_b = fht_ux*self.Grad(b,xzn0)
				
        # RHS +sv Div dd uxff 
        self.plus_eht_sv_div_eht_dd_uxff = sv*self.Div(dd*(ux-ddux/dd),xzn0) 

        # RHS -eht_dd Div uxf svf
        self.minus_eht_dd_div_uxf_svf = -dd*self.Div(svux-sv*ux,xzn0)

        # RHS +2 eht_dd eht svf df
        self.plus_two_eht_dd_eht_svf_df = 2.*dd*(svdivu-sv*divu)

        # -res
        self.minus_resBequation = -(self.minus_dt_b + self.minus_fht_ux_gradx_b + self.plus_eht_sv_div_eht_dd_uxff + \
           self.minus_eht_dd_div_uxf_svf + self.plus_two_eht_dd_eht_svf_df)
        				
        ######################################################
        # END DENSITY-SPECIFIC VOLUME COVARIANCE or B EQUATION 
        ######################################################						

    def plot_b(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot density-specific volume covariance stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = 1.-self.sv*self.dd
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('density-specific volume covariance')
        plt.plot(grd1,plt1,color='brown',label = r'$b$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"b"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'mean_b.png')		
		
						
    def plot_b_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot density-specific volume covariance equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = self.minus_dt_b
        lhs1 = self.minus_fht_ux_gradx_b
		
        rhs0 = self.plus_eht_sv_div_eht_dd_uxff
        rhs1 = self.minus_eht_dd_div_uxf_svf
        rhs2 = self.plus_two_eht_dd_eht_svf_df
		
        res = self.minus_resBequation
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		

        # set plot boundaries   
        to_plot = [lhs0,lhs1,rhs0,rhs1,rhs2,res]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)
		
        # plot DATA 
        plt.title('b equation')
        plt.plot(grd1,lhs0,color='c',label = r"$-\partial_t b$")
        plt.plot(grd1,lhs1,color='m',label = r"$-\nabla_r b $")		
        plt.plot(grd1,rhs0,color='b',label=r"$+v \nabla_r (\overline{v} \overline{u''_r})$")
        plt.plot(grd1,rhs1,color='g',label=r"$-v \nabla_r (\overline{\rho} \overline{(u'_r v'})$")
        plt.plot(grd1,rhs2,color='r',label=r"$+2 \overline{\rho} \overline{v'd'}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"b"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'b_eq.png')			
		
		
		
		
		