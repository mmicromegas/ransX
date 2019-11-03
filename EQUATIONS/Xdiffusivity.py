import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al
import os

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Xdiffusivity(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,lc,uconv,bconv,tconv,intc,data_prefix):
        super(Xdiffusivity,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        # pick specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf		
        # assign global data to be shared across whole class

        self.dd     = np.asarray(eht.item().get('dd')[intc])
        self.pp     = np.asarray(eht.item().get('pp')[intc])
        self.tt     = np.asarray(eht.item().get('tt')[intc])
        self.ddxi   = np.asarray(eht.item().get('ddx'+inuc)[intc])
        self.ddux   = np.asarray(eht.item().get('ddux')[intc])
        self.ddtt   = np.asarray(eht.item().get('ddtt')[intc])
        self.ddhh   = np.asarray(eht.item().get('ddhh')[intc])
        self.ddcp   = np.asarray(eht.item().get('ddcp')[intc])
        self.ddxiux = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        self.ddhhux = np.asarray(eht.item().get('ddhhux')[intc])
        self.ddttsq = np.asarray(eht.item().get('ddttsq')[intc])
	
        self.data_prefix = data_prefix
        self.xzn0    = np.asarray(eht.item().get('xzn0')) 
        self.element = element
        self.inuc    = inuc
        self.lc      = lc
        self.uconv   = uconv 		

        self.bconv   = bconv
        self.tconv	 = tconv 
		
    def plot_X_Ediffusivity(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        # Eulerian diffusivity
	
        # convert nuc ID to string
        xnucid = str(self.inuc)
        lc = self.lc
        uconv = self.uconv		
        element = self.element
		
        # load x GRID
        grd1 = self.xzn0		
        xzn0 = self.xzn0
		
        dd = self.dd
        pp = self.pp
		
        ddux = self.ddux   		
        ddxi = self.ddxi
        ddtt = self.ddtt
        ddhh = self.ddhh
        ddcp = self.ddcp 		
		
        ddxiux = self.ddxiux
        ddhhux = self.ddhhux
        ddttsq = self.ddttsq
		
        fht_xi = ddxi/dd
        fht_cp = ddcp/dd 

        # composition flux
        fxi = ddxiux - ddxi*ddux/dd 

        # enthalpy flux 
        fhh = ddhhux - ddhh*ddux/dd

        # variance of temperature fluctuations		
        sigmatt = (ddttsq-ddtt*ddtt/dd)/dd	

        # T_rms fluctuations
        tt_rms = sigmatt**0.5		
	
        # effective diffusivity
        Deff = -fxi/(dd*self.Grad(fht_xi,xzn0))
		
        # urms diffusivity		
        Durms      = (1./3.)*uconv*lc

        # pressure scale heigth
        hp = - pp/self.Grad(pp,xzn0)
        #print(hp)		
		
        hp = 2.5e8		
		
        # mlt velocity
        alphae = 1.		
        u_mlt = fhh/(alphae*dd*fht_cp*tt_rms)

        # this should be OS independent
        dir_model = os.path.join(os.path.realpath('.'),'DATA','INIMODEL', 'imodel.tycho')	
		
        data = np.loadtxt(dir_model,skiprows=26)		
        nxmax = 500
		
        rr = data[1:nxmax,2]
        vmlt_3 = data[1:nxmax,8]		
        u_mlt = vmlt_3		
		
        Dumlt1     = (1./3.)*u_mlt*lc		
		
        alpha = 1.5
        Dumlt2 = (1./3.)*u_mlt*alpha*hp        

        alpha = 1.6
        Dumlt3 = (1./3.)*u_mlt*alpha*hp        

        #self.lagr = (4.*np.pi*(self.xzn0**2.)*self.dd)**2.	
		
        term0 = Deff
        term1 = Durms
        term2 = Dumlt1
        term3 = Dumlt2
        term4 = Dumlt3		
				
        # create FIGURE
        plt.figure(figsize=(7,6))
	
        # set plot boundaries   
        to_plot = [term0,term1,term2,term3,term4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
	
        # plot DATA 		
        plt.title(r'Eulerian Diff for '+self.element)
        plt.plot(grd1,term0,label=r"$D_{eff} = - f_i/(\overline{\rho} \ \partial_r \widetilde{X}_i)$")
        #plt.plot(grd1,term1,label=r"$D_{urms} = (1/3) \ u_{rms} \ l_c $")
        #plt.plot(rr,term2,label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ l_c $")        
        plt.plot(rr,term3,label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.5)")
        #plt.plot(rr,term4,label=r"$D_{mlt} = + (1/3) \ u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.6)") 

        # convective boundary markers
        plt.axvline(self.bconv+0.46e8,linestyle='--',linewidth=0.7,color='k')		
        plt.axvline(self.tconv,linestyle='--',linewidth=0.7,color='k') 		
		
        # https://stackoverflow.com/questions/19206332/gaussian-fit-for-python		
		
        def gauss(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*(sigma**2)))		
		
        #p0 = [1.e15, 6.e8, 5.e7]
        #coeff, var_matrix = curve_fit(gauss, self.xzn0, Deff, p0=[1.e15, 6.e8, 5.e7])
        # Get the fitted curve
        #Deff_fit = gauss(self.xzn0, *coeff)		
		
        #plt.plot(grd1,Deff_fit,label=r"$gauss fit$",linewidth=0.7) 

        ampl = max(term3)
        #xx0 = (self.bconv+0.46e8+self.tconv)/2.
        xx0 = (self.bconv+self.tconv)/2.		
        #width = 5.e7

        #Dgauss = gauss(self.xzn0,ampl,xx0,width)		
        #plt.plot(grd1,Dgauss,color='b',label='model gauss')		
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"cm$^{-2}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':15})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'Ediff_'+element+'.png')			

		
		
	
