import numpy as np
import matplotlib.pyplot as plt
import CALCULUS as calc
import ALIMIT as al


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

# https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf

class Xdiffusivity(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,inuc,element,lc,uconv,intc,data_prefix):
        super(Xdiffusivity,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
	
        self.data_prefix = data_prefix
        self.inuc = inuc
        self.element = element		
        self.lc = lc
        self.uconv = uconv 		
	
        self.xzn0      = np.asarray(eht.item().get('xzn0')) 
	
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
		
		# effective diffusivity
        Deff = -fxi/(dd*self.Grad(fht_xi,xzn0))
		
        # urms diffusivity		
        Durms      = (1./3.)*uconv*lc

        # pressure scale heigth
        hp = - pp/self.Grad(pp,xzn0)
		
        # mlt velocity
        alphae = 1.		
        u_mlt = fhh/(alphae*fht_cp*sigmatt)
		
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
        plt.plot(grd1,term0,label=r"$\sigma_{eff} = - f_i/(\overline{\rho} \ \partial_r \widetilde{X}_i)$")
        plt.plot(grd1,term1,label=r"$\sigma_{urms} = (1/3) \ u_{rms} \ l_c $")
        plt.plot(grd1,term2,label=r"$\sigma_{umlt} = + u_{mlt} \ l_c $")        
        plt.plot(grd1,term3,label=r"$\sigma_{umlt} = + u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.5)")
        plt.plot(grd1,term4,label=r"$\sigma_{umlt} = + u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.6)") 

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"cm$^{-2}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'Ediff_'+element+'.png')			
			
				
    def gauss(x, *p): 
    # Define model function to be used to fit to the data above:
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))		
		
	
