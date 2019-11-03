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

class FullTurbulenceVelocityFieldHypothesis(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,ieos,intc,data_prefix):
        super(FullTurbulenceVelocityFieldHypothesis,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)		

        # load grid
        xzn0 = np.asarray(eht.item().get('xzn0')) 	
        nx = np.asarray(eht.item().get('nx')) 
		
        # pick equation-specific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf	

        dd = np.asarray(eht.item().get('dd')[intc])
        ux = np.asarray(eht.item().get('ux')[intc])
        uy = np.asarray(eht.item().get('uy')[intc])
        uz = np.asarray(eht.item().get('uz')[intc])		

        pp     = np.asarray(eht.item().get('pp')[intc])	
        ddgg   = -np.asarray(eht.item().get('ddgg')[intc])
        gamma1 = np.asarray(eht.item().get('gamma1')[intc])
		
        ddux  = np.asarray(eht.item().get('ddux')[intc])		
        dduy  = np.asarray(eht.item().get('dduy')[intc])
        dduz  = np.asarray(eht.item().get('dduz')[intc])		
		
        uxux = np.asarray(eht.item().get('uxux')[intc])
        uxuy = np.asarray(eht.item().get('uxuy')[intc])
        uxuz = np.asarray(eht.item().get('uxuz')[intc])

        dduxux = np.asarray(eht.item().get('dduxux')[intc])
        dduxuy = np.asarray(eht.item().get('dduxuy')[intc])
        dduxuz = np.asarray(eht.item().get('dduxuz')[intc])
		
        divu  = np.asarray(eht.item().get('divu')[intc])		
        dddivu  = np.asarray(eht.item().get('dddivu')[intc])
		
        uxdivu  = np.asarray(eht.item().get('uxdivu')[intc])
        uydivu  = np.asarray(eht.item().get('uydivu')[intc])
        uzdivu  = np.asarray(eht.item().get('uzdivu')[intc])		

        uxdivux  = np.asarray(eht.item().get('uxdivux')[intc])
        uydivux  = np.asarray(eht.item().get('uydivux')[intc])
        uzdivux  = np.asarray(eht.item().get('uzdivux')[intc])

        uxdivuy  = np.asarray(eht.item().get('uxdivuy')[intc])
        uydivuy  = np.asarray(eht.item().get('uydivuy')[intc])
        uzdivuy  = np.asarray(eht.item().get('uzdivuy')[intc])
		
        uxdivuz  = np.asarray(eht.item().get('uxdivuz')[intc])
        uydivuz  = np.asarray(eht.item().get('uydivuz')[intc])
        uzdivuz  = np.asarray(eht.item().get('uzdivuz')[intc])		
		
        divux = np.asarray(eht.item().get('divux')[intc])
        divuy = np.asarray(eht.item().get('divuy')[intc])	
        divuz = np.asarray(eht.item().get('divuz')[intc])		
		
        dduxdivu  = np.asarray(eht.item().get('dduxdivu')[intc])
        dduydivu  = np.asarray(eht.item().get('dduydivu')[intc])
        dduzdivu  = np.asarray(eht.item().get('dduzdivu')[intc])		

        dduxdivux  = np.asarray(eht.item().get('dduxdivux')[intc])
        dduydivux  = np.asarray(eht.item().get('dduydivux')[intc])
        dduzdivux  = np.asarray(eht.item().get('dduzdivux')[intc])

        dduxdivuy  = np.asarray(eht.item().get('dduxdivuy')[intc])
        dduydivuy  = np.asarray(eht.item().get('dduydivuy')[intc])
        dduzdivuy  = np.asarray(eht.item().get('dduzdivuy')[intc])
		
        dduxdivuz  = np.asarray(eht.item().get('dduxdivuz')[intc])
        dduydivuz  = np.asarray(eht.item().get('dduydivuz')[intc])
        dduzdivuz  = np.asarray(eht.item().get('dduzdivuz')[intc])		
		
        dddivux = np.asarray(eht.item().get('dddivux')[intc])
        dddivuy = np.asarray(eht.item().get('dddivuy')[intc])	
        dddivuz = np.asarray(eht.item().get('dddivuz')[intc])		

        dduxuxx  = np.asarray(eht.item().get('dduxuxx')[intc])
        dduyuxx  = np.asarray(eht.item().get('dduyuxx')[intc])
        dduzuxx  = np.asarray(eht.item().get('dduzuxx')[intc])

        dduxuyy  = np.asarray(eht.item().get('dduxuyy')[intc])
        dduyuyy  = np.asarray(eht.item().get('dduyuyy')[intc])
        dduzuyy  = np.asarray(eht.item().get('dduzuyy')[intc])
		
        dduxuzz  = np.asarray(eht.item().get('dduxuzz')[intc])
        dduyuzz  = np.asarray(eht.item().get('dduyuzz')[intc])
        dduzuzz  = np.asarray(eht.item().get('dduzuzz')[intc])		

        #dduxx = np.asarray(eht.item().get('dduxx')[intc])
        #dduyy = np.asarray(eht.item().get('dduyy')[intc])	
        #dduzz = np.asarray(eht.item().get('dduzz')[intc])		
		
        uxuxx  = np.asarray(eht.item().get('uxuxx')[intc])
        uyuxx  = np.asarray(eht.item().get('uyuxx')[intc])
        uzuxx  = np.asarray(eht.item().get('uzuxx')[intc])

        uxuyy  = np.asarray(eht.item().get('uxuyy')[intc])
        uyuyy  = np.asarray(eht.item().get('uyuyy')[intc])
        uzuyy  = np.asarray(eht.item().get('uzuyy')[intc])
		
        uxuzz  = np.asarray(eht.item().get('uxuzz')[intc])
        uyuzz  = np.asarray(eht.item().get('uyuzz')[intc])
        uzuzz  = np.asarray(eht.item().get('uzuzz')[intc])

        uxx = np.asarray(eht.item().get('uxx')[intc])
        uyy = np.asarray(eht.item().get('uyy')[intc])	
        uzz = np.asarray(eht.item().get('uzz')[intc])			

        pp = np.asarray(eht.item().get('pp')[intc])		
        divu = np.asarray(eht.item().get('divu')[intc])
        dddivu = np.asarray(eht.item().get('dddivu')[intc])		
        ppdivu = np.asarray(eht.item().get('ppdivu')[intc])		
		
        ppux = np.asarray(eht.item().get('ppux')[intc])		
        uxdivu = np.asarray(eht.item().get('uxdivu')[intc])	
        uxppdivu = np.asarray(eht.item().get('uxppdivu')[intc])		

        ppuy = np.asarray(eht.item().get('ppuy')[intc])		
        uydivu = np.asarray(eht.item().get('uydivu')[intc])	
        uyppdivu = np.asarray(eht.item().get('uyppdivu')[intc])

        ppuz = np.asarray(eht.item().get('ppuz')[intc])		
        uzdivu = np.asarray(eht.item().get('uzdivu')[intc])	
        uzppdivu = np.asarray(eht.item().get('uzppdivu')[intc])		

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if(ieos == 1):
            cp = np.asarray(eht.item().get('cp')[intc])   
            cv = np.asarray(eht.item().get('cv')[intc])
            gamma1 = cp/cv   # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
        
        # construct equation-specific mean fields		
        fht_ux = ddux/dd			
        fht_uy = dduy/dd
        fht_uz = dduz/dd		
	
        ###########################################
        # FULL TURBULENCE VELOCITY FIELD HYPOTHESIS
        ###########################################
						
        if (True):
            self.rxx = uxux - ux*ux
            self.rxy = uxuy - ux*uy
            self.rxz = uxuz - ux*uz		

        if (False):   		
            self.rxx = dduxux/dd - ddux*ddux/(dd*dd)
            self.ryx = dduxuy/dd - ddux*dduy/(dd*dd)
            self.rzx = dduxuz/dd - ddux*dduz/(dd*dd)	
			
        self.eht_uxf_divuf   = uxdivu  - ux*divu			
        self.eht_uyf_divuf   = uydivu  - uy*divu
        self.eht_uzf_divuf   = uzdivu  - uz*divu			
		
        self.eht_uxf_divuxf  = uxdivux - ux*divux
        self.eht_uxf_divuyf  = uxdivuy - ux*divuy			
        self.eht_uxf_divuzf  = uxdivuz - ux*divuz
		
        self.eht_uyf_divuxf  = uydivux - uy*divux
        self.eht_uyf_divuyf  = uydivuy - uy*divuy			
        self.eht_uyf_divuzf  = uydivuz - uy*divuz				
		
        self.eht_uzf_divuxf  = uzdivux - uz*divux
        self.eht_uzf_divuyf  = uzdivuy - uz*divuy			
        self.eht_uzf_divuzf  = uzdivuz - uz*divuz				
       	
        self.eht_uxff_divuff   = dduxdivu/dd  - ddux*dddivu/(dd*dd)			
        self.eht_uyff_divuff   = dduydivu/dd  - dduy*dddivu/(dd*dd)
        self.eht_uzff_divuff   = dduzdivu/dd  - dduz*dddivu/(dd*dd)			
		
        self.eht_uxff_divuxff  = dduxdivux/dd - ddux*dddivux/(dd*dd)
        self.eht_uxff_divuyff  = dduxdivuy/dd - ddux*dddivuy/(dd*dd)			
        self.eht_uxff_divuzff  = dduxdivuz/dd - ddux*dddivuz/(dd*dd)
		
        self.eht_uyff_divuxff  = dduydivux/dd - dduy*dddivux/(dd*dd)
        self.eht_uyff_divuyff  = dduydivuy/dd - dduy*dddivuy/(dd*dd)			
        self.eht_uyff_divuzff  = dduydivuz/dd - dduy*dddivuz/(dd*dd)				
		
        self.eht_uzff_divuxff  = dduzdivux/dd - dduz*dddivux/(dd*dd)
        self.eht_uzff_divuyff  = dduzdivuy/dd - dduz*dddivuy/(dd*dd)			
        self.eht_uzff_divuzff  = dduzdivuz/dd - dduz*dddivuz/(dd*dd)
		
        self.eht_uxf_uxxf  = uxuxx - ux*uxx
        self.eht_uxf_uyyf  = uxuyy - ux*uyy			
        self.eht_uxf_uzzf  = uxuzz - ux*uzz
		
        self.eht_uyf_uxxf  = uyuxx - uy*uxx
        self.eht_uyf_uyyf  = uyuyy - uy*uyy			
        self.eht_uyf_uzzf  = uyuzz - uy*uzz				
		
        self.eht_uzf_uxxf  = uzuxx - uz*uxx
        self.eht_uzf_uyyf  = uzuyy - uz*uyy			
        self.eht_uzf_uzzf  = uzuzz - uz*uzz				
       	
        self.eht_uxff_divuff   = dduxdivu/dd  - ddux*dddivu/(dd*dd)			
        self.eht_uyff_divuff   = dduydivu/dd  - dduy*dddivu/(dd*dd)
        self.eht_uzff_divuff   = dduzdivu/dd  - dduz*dddivu/(dd*dd)			
		
        #self.eht_uxff_uxxff  = dduxuxx/dd - ddux*dduxx/(dd*dd)
        #self.eht_uxff_uyyff  = dduxuyy/dd - ddux*dduyy/(dd*dd)			
        #self.eht_uxff_uzzff  = dduxuzz/dd - ddux*dduzz/(dd*dd)
		
        #self.eht_uyff_uxxff  = dduyuxx/dd - dduy*dduxx/(dd*dd)
        #self.eht_uyff_uyyff  = dduyuyy/dd - dduy*dduyy/(dd*dd)			
        #self.eht_uyff_uzzff  = dduyuzz/dd - dduy*dduzz/(dd*dd)				
		
        #self.eht_uzff_uxxff  = dduzuxx/dd - dduz*dduxx/(dd*dd)
        #self.eht_uzff_uyyff  = dduzuyy/dd - dduz*dduyy/(dd*dd)			
        #self.eht_uzff_uzzff  = dduzuzz/dd - dduz*dduzz/(dd*dd)						
						
					
        ###############################################
        # END FULL TURBULENCE VELOCITY FIELD HYPOTHESIS
        ###############################################
		
        self.eht_uxfppdivu = uxppdivu - ux*ppdivu		
        self.ppfuxf_fht_divu = (ppux - pp*ux)*(dddivu/dd)
        self.pp_eht_uxf_divuff = pp*(uxdivu-ux*divu)
        self.eht_ppf_uxf_divuff = ppux*divu-ppux*(dddivu/dd)-pp*ux*divu+pp*ux*(dddivu/dd)
		
        self.eht_uyfppdivu = uyppdivu - uy*ppdivu		
        self.ppfuyf_fht_divu = (ppuy - pp*uy)*(dddivu/dd)
        self.pp_eht_uyf_divuff = pp*(uydivu-uy*divu)
        self.eht_ppf_uyf_divuff = ppuy*divu-ppuy*(dddivu/dd)-pp*uy*divu+pp*uy*(dddivu/dd)
		
        self.eht_uzfppdivu = uzppdivu - uz*ppdivu		
        self.ppfuzf_fht_divu = (ppuz - pp*uz)*(dddivu/dd)
        self.pp_eht_uzf_divuff = pp*(uzdivu-uz*divu)
        self.eht_ppf_uzf_divuff = ppuz*divu-ppuz*(dddivu/dd)-pp*uz*divu+pp*uz*(dddivu/dd)
        self.eht_divu1 = divu
        self.eht_divu2 = divux+divuy+divuz		
		
		
        # assign global data to be shared across whole class
        self.data_prefix = data_prefix		
        self.xzn0 = xzn0
        self.dd   = dd	
        self.nx   = nx
        self.ig   = ig		

        self.pp     = pp       
        self.ddgg   = ddgg   
        self.gamma1 = gamma1 
		
    def plot_ftvfhX_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot ftvfh in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_uxf_uxxf
        plt2 = self.eht_uxf_uyyf
        plt3 = self.eht_uxf_uzzf
		
        #plt1 = self.eht_uxf_divuxf		
        #plt2 = self.eht_uxf_divuyf		
        #plt3 = self.eht_uxf_divuzf		
		
        plt4 = -self.ddgg*self.rxx/(self.gamma1*self.pp)
        res = plt1+plt2+plt3+plt4			
		
		# create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2,plt3,plt4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title(r"turbulence velocity field hypothesis r")
        plt.plot(grd1,plt1,color='r',label = r"$+\overline{u'_r \nabla_r u'_r}$")
        plt.plot(grd1,plt2,color='g',label = r"$+\overline{u'_r \nabla_\theta u'_\theta}$")		
        plt.plot(grd1,plt3,color='b',label = r"$+\overline{u'_r \nabla_\phi u'_\phi}$")
        plt.plot(grd1,plt4,color='m',label = r"$-\overline{\rho} \ \overline{u'_r u'_r} \ \widetilde{g}_r/\Gamma_1 \ \overline{P}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label = r"$res$")
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 
			
        setylabel = r"cm s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'full_turb_velX_field_hypothesis.png')
        plt.savefig('RESULTS/'+self.data_prefix+'full_turb_velX_field_hypothesis.eps')
		

    def plot_ftvfhY_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot ftvfh in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_uyf_uxxf
        plt2 = self.eht_uyf_uyyf
        plt3 = self.eht_uyf_uzzf
        plt4 = -self.ddgg*self.rxy/(self.gamma1*self.pp)
        res = plt1+plt2+plt3+plt4					
		
		# create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2,plt3,plt4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title(r"turbulence velocity field hypothesis $\theta$")
        plt.plot(grd1,plt1,color='r',label = r"$+\overline{u'_\theta \nabla_r u'_r}$")
        plt.plot(grd1,plt2,color='g',label = r"$+\overline{u'_\theta \nabla_\theta u'_\theta}$")		
        plt.plot(grd1,plt3,color='b',label = r"$+\overline{u'_\theta \nabla_\phi u'_\phi}$")
        plt.plot(grd1,plt4,color='m',label = r"$-\overline{\rho} \ \overline{u'_\theta u'_r} \ \widetilde{g}_r/\Gamma_1 \ \overline{P}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label = r"$res$")
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 
			
        setylabel = r"cm s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'full_turb_velY_field_hypothesis.png')	
        plt.savefig('RESULTS/'+self.data_prefix+'full_turb_velY_field_hypothesis.eps')
		
    def plot_ftvfhZ_equation(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot ftvfh in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_uzf_uxxf
        plt2 = self.eht_uzf_uyyf
        plt3 = self.eht_uzf_uzzf
        plt4 = -self.ddgg*self.rxz/(self.gamma1*self.pp)
        res = plt1+plt2+plt3+plt4		
		
		# create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries    
        to_plot = [plt1,plt2,plt3,plt4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title(r"turbulence velocity field hypothesis $\phi$")
        plt.plot(grd1,plt1,color='r',label = r"$+\overline{u'_\phi \nabla_r u'_r}$")
        plt.plot(grd1,plt2,color='g',label = r"$+\overline{u'_\phi \nabla_\theta u'_\theta}$")		
        plt.plot(grd1,plt3,color='b',label = r"$+\overline{u'_\phi \nabla_\phi u'_\phi}$")
        plt.plot(grd1,plt4,color='m',label = r"$-\overline{\rho} \ \overline{u'_\phi u'_r} \ \widetilde{g}_r/\Gamma_1 \ \overline{P}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label = r"$res$")
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 
			
        setylabel = r"cm s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'full_turb_velZ_field_hypothesis.png')	
        plt.savefig('RESULTS/'+self.data_prefix+'full_turb_velZ_field_hypothesis.eps')
		
	
    def plot_uxfpd_identity(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot upd in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_uxfppdivu
        plt2 = -self.ppfuxf_fht_divu
        plt3 = -self.pp_eht_uxf_divuff
        plt4 = -self.eht_ppf_uxf_divuff 
        res = plt1+plt2+plt3+plt4		
		
		# create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries    
        to_plot = [plt1,plt2,plt3,plt4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('uxpd identity')
        plt.plot(grd1,plt1,color='r',label = r"$+\overline{u'_r P d}$")
        plt.plot(grd1,plt2,color='g',label = r"$-\overline{P'u'_r} \ \widetilde{d}$")		
        plt.plot(grd1,plt3,color='b',label = r"$-\overline{P} \ \overline{u'_r d''}$")
        plt.plot(grd1,plt4,color='m',label = r"$-\overline{P' u'_r d''}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label = r"$res$")
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 
			
        setylabel = r"cm s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'uxfpd_identity.png')		
	
    def plot_uyfpd_identity(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot upd in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_uyfppdivu
        plt2 = -self.ppfuyf_fht_divu
        plt3 = -self.pp_eht_uyf_divuff
        plt4 = -self.eht_ppf_uyf_divuff 
        res = plt1+plt2+plt3+plt4		
		
		# create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries    
        to_plot = [plt1,plt2,plt3,plt4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('uypd identity')
        plt.plot(grd1,plt1,color='r',label = r"$+\overline{u'_\theta P d}$")
        plt.plot(grd1,plt2,color='g',label = r"$-\overline{P'u'_\theta} \ \widetilde{d}$")		
        plt.plot(grd1,plt3,color='b',label = r"$-\overline{P} \ \overline{u'_\theta d''}$")
        plt.plot(grd1,plt4,color='m',label = r"$-\overline{P' u'_\theta d''}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label = r"$res$")
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 
			
        setylabel = r"cm s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'uyfpd_identity.png')		
		
	
    def plot_uzfpd_identity(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot upd in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_uzfppdivu
        plt2 = -self.ppfuzf_fht_divu
        plt3 = -self.pp_eht_uzf_divuff
        plt4 = -self.eht_ppf_uzf_divuff 
        res = plt1+plt2+plt3+plt4		
		
		# create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries    
        to_plot = [plt1,plt2,plt3,plt4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('uzpd identity')
        plt.plot(grd1,plt1,color='r',label = r"$+\overline{u'_\phi P d}$")
        plt.plot(grd1,plt2,color='g',label = r"$-\overline{P'u'_\phi} \ \widetilde{d}$")		
        plt.plot(grd1,plt3,color='b',label = r"$-\overline{P} \ \overline{u'_\phi d''}$")
        plt.plot(grd1,plt4,color='m',label = r"$-\overline{P' u'_\phi d''}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label = r"$res$")
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 
			
        setylabel = r"cm s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'uzfpd_identity.png')	

    def plot_divu(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot divu in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.eht_divu1
        plt2 = self.eht_divu2		
        
	# create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries    
        to_plot = [plt1,plt2]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('divu')
        plt.plot(grd1,plt1,color='r',label = r"$divu1$")
        plt.plot(grd1,plt2,color='g',label = r"$divu2$")				
		
        # define and show x/y LABELS
        if (self.ig == 1):	
            setxlabel = r'x (10$^{8}$ cm)'	
        elif (self.ig == 2):	
            setxlabel = r'r (10$^{8}$ cm)'
        else:
            print("ERROR: geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit() 
			
        setylabel = r"cm s$^{-2}$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+self.data_prefix+'divu.png')	

		
	
