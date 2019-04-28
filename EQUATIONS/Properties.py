import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Properties(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,params):
        ig = params.getForProp('prop')['ig'] # load geometry	
        super(Properties,self).__init__(ig) 

        filename = params.getForProp('prop')['eht_data']
        intc     = params.getForProp('prop')['intc']
		
        # load data to structured array
        eht = np.load(filename)		

        timec  = eht.item().get('timec')[intc] 
        tavg   = np.asarray(eht.item().get('tavg'))	
        trange = np.asarray(eht.item().get('trange'))
		
        # load grid
        nx  = np.asarray(eht.item().get('nx')) 	
        ny  = np.asarray(eht.item().get('ny')) 	
        nz  = np.asarray(eht.item().get('nz')) 			
		
        xzn0  = np.asarray(eht.item().get('xzn0')) 
        xznl  = np.asarray(eht.item().get('xznl'))
        xznr  = np.asarray(eht.item().get('xznr'))
		
        xbl   = params.getForProp('prop')['xbl']
        xbr   = params.getForProp('prop')['xbr']		
        laxis = params.getForProp('prop')['laxis']
		
		
        # pick pecific Reynolds-averaged mean fields according to:
        # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf			
		
        dd    = np.asarray(eht.item().get('dd')[intc])
        ux    = np.asarray(eht.item().get('ux')[intc])	
        pp    = np.asarray(eht.item().get('pp')[intc])		
		
        ddux  = np.asarray(eht.item().get('ddux')[intc])
        dduy  = np.asarray(eht.item().get('dduy')[intc])
        dduz  = np.asarray(eht.item().get('dduz')[intc])		
		
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

        enuc1 = np.asarray(eht.item().get('enuc1')[intc])
        enuc2 = np.asarray(eht.item().get('enuc2')[intc])		

        uxux = np.asarray(eht.item().get('uxux')[intc])		
		
        gamma1 = np.asarray(eht.item().get('ux')[intc])			
        #gamma1 = np.asarray(eht.item().get('gamma1')[intc])	 
		
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
		
        tke = 0.5*(uxffuxff + uyffuyff + uzffuzff)
		
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
									   						
        #######################################
        # END TURBULENT KINETIC ENERGY EQUATION 
        #######################################  

        # assign global data to be shared across whole class	
        self.xzn0 = xzn0
        self.xznl = xznl 
        self.xznr = xznr
        self.tke  = tke 		

        self.laxis = laxis
        self.xbl   = xbl
        self.xbr   = xbr
        self.nx    = nx
        self.ny    = ny		
        self.nz    = nz		
        self.dd    = dd
        self.pp    = pp
        self.uxux  = uxux
        self.enuc1 = enuc1
        self.enuc2 = enuc2
        self.gamma1 = gamma1 		
		
        self.filename = filename
        self.tavg     = tavg
        self.timec    = timec
        self.trange   = trange		
		
    def properties(self,laxis,xbl,xbr):
        """ Print properties of your simulation""" 

        ##############		
        # PROPERTIES #
        ##############
	
        xzn0 = self.xzn0
	
        # get inner and outer boundary of computational domain  
        xzn0in  = self.xzn0[0]
        xzn0out = self.xzn0[self.nx-1]

        # load density and pressure
        dd = self.dd 		
        pp = self.pp
		
        # load uxsq		
        uxux = self.uxux
		
        # load TKE
        tke = self.tke		
		
        # load TKE dissipation
        diss = abs(self.minus_resTkeEquation)

        # load enuc
        enuc1 = self.enuc1	 
        enuc2 = self.enuc2		
		
			
        # calculate INDICES for grid boundaries 
        if laxis == 0:
            idxl = 0
            idxr = self.nx-1
        if laxis == 1:
            idxl, idxr = self.idx_bndry(xbl,xbr)
        if laxis == 2:
            idxl, idxr = self.idx_bndry(xbl,xbr)			
		
        # Get rid of the numerical mess at inner boundary 
        diss[0:idxl] = 0.
        # Get rid of the numerical mess at outer boundary 
        diss[idxr:self.nx] = 0.
 
        diss_max = diss.max()
        ind = np.where( (diss > 0.02*diss_max) )[0]
		
        xzn0inc  = xzn0[ind[0]]
        xzn0outc = xzn0[ind[-1]]		

        ibot = ind[0]
        itop = ind[-1]

        lc = xzn0outc - xzn0inc		
		
        # Reynolds number
        nc = itop-ibot
        Re = nc**(4./3.)		
		
        Vol = 4./3.*np.pi*(self.xznr**3-self.xznl**3)

        # Calculate full dissipation rate and timescale
        TKE = (dd*tke*Vol)[ind].sum()
        epsD = abs((diss*Vol)[ind].sum())
        tD = TKE/epsD

        # RMS velocities
        M=(dd*Vol)[ind].sum()
        urms = np.sqrt(2.*TKE/M)

        # Turnover timescale
        tc = 2.*(xzn0outc-xzn0inc)/urms

        # Dissipation length-scale
        ld = M*urms**3./epsD

        # Total nuclear luminosity
        tenuc = ((dd*(enuc1+enuc2))*Vol)[ind].sum()

        # Pturb over Pgas 
        gamma1 = self.gamma1
        cs2 = (gamma1*pp)/dd
        ur2 = uxux
        pturb_o_pgas = (gamma1*ur2/cs2)[ind].mean()
    
        # Calculate size of convection zone in pressure scale heights

        hp = -pp/self.Grad(pp,xzn0)
        pbot = pp[ibot]
        lcz_vs_hp = np.log(pbot/pp[ibot:itop])	
		
        print('#----------------------------------------------------#')
        print('Datafile with space-time averages: ',self.filename)		
        print('Central time (in s): ',round(self.timec,1))	
        print('Averaging windows (in s): ',self.tavg.item(0))
        print('Time range (in s from-to): ',round(self.trange[0],1),round(self.trange[1],1))		
		
        print '---------------'
        print 'Resolution: %i' % self.nx,self.ny,self.nz
        print 'Radial size of computational domain (in cm): %.2e %.2e' % (xzn0in,xzn0out)
        print 'Radial size of convection zone (in cm):  %.2e %.2e' % (xzn0inc,xzn0outc)
        if laxis != 0: print 'Extent of convection zone (in Hp): %f' % lcz_vs_hp[itop-ibot-1]
        print 'Averaging time window (in s): %f' % self.tavg
        print 'RMS velocities in convection zone (in cm/s):  %.2e' % urms
        print 'Convective turnover timescale (in s)  %.2e' % tc
        print 'P_turb o P_gas %.2e' % pturb_o_pgas
        print 'Dissipation length scale (in cm): %.2e' % ld
        print 'Total nuclear luminosity (in erg/s): %.2e' % tenuc
        print 'Rate of TKE dissipation (in erg/s): %.2e' % epsD
        print 'Dissipation timescale for TKE (in s): %f' % tD
        print 'Reynolds number: %i' % Re
        #print 'Dissipation timescale for radial TKE (in s): %f' % tD_rad
        #print 'Dissipation timescale for horizontal TKE (in s): %f' % tD_hor				
		

        uconv = (2.*tke)**0.5
        if lc != 0.: 
            kolm_tke_diss_rate = (uconv**3)/lc
            tauL = tke/kolm_tke_diss_rate
        else:
            print('ERROR: Estimated size of convection zone is 0')
            kolm_tke_diss_rate = 99999999999.
            tauL = 9999999999. 			
            #sys.exit()
		
        return {'tauL':tauL,'kolm_tke_diss_rate':kolm_tke_diss_rate,'tke_diss':diss,'tke':tke,'lc':lc,'uconv':uconv,'xzn0inc':xzn0inc,'xzn0outc':xzn0outc}			
		
    def execute(self):
        p = self.properties(self.laxis,self.xbl,self.xbr)
        return {'tauL':p['tauL'],'kolm_tke_diss_rate':p['kolm_tke_diss_rate'],\
                'tke_diss':p['tke_diss'],'tke':p['tke'],'lc':p['lc'],\
                'uconv':p['uconv'],'xzn0inc':p['xzn0inc'],'xzn0outc':p['xzn0outc']}		
		
		
