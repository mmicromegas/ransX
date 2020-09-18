import numpy as np
import sys
import UTILS.Calculus as uCalc
import UTILS.SetAxisLimit as uSal
import UTILS.Errors as eR
import UTILS.Tools as uT
import EQUATIONS.TurbulentKineticEnergyCalculation as tkeCalc
import EQUATIONS.ContinuityEquationWithMassFluxCalculation as contCalc
import EQUATIONS.TotalEnergyEquationCalculation as teeCalc


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Properties(uCalc.Calculus, uSal.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr):
        super(Properties, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        timec = self.getRAdata(eht, 'timec')[intc]
        tavg = self.getRAdata(eht, 'tavg')
        trange = self.getRAdata(eht, 'trange')

        # load grid
        nx = self.getRAdata(eht, 'nx')
        ny = self.getRAdata(eht, 'ny')
        nz = self.getRAdata(eht, 'nz')

        xzn0 = self.getRAdata(eht, 'xzn0')
        xznl = self.getRAdata(eht, 'xznl')
        xznr = self.getRAdata(eht, 'xznr')

        yzn0 = self.getRAdata(eht, 'yzn0')
        zzn0 = self.getRAdata(eht, 'zzn0')

        enuc1 = self.getRAdata(eht, 'enuc1')[intc]
        enuc2 = self.getRAdata(eht, 'enuc2')[intc]

        dd = self.getRAdata(eht, 'dd')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]
        uxux = self.getRAdata(eht, 'uxux')[intc]
        gamma1 = self.getRAdata(eht, 'gamma1')[intc]

        if plabel == 'ccptwo':
            ddux = self.getRAdata(eht, 'ddux')[intc]
            ddxi = self.getRAdata(eht, 'ddx0001')[intc]
            ddxiux = self.getRAdata(eht, 'ddx0001ux')[intc]
            fxi = ddxiux - ddxi * ddux / dd
        else:
            ddux = self.getRAdata(eht, 'ddux')[intc]
            ddxi = self.getRAdata(eht, 'ddx0005')[intc]
            ddxiux = self.getRAdata(eht, 'ddx0005ux')[intc]
            fxi = ddxiux - ddxi * ddux / dd
            # fxi = np.zeros(nx)

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma1 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110
            # gamma3 = gamma1

        pp = self.getRAdata(eht, 'pp')[intc]
        tt = self.getRAdata(eht, 'tt')[intc]
        mu = self.getRAdata(eht, 'abar')[intc]
        chim = self.getRAdata(eht, 'chim')[intc]
        chit = self.getRAdata(eht, 'chit')[intc]
        gamma2 = self.getRAdata(eht, 'gamma2')[intc]
        # print(chim,chit,gamma2)

        # override gamma for ideal gas eos (need to be fixed in PROMPI later)
        if ieos == 1:
            cp = self.getRAdata(eht, 'cp')[intc]
            cv = self.getRAdata(eht, 'cv')[intc]
            gamma2 = cp / cv  # gamma1,gamma2,gamma3 = gamma = cp/cv Cox & Giuli 2nd Ed. page 230, Eq.9.110

        lntt = np.log(tt)
        lnpp = np.log(pp)
        lnmu = np.log(mu)

        # calculate temperature gradients
        self.nabla = self.deriv(lntt, lnpp)
        self.nabla_ad = (gamma2 - 1.) / gamma2

        # for ccp project
        if plabel == 'ccptwo':
            x0002 = self.getRAdata(eht, 'x0002')[intc]
        elif plabel == 'oburn':
            x0002 = self.getRAdata(eht, 'x0002')[intc]  # track prot
        elif plabel == 'neshell':
            x0002 = self.getRAdata(eht, 'x0002')[intc]  # track
        elif plabel == 'heflash':
            x0002 = self.getRAdata(eht, 'x0002')[intc]  # track
        elif plabel == 'thpulse':
            x0002 = self.getRAdata(eht, 'x0002')[intc]  # track
        elif plabel == 'cflash':
            x0002 = self.getRAdata(eht, 'x0002')[intc]  # track
        elif plabel == 'heflash':
            x0002 = self.getRAdata(eht, 'x0002')[intc]  # track
        else:
            x0002 = np.zeros(nx)

        ####################################################################

        # instantiate turbulent kinetic energy object
        tkeF = tkeCalc.TurbulentKineticEnergyCalculation(filename, ig, intc)

        # load fields
        tkefields = tkeF.getTKEfield()

        # get turbulent kinetic energy
        self.tke = tkefields['tke']

        # get turbulent kinetic energy dissipation
        self.minus_resTkeEquation = tkefields['minus_resTkeEquation']

        ####################################################################

        # instantiate continuity equation object
        contF = contCalc.ContinuityEquationWithMassFluxCalculation(filename, ig, intc)

        # load fields
        contfields = contF.getCONTfield()

        # get residual of the continuity equation
        self.minus_resContEquation = contfields['minus_resContEquation']

        ####################################################################

        # instantiate total energy equation object
        teeF = teeCalc.TotalEnergyEquationCalculation(filename, ig, intc)

        # load fields
        teefields = teeF.getTotalEnergyEquationField()

        # get residual of the total energy equation equation
        self.minus_resTeEquation = teefields['minus_resTeEquation']

        ####################################################################

        # assign global data to be shared across whole class
        self.xzn0 = xzn0
        self.xznl = xznl
        self.xznr = xznr

        self.yzn0 = yzn0
        self.zzn0 = zzn0

        self.xbl = xbl
        self.xbr = xbr
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dd = dd
        self.pp = pp
        self.uxux = uxux
        self.enuc1 = enuc1
        self.enuc2 = enuc2
        self.gamma1 = gamma1

        self.filename = filename
        self.plabel = plabel
        self.tavg = tavg
        self.timec = timec
        self.trange = trange
        self.ig = ig
        self.nsdim = nsdim
        self.laxis = laxis

        self.x0002 = x0002
        self.fxi = fxi

    def properties(self):
        """ Print properties of your simulation"""

        # check supported geometries
        if self.ig != 1 and self.ig != 2:
            print("ERROR(Properties.py): " + self.errorGeometry(self.ig))
            sys.exit()

        ##############
        # PROPERTIES #
        ##############

        laxis = self.laxis
        xbl = self.xbl
        xbr = self.xbr

        tavg = self.tavg

        # get grid
        xzn0 = self.xzn0
        yzn0 = self.yzn0
        zzn0 = self.zzn0

        xznl = self.xznl
        xznr = self.xznr

        nx = self.nx
        ny = self.ny
        nz = self.nz

        # get inner and outer boundary of computational domain  
        xzn0in = self.xzn0[0]
        xzn0out = self.xzn0[self.nx - 1]

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
        idxl = 0
        idxr = self.nx - 1

        # override
        if laxis == 1:
            idxl, idxr = self.idx_bndry(xbl, xbr)
        if laxis == 2:
            idxl, idxr = self.idx_bndry(xbl, xbr)

        # Get rid of the numerical mess at inner boundary 
        diss[0:idxl] = 0.
        # Get rid of the numerical mess at outer boundary 
        diss[idxr:self.nx] = 0.

        self.nabla[0:idxl] = 0.
        self.nabla[idxr:self.nx] = 0.

        self.nabla_ad[0:idxl] = 0.
        self.nabla_ad[idxr:self.nx] = 0.

        self.fxi[0:idxl] = 0.
        self.fxi[idxr:self.nx] = 0.

        if self.plabel == "ccptwo":
            fxi_max = self.fxi.max()
            ind = np.where((np.abs(self.fxi) > 0.02 * fxi_max))[0]

            xzn0inc = xzn0[ind[0]]
            xzn0outc = xzn0[ind[-1]]
        else:
            #diss_max = diss.max()
            #ind = np.where((diss > 0.02 * diss_max))[0]

            #ind = np.where((self.nabla > self.nabla_ad))[0] # superadiabatic region

            fxi_max = self.fxi.max()
            ind = np.where((np.abs(self.fxi) > 0.1 * fxi_max))[0]

            xzn0inc = xzn0[ind[0]]
            xzn0outc = xzn0[ind[-1]]

        ibot = ind[0]
        itop = ind[-1]

        print(ibot,itop)

        lc = xzn0outc - xzn0inc

        # Reynolds number
        nc = itop - ibot
        Re = nc ** (4. / 3.)

        Vol = np.zeros(nx)
        # handle volume for different geometries
        if self.ig == 1 and self.nsdim == 3:
            surface = (yzn0[-1] - yzn0[0]) * (zzn0[-1] - zzn0[0])
            Vol = surface * (xznr - xznl)
        elif self.ig == 1 and self.nsdim == 2:
            surface = (yzn0[-1] - yzn0[0])*(yzn0[-1] - yzn0[0]) # mock for 2D
            Vol = surface * (xznr - xznl)
        elif self.ig == 2:
            Vol = 4. / 3. * np.pi * (xznr ** 3 - xznl ** 3)

        # Calculate full dissipation rate and timescale
        TKEsum = (dd * tke * Vol)[ind].sum()
        epsD = abs((diss * Vol)[ind].sum())
        tD = TKEsum / epsD

        # RMS velocities
        M = (dd * Vol)[ind].sum()
        urms = np.sqrt(2. * TKEsum / M)

        # Turnover timescale
        tc = 2. * (xzn0outc - xzn0inc) / urms

        # Dissipation length-scale
        ld = M * urms ** 3. / epsD

        # Total nuclear luminosity
        tenuc = ((dd * (enuc1 + enuc2)) * Vol).sum()

        # Pturb over Pgas 
        gamma1 = self.gamma1
        cs2 = (gamma1 * pp) / dd
        ur2 = uxux
        pturb_o_pgas = (gamma1 * ur2 / cs2)[ind].mean()

        # Mach number
        mach2 = uxux / cs2
        mach = mach2 ** 0.5

        machMax = mach[ind].max()
        machMean = mach[ind].mean()

        # Calculate size of convection zone in pressure scale heights
        # hp = -pp / self.Grad(pp, xzn0)
        pbot = pp[ibot]
        lcz_vs_hp = np.log(pbot / pp[ibot:itop])
        cnvz_in_hp = lcz_vs_hp[itop - ibot - 1]

        print('#----------------------------------------------------#')
        print('Datafile with space-time averages: ', self.filename)
        print('Central time (in s): ', round(self.timec, 1))
        print('Averaging windows (in s): ', tavg.item(0))
        print('Time range (in s from-to): ', round(self.trange[0], 1), round(self.trange[1], 1))

        print '---------------'
        print 'Resolution: %i' % self.nx, self.ny, self.nz
        print 'Radial size of computational domain (in cm): %.2e %.2e' % (xzn0in, xzn0out)
        print 'Radial size of convection zone (in cm):  %.2e %.2e' % (xzn0inc, xzn0outc)
        if laxis != 0:
            print 'Extent of convection zone (in Hp): %f' % cnvz_in_hp
        print 'Averaging time window (in s): %f' % tavg
        print 'RMS velocities in convection zone (in cm/s):  %.2e' % urms
        print 'Convective turnover timescale (in s)  %.2e' % tc
        print 'P_turb o P_gas %.2e' % pturb_o_pgas
        print 'Mach number Max %.2e' % machMax
        print 'Mach number Mean %.2e' % machMean
        print 'Dissipation length scale (in cm): %.2e' % ld
        print 'Total nuclear luminosity (in erg/s): %.2e' % tenuc
        print 'Rate of TKE dissipation (in erg/s): %.2e' % epsD
        print 'Dissipation timescale for TKE (in s): %f' % tD
        print 'Reynolds number: %i' % Re
        # print 'Dissipation timescale for radial TKE (in s): %f' % tD_rad
        # print 'Dissipation timescale for horizontal TKE (in s): %f' % tD_hor

        uconv = (2. * tke) ** 0.5
        if lc != 0.:
            kolm_tke_diss_rate = (uconv ** 3) / lc
            tauL = tke / kolm_tke_diss_rate
        else:
            print('ERROR: Estimated size of convection zone is 0')
            kolm_tke_diss_rate = 99999999999.
            tauL = 9999999999.
            # sys.exit()

        if self.plabel == "ccptwo" or self.plabel == "ccpone":
            # ccp project - get averaged X in bottom 2/3 of convection zone (approx. 4-8e8cm)
            indCCP = np.where((xzn0 < 6.66e8))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 8.0e8) & (xzn0 > 4.5e8))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        elif self.plabel == "oburn":
            indCCP = np.where((xzn0 < 8.1e8) & (xzn0 > 4.55e8))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 8.1e8) & (xzn0 > 4.55e8))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        elif self.plabel == "dcf":
            indCCP = np.where((xzn0 < 2.e9))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 5.e9) & (xzn0 > 2.e9))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        elif self.plabel == "neshell":
            indCCP = np.where((xzn0 < 3.85e8) & (xzn0 > 3.6e8))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 3.85e8) & (xzn0 > 3.6e8))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        elif self.plabel == "heflash":
            indCCP = np.where((xzn0 < 8.e8) & (xzn0 > 5.e8))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 8.e8) & (xzn0 > 5.e8))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        elif self.plabel == "thpulse":
            indCCP = np.where((xzn0 < 1.2e9) & (xzn0 > 8.e8))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 1.2e9) & (xzn0 > 8.e8))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        elif self.plabel == "cflash":
            indCCP = np.where((xzn0 < 7.5e8) & (xzn0 > 5.e8))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 7.5e8) & (xzn0 > 5.e8))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        elif self.plabel == "heflash":
            indCCP = np.where((xzn0 < 8.e8) & (xzn0 > 5.5e8))[0]
            x0002mean_cnvz = np.mean(self.x0002[indCCP])

            indRES = np.where((xzn0 < 8.e8) & (xzn0 > 5.5e8))[0]
            # residual from continuity equation
            resCont = np.abs(self.minus_resContEquation)
            resContMax = np.max(resCont[indRES])
            resContMean = np.mean(resCont[indRES])

            # residual from total energy equation
            resTee = np.abs(self.minus_resTeEquation)
            resTeeMax = np.max(resTee[indRES])
            resTeeMean = np.mean(resTee[indRES])
        else:
            print("ERROR(Properties.py): " + self.errorProject(self.plabel))
            sys.exit()

        ig = self.ig

        p = {'tauL': tauL, 'kolm_tke_diss_rate': kolm_tke_diss_rate, 'tke_diss': diss, 'tavg': self.tavg,
             'tke': tke, 'lc': lc, 'uconv': uconv, 'xzn0inc': xzn0inc, 'xzn0outc': xzn0outc, 'cnvz_in_hp': cnvz_in_hp,
             'tc': tc, 'nx': nx, 'ny': ny, 'nz': nz, 'machMax': machMax, 'machMean': machMean, 'xzn0': xzn0,
             'ig': ig, 'dd': dd, 'x0002mean_cnvz': x0002mean_cnvz, 'pturb_o_pgas': pturb_o_pgas, 'TKEsum': TKEsum,
             'epsD': epsD, 'tD': tD, 'tenuc': tenuc, 'urms': urms, 'resContMax': resContMax, 'resContMean': resContMean,
             'resTeeMax': resTeeMax, 'resTeeMean': resTeeMean, 'xznl': xznl, 'xznr': xznr}

        return {'tauL': p['tauL'], 'kolm_tke_diss_rate': p['kolm_tke_diss_rate'], 'tavg': p['tavg'],
                'tke_diss': p['tke_diss'], 'tke': p['tke'], 'lc': p['lc'], 'dd': p['dd'],
                'uconv': p['uconv'], 'xzn0inc': p['xzn0inc'], 'xzn0outc': p['xzn0outc'],
                'tc': p['tc'], 'nx': p['nx'], 'ny': p['ny'], 'nz': p['nz'], 'machMax': p['machMax'],
                'machMean': p['machMean'], 'xzn0': p['xzn0'], 'ig': p['ig'], 'TKEsum': p['TKEsum'],
                'x0002mean_cnvz': p['x0002mean_cnvz'], 'pturb_o_pgas': p['pturb_o_pgas'], 'cnvz_in_hp': p['cnvz_in_hp'],
                'epsD': p['epsD'], 'tD': p['tD'], 'tenuc': p['tenuc'], 'resContMean': p['resContMean'],
                'resContMax': p['resContMax'], 'resTeeMax': p['resTeeMax'], 'resTeeMean': p['resTeeMean'],
                'xznl': p['xznl'], 'xznr': p['xznr'], 'urms': p['urms']}
