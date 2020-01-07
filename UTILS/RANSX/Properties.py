import numpy as np
import sys
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al
import EQUATIONS.TurbulentKineticEnergyCalculation as tkeCalc


# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Properties(calc.Calculus, al.SetAxisLimit, object):

    def __init__(self, filename, ig, ieos, intc, laxis, xbl, xbr):
        super(Properties, self).__init__(ig)

        # load data to structured array
        eht = np.load(filename)

        timec = eht.item().get('timec')[intc]
        tavg = np.asarray(eht.item().get('tavg'))
        trange = np.asarray(eht.item().get('trange'))

        # load grid
        nx = np.asarray(eht.item().get('nx'))
        ny = np.asarray(eht.item().get('ny'))
        nz = np.asarray(eht.item().get('nz'))

        xzn0 = np.asarray(eht.item().get('xzn0'))
        xznl = np.asarray(eht.item().get('xznl'))
        xznr = np.asarray(eht.item().get('xznr'))

        yzn0 = np.asarray(eht.item().get('yzn0'))
        zzn0 = np.asarray(eht.item().get('zzn0'))

        enuc1 = np.asarray(eht.item().get('enuc1')[intc])
        enuc2 = np.asarray(eht.item().get('enuc2')[intc])

        dd = np.asarray(eht.item().get('dd')[intc])
        pp = np.asarray(eht.item().get('pp')[intc])
        uxux = np.asarray(eht.item().get('uxux')[intc])
        gamma1 = np.asarray(eht.item().get('gamma1')[intc])

        # for ccp project
        x0002 = np.asarray(eht.item().get('x0002')[intc])

        # instantiate turbulent kinetic energy object
        tkeF = tkeCalc.TurbulentKineticEnergyCalculation(filename, ig, ieos, intc)

        # load fields
        tkefields = tkeF.getTKEfield()

        # get turbulent kinetic energy
        self.tke = tkefields['tke']

        # get turbulent kinetic energy dissipation
        self.minus_resTkeEquation = tkefields['minus_resTkeEquation']

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
        self.tavg = tavg
        self.timec = timec
        self.trange = trange
        self.ig = ig
        self.laxis = laxis

        self.x0002 = x0002

    def properties(self):
        """ Print properties of your simulation"""
        """ Share Turbulent Kinetic Energy Equations Terms  """

        laxis = self.laxis
        xbl = self.xbl
        xbr = self.xbr

        ##############
        # PROPERTIES #
        ##############

        # load grid
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
        if laxis == 0:
            idxl = 0
            idxr = self.nx - 1
        if laxis == 1:
            idxl, idxr = self.idx_bndry(xbl, xbr)
        if laxis == 2:
            idxl, idxr = self.idx_bndry(xbl, xbr)

        # Get rid of the numerical mess at inner boundary 
        diss[0:idxl] = 0.
        # Get rid of the numerical mess at outer boundary 
        diss[idxr:self.nx] = 0.

        diss_max = diss.max()
        ind = np.where((diss > 0.02 * diss_max))[0]
        # ind = np.where( (diss > 0.015*diss_max) )[0]

        xzn0inc = xzn0[ind[0]]
        xzn0outc = xzn0[ind[-1]]

        ibot = ind[0]
        itop = ind[-1]

        lc = xzn0outc - xzn0inc

        # Reynolds number
        nc = itop - ibot
        Re = nc ** (4. / 3.)

        # handle volume for different geometries
        if (self.ig == 1):
            surface = (yzn0[-1] - yzn0[0]) * (zzn0[-1] - zzn0[0])
            Vol = surface * (xznr - xznl)
        elif (self.ig == 2):
            Vol = 4. / 3. * np.pi * (xznr ** 3 - xznl ** 3)
        else:
            print(
                "ERROR (Properties.py): geometry not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ...")
            sys.exit()

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
        hp = -pp / self.Grad(pp, xzn0)
        pbot = pp[ibot]
        lcz_vs_hp = np.log(pbot / pp[ibot:itop])

        print('#----------------------------------------------------#')
        print('Datafile with space-time averages: ', self.filename)
        print('Central time (in s): ', round(self.timec, 1))
        print('Averaging windows (in s): ', self.tavg.item(0))
        print('Time range (in s from-to): ', round(self.trange[0], 1), round(self.trange[1], 1))

        print '---------------'
        print 'Resolution: %i' % self.nx, self.ny, self.nz
        print 'Radial size of computational domain (in cm): %.2e %.2e' % (xzn0in, xzn0out)
        print 'Radial size of convection zone (in cm):  %.2e %.2e' % (xzn0inc, xzn0outc)
        if laxis != 0: print 'Extent of convection zone (in Hp): %f' % lcz_vs_hp[itop - ibot - 1]
        print 'Averaging time window (in s): %f' % self.tavg
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

        # ccp project - get averaged X in bottom 2/3 of convection zone (approx. 4-8e8cm)
        ind = np.where((xzn0 < 6.66e8))[0]
        x0002mean_cnvz = np.mean(self.x0002[ind])

        """ Share Turbulent Kinetic Energy Equations Terms  """

        ig = self.ig

        p = {'tauL': tauL, 'kolm_tke_diss_rate': kolm_tke_diss_rate, 'tke_diss': diss,
                'tke': tke, 'lc': lc, 'uconv': uconv, 'xzn0inc': xzn0inc, 'xzn0outc': xzn0outc,
                'tc': tc, 'nx': nx, 'ny': ny, 'nz': nz, 'machMax': machMax, 'machMean': machMean, 'xzn0': xzn0,
                'ig': ig, 'dd': dd, 'x0002mean_cnvz': x0002mean_cnvz, 'pturb_o_pgas': pturb_o_pgas, 'TKEsum': TKEsum,
                'epsD': epsD, 'tD': tD, 'tc': tc, 'tenuc': tenuc, 'xznl': xznl, 'xznr': xznr}

        return {'tauL': p['tauL'], 'kolm_tke_diss_rate': p['kolm_tke_diss_rate'],
                'tke_diss': p['tke_diss'], 'tke': p['tke'], 'lc': p['lc'], 'dd': p['dd'],
                'uconv': p['uconv'], 'xzn0inc': p['xzn0inc'], 'xzn0outc': p['xzn0outc'],
                'tc': p['tc'], 'nx': p['nx'], 'ny': p['ny'], 'nz': p['nz'], 'machMax': p['machMax'],
                'machMean': p['machMean'], 'xzn0': p['xzn0'], 'ig': p['ig'], 'TKEsum': p['TKEsum'],
                'x0002mean_cnvz': p['x0002mean_cnvz'], 'pturb_o_pgas': p['pturb_o_pgas'],
                'epsD': p['epsD'], 'tD': p['tD'], 'tc': p['tc'], 'tenuc': p['tenuc'],
                'xznl': p['xznl'], 'xznr': p['xznr']}
