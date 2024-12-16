import numpy as np
import sys

from UTILS.Calculus import Calculus
from UTILS.Errors import Errors
from UTILS.Tools import Tools
from EQUATIONS.TurbulentKineticEnergyCalculation import TurbulentKineticEnergyCalculation

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class Properties(Calculus, Tools, Errors, object):

    def __init__(self, params):
        super(Properties, self).__init__(params.getForProp('prop')['ig'])

        # get input parameters
        filename = params.getForProp('prop')['eht_data']
        plabel = params.getForProp('prop')['plabel']
        code = params.getForProp('prop')['code']
        ig = params.getForProp('prop')['ig']
        nsdim = params.getForProp('prop')['nsdim']
        ieos = params.getForProp('prop')['ieos']
        intc = params.getForProp('prop')['intc']
        laxis = params.getForProp('prop')['laxis']
        xbl = params.getForProp('prop')['xbl']
        xbr = params.getForProp('prop')['xbr']


        # load data to structured array
        eht = np.load(filename, allow_pickle=True, encoding='latin1')

        timec = self.getRAdata(eht, 'timec')[intc]
        tavg = self.getRAdata(eht, 'tavg')
        trange = self.getRAdata(eht, 'trange')

        # load grid
        nx = self.getRAdata(eht, 'nx')
        #ny = self.getRAdata(eht, 'ny')
        #nz = self.getRAdata(eht, 'nz')
        ny = nx
        nz = nx


        xzn0 = self.getRAdata(eht, 'xzn0')

        #xznl = self.getRAdata(eht, 'xznl')
        #xznr = self.getRAdata(eht, 'xznr')
        #yzn0 = self.getRAdata(eht, 'yzn0')
        #zzn0 = self.getRAdata(eht, 'zzn0')

        #xznl = self.getRAdata(eht, 'xzn0')
        #xznr = np.roll(xznl,1)

        dx = (xzn0[-1]-xzn0[0])/nx
        xznl = xzn0 - dx/2.
        xznr = xzn0 + dx/2.

        #yzn0 = self.getRAdata(eht, 'xzn0')
        #zzn0 = self.getRAdata(eht, 'xzn0')

        yzn0 = np.linspace(0.,2.,nx)
        zzn0 = np.linspace(0.,2.,nx)

        # print('[WARNING Properties.py] - some grid properties hardcoded. This needs to be fixed first in rawdata class')

        dd = self.getRAdata(eht, 'dd')[intc]
        pp = self.getRAdata(eht, 'pp')[intc]

        ux = self.getRAdata(eht, 'ux')[intc]
        uy = self.getRAdata(eht, 'uy')[intc]
        uz = self.getRAdata(eht, 'uz')[intc]

        uxux = self.getRAdata(eht, 'uxux')[intc]
        uyuy = self.getRAdata(eht, 'uyuy')[intc]
        uzuz = self.getRAdata(eht, 'uzuz')[intc]

        if plabel == 'ccptwo':
            ddux = self.getRAdata(eht, 'ddux')[intc]
            ddxi = self.getRAdata(eht, 'ddx0001')[intc]
            ddxiux = self.getRAdata(eht, 'ddx0001ux')[intc]
            fxi = ddxiux - ddxi * ddux / dd
        else:
            print("ERROR(Properties.py): Project " + plabel + " not supported.")
            sys.exit()

        pp = self.getRAdata(eht, 'pp')[intc]


        # for ccp project
        if plabel == 'ccptwo':
            x0002 = self.getRAdata(eht, 'x0002')[intc]
        else:
            x0002 = np.zeros(nx)

        ####################################################################

        # instantiate turbulent kinetic energy object
        tkeF = TurbulentKineticEnergyCalculation(filename, ig, intc)

        # load fields
        tkefields = tkeF.getTKEfield()

        # get turbulent kinetic energy
        self.tke = tkefields['tke']
        self.tkex = tkefields['tkex']
        self.tkey = tkefields['tkey']
        self.tkez = tkefields['tkez']

        # get turbulent kinetic energy dissipation
        self.minus_resTkeEquation = tkefields['minus_resTkeEquation']
        self.minus_resTkeEquationX = tkefields['minus_resTkeEquationX']
        self.minus_resTkeEquationY = tkefields['minus_resTkeEquationY']
        self.minus_resTkeEquationZ = tkefields['minus_resTkeEquationZ']

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
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.uxux = uxux
        self.uyuy = uyuy
        self.uzuz = uzuz


        self.filename = filename
        self.plabel = plabel
        self.code = code
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
        ux = self.ux
        uy = self.uy
        uz = self.uz

        # load uxsq		
        uxux = self.uxux
        uyuy = self.uyuy
        uzuz = self.uzuz

        # load TKE
        tke = self.tke
        tkever = self.tkex
        tkehor = self.tkey + self.tkez

        # load TKE dissipation
        diss = abs(self.minus_resTkeEquation)
        dissver = abs(self.minus_resTkeEquationX)
        disshor = abs(self.minus_resTkeEquationY + self.minus_resTkeEquationZ)

        # calculate INDICES for grid boundaries 
        idxl = 0
        idxr = self.nx - 1

        # override
        if laxis == 1:
            idxl, idxr = self.idx_bndry(xbl, xbr, xzn0)
        if laxis == 2:
            idxl, idxr = self.idx_bndry(xbl, xbr, xzn0)

        # Get rid of the numerical mess at inner boundary 
        diss[0:idxl] = 0.
        # Get rid of the numerical mess at outer boundary 
        diss[idxr:self.nx] = 0.

        self.fxi[0:idxl] = 0.
        self.fxi[idxr:self.nx] = 0.

        if self.plabel == "ccptwo":
            fxi_max = self.fxi.max()
            ind = np.where((np.abs(self.fxi) > 0.02 * fxi_max))[0]

            xzn0inc = xzn0[ind[0]]
            xzn0outc = xzn0[ind[-1]]
        else:
            # diss_max = diss.max()
            # ind = np.where((diss > 0.02 * diss_max))[0]

            # ind = np.where((self.nabla > self.nabla_ad))[0] # superadiabatic region

            fxi_max = np.max(np.abs(self.fxi))
            ind = np.where((np.abs(self.fxi) > 0.003 * fxi_max))[0]

            xzn0inc = xzn0[ind[0]]
            xzn0outc = xzn0[ind[-1]]

        ibot = ind[0]
        itop = ind[-1]

        lc = xzn0outc - xzn0inc

        # Reynolds number
        nc = itop - ibot
        Re = nc ** (4. / 3.)

        Vol = np.zeros(nx)
        # handle volume for different geometries
        if self.ig == 1 and self.nsdim == 3:
            surface = (yzn0[-1] - yzn0[0]) * (zzn0[-1] - zzn0[0])
            Vol = surface * (xznr - xznl)
            #print(surface)
            #print(xznr-xznl)
        elif self.ig == 1 and self.nsdim == 2:
            surface = (yzn0[-1] - yzn0[0]) * (yzn0[-1] - yzn0[0])  # mock for 2D
            Vol = surface * (xznr - xznl)
        elif self.ig == 2:
            Vol = 4. / 3. * np.pi * (xznr ** 3 - xznl ** 3)

        # Calculate full dissipation rate and timescale
        TKEsum = (dd * tke * Vol)[ind].sum()
        epsD = abs((diss * Vol)[ind].sum())
        tD = TKEsum / epsD
        #print(Vol)
        #print(TKEsum, epsD, tD)

        TKEversum = (dd * tkever * Vol)[ind].sum()
        epsDver = abs((dissver * Vol)[ind].sum())
        tDver = TKEversum / epsDver

        TKEhorsum = (dd * tkehor * Vol)[ind].sum()
        epsDhor = abs((disshor * Vol)[ind].sum())
        tDhor = TKEhorsum / epsDhor

        # RMS velocities
        M = (dd * Vol)[ind].sum()
        urms = np.sqrt(2. * TKEsum / M)

        # Turnover timescale
        tc = 2. * (xzn0outc - xzn0inc) / urms

        # Dissipation length-scale
        ld = M * urms ** 3. / epsD

        # Calculate size of convection zone in pressure scale heights
        hp = -pp / self.Grad(pp, xzn0)
        pbot = pp[ibot]
        lcz_vs_hp = np.log(pbot / pp[ibot:itop])
        cnvz_in_hp = lcz_vs_hp[itop - ibot - 1]

        # calculate width of overshooting regions in Hp
        # hp = -pp / self.Grad(pp, xzn0)
        # pbot = pp[ibot]
        # tmp = np.log(pbot / pp[ibot:ibot_super_ad])
        # ov_in_hp = tmp[ibot_super_ad - ibot - 1]

        # pbot = pp[itop_super_ad]
        # tmp = np.log(pbot / pp[itop_super_ad:itop])
        # ov_out_hp = tmp[itop - itop_super_ad - 1]

        ov_in_hp = 0.
        ov_out_hp = 0.

        #print('#----------------------------------------------------#')
        #print('Datafile with space-time averages: ', self.filename)
        #print('Central time (in s): ', round(self.timec, 1))
        #print('Averaging windows (in s): ', tavg.item(0))
        #print('Time range (in s from-to): ', round(self.trange[0], 1), round(self.trange[1], 1))

        #print('---------------')
        #print('Resolution: %i' % self.nx, self.ny, self.nz)
        #print('Radial size of computational domain (in cm): %.2e %.2e' % (xzn0in, xzn0out))
        #print('Radial size of convection zone (in cm):  %.2e %.2e' % (xzn0inc, xzn0outc))
        #if laxis != 0:
        #    print('Extent of convection zone (in Hp): %f' % cnvz_in_hp)
        #else:
        #    cnvz_in_hp = 0.
        #print('Overshooting at inner/outer convection boundary (in Hp): ', round(ov_in_hp, 2), round(ov_out_hp, 2))
        #print('Averaging time window (in s): %f' % tavg)
        #print('RMS velocities in convection zone (in cm/s):  %.2e' % urms)
        #print('Convective turnover timescale (in s)  %.2e' % tc)
        #print('P_turb o P_gas %.2e' % pturb_o_pgas)
        #print('Mach number Max (using uxux) %.2e' % machMax_1)
        #print('Mach number Mean (using uxux) %.2e' % machMean_1)
        #print('Mach number Max (using uu) %.2e' % machMax_2)
        #print('Mach number Mean (using uu) %.2e' % machMean_2)
        #print('Dissipation length scale (in cm): %.2e' % ld)
        #print('Total nuclear luminosity (in erg/s): %.2e' % tenuc)
        #print('Rate of TKE dissipation (in erg/s): %.2e' % epsD)
        #print('Dissipation timescale for TKE (in s): %f' % tD)
        #print('Dissipation timescale for TKE vertical (in s): %f' % tDver)
        #print('Dissipation timescale for TKE horizontal (in s): %f' % tDhor)
        #print('Reynolds number: %i' % Re)
        # print ('Dissipation timescale for radial TKE (in s): %f' % tD_rad)
        # print ('Dissipation timescale for horizontal TKE (in s): %f' % tD_hor)

        uconv = (2. * tke) ** 0.5

        uyrms = np.sqrt(uyuy - uy * uy)
        uzrms = np.sqrt(uzuz - uz * uz)

        uiso = np.sqrt((3. / 2.) * (uyrms ** 2 + uzrms ** 2))
        if lc != 0.:
            # kolm_tke_diss_rate = (uconv ** 3) / lc
            # kolm_tke_diss_rate = (uiso ** 3) / ld
            kolm_tke_diss_rate = (uconv ** 3) / ld
            tauL = tke / kolm_tke_diss_rate
        else:
            print('ERROR: Estimated size of convection zone is 0')
            kolm_tke_diss_rate = 99999999999.
            tauL = 9999999999.
            # sys.exit()

        ig = self.ig

        trange = [round(self.trange[0], 1), round(self.trange[1], 1)]

        return {'tauL': tauL, 'kolm_tke_diss_rate': kolm_tke_diss_rate, 'tke_diss': diss, 'tavg': tavg.item(0),
                'tke': tke, 'lc': lc, 'uconv': uconv, 'xzn0inc': xzn0inc, 'xzn0outc': xzn0outc,
                'xzn0in': xzn0in, 'xzn0out': xzn0out, 'cnvz_in_hp': cnvz_in_hp,
                'tc': round(tc,1), 'nx': nx.item(0), 'ny': ny.item(0), 'nz': nz.item(0),
                'xzn0': xzn0,
                'ig': ig, 'dd': dd, 'TKEsum': TKEsum,
                'epsD': epsD, 'tD': round(tD,1),'urms': round(urms,4), 'xznl': xznl, 'xznr': xznr,
                'filename': self.filename, 'timec': round(self.timec, 1),
                'timerange_beg': round(self.trange[0], 1),'timerange_end': round(self.trange[1], 1),'Re': round(Re),
                'tavg_to': round(tavg.item(0)/tc), 'trange': trange, 'plabel': self.plabel, 'code': self.code}


