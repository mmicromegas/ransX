from __future__ import print_function

import os
import sys
import re
import glob
import mmap
import numpy as np
import hashlib
import matplotlib.pyplot as plt


import numpy as np
# import slhoutput as slhout
import collections


def get_haxes(ndim, vaxis):
   if vaxis >= ndim:
      print('Error: vaxis = {:d} does not exist with ndim = {:d}.'.\
            format(vaxis, ndim))
      return None

   if ndim not in (2, 3):
      print('Error: ndim = {:d} not supported.'.format(ndim))
      return None

   haxes = list(range(ndim))
   haxes.remove(vaxis)

   return tuple(haxes)

def get_reshape_vec(res, vaxis):
   ndim = len(res)
   if vaxis >= ndim:
      print('Error: vaxis = {:d} does not exist with ndim = {:d}.'.\
            format(vaxis, ndim))
      return None

   reshape_vec= [1,]*ndim
   reshape_vec[vaxis] = res[vaxis]

   return tuple(reshape_vec)

class Analysis(object):

    def __init__(self, **kwargs):
        self.gamma = kwargs.pop('gamma', 1.4)  # diatomic gas
        self.mu = kwargs.pop('mu', 28.96)  # g / mol for air (from Wolfram Alpha)
        self.R = 8.314472e7  # erg / (mol K)
        self.u0 = kwargs.pop('u0', 1e4)  # cm / s
        self.k = kwargs.pop('k', 1e-2)  # 1 / cm

    @property
    def tstar(self):
        return self.time * self.k * self.u0

    @property
    def cspeed(self):
        return np.sqrt(self.gamma * self.pres / self.rho)

    @property
    def mach(self):
        return self.vel / self.cspeed

    @property
    def temp(self):
        try:
            return self._temp
        except AttributeError:
            return self.pres / (self.rho * self.R) * self.mu

    @temp.setter
    def temp(self, val):
        self._temp = val

    @temp.deleter
    def temp(self):
        del self._temp

    @property
    def velgradtensor(self):
        """velocity gradient tensor"""
        _, dx, dy, dz = np.gradient(self.vel, 1, *self.dx)
        return np.concatenate((dx[:, None], dy[:, None], dz[:, None]), axis=1)

    @property
    def vol(self):
        return np.ones(self.rho.shape)*np.prod(self.dx)

    @property
    def mass(self):
        return self.vol * self.rho

    @property
    def vorticity(self):
        vort = curl(self.vel, 1, *self.dx)
        return np.sum(vort**2, axis=0)**0.5

    @property
    def enstrophy(self):
        return 0.5 * (self.vorticity**2).sum(axis=0)

    @property
    def absvel(self):
        return np.sqrt((self.vel**2).sum(axis=0))

    @property
    def ekin(self):
        return 0.5 * self.rho * (self.vel**2).sum(axis=0)

    @property
    def A(self):
        return self.pres/self.rho**(5./3.)

    @property
    def Kstar(self):
        return 0.5 * (self.vel**2).sum(axis=0, dtype=np.float64).mean() / self.u0**2

    @property
    def enthalpy(self):
        return self.eps + self.pres

    def enthalpy_flux(self,vaxis):
        enthalpy = self.enthalpy
        vol = self.vol
        vel = self.vel[vaxis]
        mass = self.mass
        enthalpy_flux = self.get_mean(enthalpy*vel, vol, vaxis) - \
                        self.get_mean(enthalpy, vol, vaxis)*\
                        self.get_mean(vel, mass, vaxis)
        return enthalpy_flux

    def kinetic_energy_flux(self,vaxis):
        ek = self.ekin
        vol = self.vol
        vel = self.vel[vaxis]
        mass = self.mass
        ek_flux = self.get_mean(ek*vel, vol, vaxis) - \
                  self.get_mean(ek, vol, vaxis)*\
                  self.get_mean(vel, mass, vaxis) 
        return ek_flux

    def filling_factor_downflow(self, vaxis):
        vel = self.vel[vaxis]
        haxes = get_haxes(len(vel.shape), vaxis)
        ffd = np.mean(vel < 0., axis=haxes)
        return ffd

    @property
    def Omegastar(self):
        return self.enstrophy.mean() / (self.k * self.u0)**2

    @property
    def lambda2(self):
        """second largest eigenvalue of S^2 + Omega^2,
        where S is the symmetric and Omega the anti-symmetric part of the velocity gradient tensor
        see Jeong and Hussain (1995)"""
        tens = self.velgradtensor
        S = 0.5 * (tens + tens.swapaxes(0,1))
        O = 0.5 * (tens - tens.swapaxes(0,1))
        S = np.einsum('il...,lj...->ij...', S, S)
        O = np.einsum('il...,lj...->ij...', O, O)
        tens = S + O
        tens = tens.reshape([9] + list(tens.shape[-3:]))

        def eigvals1d(a):
            a = np.asanyarray(a)
            return np.sort(np.linalg.eigvals(a.reshape((3, 3))))[-2]

        return np.apply_along_axis(eigvals1d, 0, tens)

    def spectrum(self, nbins=None):
        v = self.vel
        ndim = self.vel.ndim-1
        ftv = np.array([np.fft.fftn(v[i], axes=range(ndim)) for i in range(ndim)])

        # wave numbers
        kn = np.array([2.0 * np.pi * np.fft.fftfreq(v.shape[i+1], d=self.dx[i]) for i in range(ndim)])
        if (ndim == 3):
            k = np.sqrt(kn[0][:, None, None]**2 + kn[1][None, :, None]**2 + kn[2][None, None, :]**2)
        elif(ndim ==2):
            k = np.sqrt(kn[0][:, None]**2 + kn[1][None, :]**2 )

        # bins
        if nbins is None:
            nbins = v.shape[-1] // 2
        minbin = 2.0 * np.pi / max(self.dx[i] * self.vel.shape[i+1] for i in range(ndim))
        k1d_edges = np.linspace(minbin, k.max(), nbins + 1.0)

        ekin3d = 0.5 * (np.abs(ftv)**2).sum(axis=0)

        ekin1d, k1d_edges = np.histogram(k, bins=k1d_edges, weights=ekin3d)
        k1d = 0.5 * (k1d_edges[:-1] + k1d_edges[1:])

        return k1d, ekin1d

    def contour(self, data=None, **kwargs):
        import mayavi.mlab as mlab
        if data is None:
            data = np.sqrt((self.mach**2).sum(axis=0))
        sl = [slice(0, 2.0*np.pi, 1j*data.shape[i]) for i in range(3)]
        x, y, z = np.mgrid[sl]

        mlab.contour3d(x, y, z, data, **kwargs)
        mlab.colorbar()

    def get_mean(self, quantity, weight, vaxis):
       haxes = get_haxes(len(quantity.shape), vaxis)
       sum_weight = np.sum(weight, axis=haxes)
       mean = np.sum(quantity*weight, axis=haxes)/sum_weight

       return mean

    def get_stdev(self, quantity, weight, vaxis):
       haxes = get_haxes(len(quantity.shape), vaxis)
       sum_weight = np.sum(weight, axis=haxes)
       mean = np.sum(quantity*weight, axis=haxes)/sum_weight

       reshape_vec = get_reshape_vec(quantity.shape, vaxis)
       tmp = np.reshape(mean, reshape_vec)
       stdev = (np.sum((quantity - tmp)**2*weight, axis=haxes)/sum_weight)**0.5

       return stdev

    def write_Rprof(self, dump_num, rprof_name):
       vaxis = 1
       haxes = get_haxes(3, vaxis)

       cell_volume = self.vol
       cell_mass = self.rho*self.vol

       vars = collections.OrderedDict()
       vars['RHO'] = {'expr':lambda g: g.rho, 'weighting':'v', 'stats':'full'}
       vars['P'] = {'expr':lambda g: g.pres, 'weighting':'v', 'stats':'full'}
       vars['TEMP'] = {'expr':lambda g: g.temp, 'weighting':'v', 'stats':'full'}
       vars['A'] = {'expr':lambda g: g.A, 'weighting':'v', 'stats':'full'}
       vars['X1'] = {'expr':lambda g: g.xnuc[1], 'weighting':'m', 'stats':'full'}
       vars['V'] = {'expr':lambda g: g.absvel, 'weighting':'m', 'stats':'full'}
       vars['VX'] = {'expr':lambda g: g.vel[0], 'weighting':'m', 'stats':'full'}
       vars['VY'] = {'expr':lambda g: g.vel[1], 'weighting':'m', 'stats':'full'}
       vars['VZ'] = {'expr':lambda g: g.vel[2], 'weighting':'m', 'stats':'full'}
       vars['|VY|'] = {'expr':lambda g: np.abs(g.vel[1]), 'weighting':'m', 'stats':'mean'}
       vars['VXZ'] = {'expr':lambda g: (g.vel[haxes[0]]**2 + g.vel[haxes[1]]**2)**0.5, 'weighting':'m', 'stats':'mean'}
       vars['VORT'] = {'expr':lambda g: np.abs(g.vorticity), 'weighting':'m', 'stats':'mean'}

       nx = self.rho.shape[vaxis]
       nbins = nx

       time = self.time
       ir = 1 + np.arange(nbins)
       y = np.mean(self.coords[vaxis], axis=haxes)
       ncols = 1*sum(1 for val in vars.values() if val['stats'] == 'mean') + \
               4*sum(1 for val in vars.values() if val['stats'] == 'full')
       ncols+=4 # FK, FH, FFD, and DISS are computed separately below
       col_names = []
       data_table = np.zeros((nbins, ncols))

       j = 0
       for i, v in enumerate(vars.keys()):
          expr = vars[v]['expr']
          stats = vars[v]['stats']
          weighting=vars[v]['weighting']
          if stats not in ('mean', 'full'):
             print("ERROR: 'stats' must be 'mean' or 'full'.")
             return None

          if weighting == 'v':
             weight = cell_volume
          elif weighting == 'm':
             weight = cell_mass
          else:
             print("ERROR: 'weighting' must be 'v' or 'm'.")
             return None

          # We always want the mean.
          col_names.append(v)
          mean = self.get_mean(expr(self), weight, vaxis)
          data_table[:,j] = mean
          j+=1

          if stats == 'full':
             min = np.min(expr(self), axis=haxes)
             col_names.append('MIN_'+v)
             data_table[:,j] = min
             j+=1

             max = np.max(expr(self), axis=haxes)
             col_names.append('MAX_'+v)
             data_table[:,j] = max
             j+=1

             stdev = self.get_stdev(expr(self), weight, vaxis)
             col_names.append('STDEV_'+v)
             data_table[:,j] = stdev
             j+=1

       col_names.append('FK')
       data_table[:,j] = self.kinetic_energy_flux(vaxis)
       j+=1

       col_names.append('FH')
       data_table[:,j] = self.enthalpy_flux(vaxis)
       j+=1

       col_names.append('FFD')
       data_table[:,j] = self.filling_factor_downflow(vaxis)
       j+=1

       # Dissipation rate to be implemented.
       diss = np.zeros(nbins)
       col_names.append('DISS')
       data_table[:,j] = diss
       j+=1

       try:
          fout = open(rprof_name, "w")
          fout.write('DUMP {:4d}, t = {:.8e}\n'.format(dump_num, time))
          fout.write('Nx = {:d}\n\n\n'.format(nx))

          cols_per_table = 8
          ntables = int(np.ceil(ncols/cols_per_table))
          for i in range(ntables):
             if i < ntables - 1:
                cols_this_table = cols_per_table
             else:
                cols_this_table = ncols - i*cols_per_table
             table = np.zeros((nbins, 2+cols_this_table))
             table[:,0] = ir
             table[:,1] = y
             idx1 = i*cols_per_table
             idx2 = idx1 + cols_this_table

             fmt = 'IR      ' + '{:18s}'*(1+cols_this_table) + '\n\n'
             header = fmt.format('Y', *col_names[idx1:idx2])
             fout.write(header)

             table[:,2:(2+cols_this_table)] = data_table[:,idx1:idx2]
             table = np.flip(table, axis=0)

             fmt = ('%4d',) + ('% .8e',)*(1+cols_this_table)
             np.savetxt(fout, table, fmt=fmt, delimiter='   ')
             fout.write('\n\n')

          fout.close()
       except EnvironmentError:
          print('An environment error has occured!')

    def write_RprofSet(self,dumps, run_id):
       for i, dmp in enumerate(dumps):
          g = slhout.slhgrid(dmp, mode='i')
          rprof_name = '{:s}-{:04d}.rprof'.format(run_id, dmp)
          write_Rprof(dmp, rprof_name, g)

class SLH(Analysis):
    @staticmethod
    def find_files(directory):
        found = []

        for f in glob.iglob(os.path.join(directory, 'grid_n*')):
            m = re.search('grid_n([0-9]{5,})\\.(slh|lhc)$', f)
            if m:
                found.append((int(m.group(1)), f))

        found.sort(key=lambda x: x[0])

        numbers = [f[0] for f in found]
        names = [f[1] for f in found]

        return numbers, names

    def __init__(self, filename, **kwargs):
        import slhoutput
        if isinstance(filename, str):
            kwargs.setdefault('mode', 'f')
        slhgridkwargs = kwargs.copy()
        for k in ['gamma', 'mu', 'u0', 'k']:
            slhgridkwargs.pop(k,None)
        g = slhoutput.slhgrid(filename, **slhgridkwargs)
        self.rho = g.rho()
        self.pres = g.pres()
        self.temp = g.temp()
        self.vel = g.vel()
        self.eps = g.eps()
        self.xnuc = g.xnuc() #this is a recarray, but a simple dict might work as well
        self.coords = g.coords()

        self.dx = g.geometry.dx
        self.time = g.time

        if 'u0' not in kwargs:
            kwargs['u0'] = g.qref_calc.velocity

        super(SLH, self).__init__(**kwargs)

class Flash(Analysis):
    def __init__(self,filename,**kwargs):
        import yt
        from yt.frontends.flash.data_structures import FLASHDataset

        #conversion factors from cgs to code units
        rhofac = 1.820940e6
        pfac = 4.644481e+23
        tempfac = 3.401423e9
        efac = 2.972468e+49
        rfac = 4e8
        tfac = 0.7920256
        velfac = rfac/tfac
        
        if isinstance(filename,FLASHDataset):
            d = filename
        else:
            d = yt.load(filename)
        shape = [complex(0,d.domain_dimensions[i]) for i in range(3)]
        grid = d.r[::shape[0],::shape[1],::shape[2]]

        self.rho = grid['dens'].to_ndarray() / rhofac
        self.pres = grid['pres'].to_ndarray() / pfac
        self.temp = grid['temp'].to_ndarray() / tempfac
        self.vel = np.array([grid['velx'].to_ndarray(),grid['vely'].to_ndarray(),grid['velz'].to_ndarray()])/velfac
        self.eps = grid['eint'].to_ndarray() / efac
        self.xnuc = np.array([grid['conv'].to_ndarray(),grid['stab'].to_ndarray()])
        self.coords = np.array([grid['x'].to_ndarray(),grid['y'].to_ndarray(),grid['z'].to_ndarray()])/rfac

        self.dx = np.array([grid['dx'].to_ndarray()[0,0,0],grid['dy'].to_ndarray()[0,0,0],grid['dz'].to_ndarray()[0,0,0]])/rfac
        self.time = d.current_time.to_ndarray() / tfac

        super(Flash,self).__init__(**kwargs)


#  run analysis.py
#  an = PROMPI('D:\\ransX\\DATA_D\\BINDATA\\ccp_two_layers\\cosma\\ccptwo.r128x128x128.cosma.00797.bindata')
#  an.write_Rprof(1,'test.rprof')

class PROMPI(Analysis):
    def __init__(self, filename, **kwargs):
        import UTILS.PROMPI.PROMPI_data as prd

        # conversion factors from cgs to code units
        rhofac = 1.820940e6
        pfac = 4.644481e+23
        tempfac = 3.401423e9
        efac = 2.972468e+49
        rfac = 4e8
        tfac = 0.7920256
        velfac = rfac / tfac

        dat = ['density', 'velx', 'vely', 'velz', 'energy',
               'press', 'temp', 'gam1', 'gam2', 'enuc1', 'enuc2',
               '0001','0002']

        block = prd.PROMPI_bindata(filename, dat)

        # PROMPI data vertical direction is X #
        # hence swap X for Y axis everywhere  #
        self.rho = np.swapaxes(block.datadict['density'],0,1) / rhofac
        self.pres = np.swapaxes(block.datadict['press'],0,1) / pfac
        self.temp = np.swapaxes(block.datadict['temp'],0,1) / tempfac

        velx = np.swapaxes(block.datadict['vely'],0,1)
        vely = np.swapaxes(block.datadict['velx'],0,1)
        velz = np.swapaxes(block.datadict['velz'],0,1)
        etot = np.swapaxes(block.datadict['energy'],0,1)

        ekin = 0.5*(velx**2.+vely**2.+velz**2.)
        eint = etot - ekin

        self.vel = np.array([velx,vely,velz]) / velfac
        self.eps = np.array(eint) / efac
        self.xnuc = np.array([np.swapaxes(block.datadict['0001'],0,1), np.swapaxes(block.datadict['0002'],0,1)])

        nx = np.array(block.datadict['qqx'])
        ny = np.array(block.datadict['qqy'])
        nz = np.array(block.datadict['qqz'])

        xzn0 = np.array(block.datadict['xzn0'])
        yzn0 = np.array(block.datadict['yzn0'])
        zzn0 = np.array(block.datadict['zzn0'])

        # dimension mind-fuck: hope this is right #
        gridx = np.empty((nx, ny, nz))
        for k, v in enumerate(yzn0/rfac): gridx[k, :, :] = v

        gridy = np.empty((nx, ny, nz))
        for k, v in enumerate(xzn0/rfac): gridy[:, k, :] = v

        gridz = np.empty((nx, ny, nz))
        for k, v in enumerate(zzn0/rfac): gridz[:, :, k] = v

        self.coords = np.array([gridx,gridy,gridz])

        deltax = np.asarray(block.datadict['yznr']) - np.asarray(block.datadict['yznl'])
        deltay = np.asarray(block.datadict['xznr']) - np.asarray(block.datadict['xznl'])
        deltaz = np.asarray(block.datadict['zznr']) - np.asarray(block.datadict['yznl'])

        dx = np.empty((nx, ny, nz))
        for k, v in enumerate(deltax/rfac): dx[k, :, :] = v

        dy = np.empty((nx, ny, nz))
        for k, v in enumerate(deltay/rfac): dy[:, k, :] = v

        dz = np.empty((nx, ny, nz))
        for k, v in enumerate(deltaz/rfac): dz[:, :, k] = v

        self.dx = np.array([dx[0,0,0], dy[0,0,0], dz[0,0,0]]) / rfac

        self.time = block.datadict['time'] / tfac

        super(PROMPI, self).__init__(**kwargs)


def divergence(f, *varargs):
    _, dx, dy, dz = np.gradient(f, *varargs)
    return dx[0] + dy[1] + dz[2]


def curl(f, *varargs):
    _, dx, dy, dz = np.gradient(f, *varargs)
    return np.array([dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]])

# code for sequence taken from SLH analysis scripts
class Sequence(object):
    """Plot a single value from a sequence of snapshots

    Computed values are cached!

    Example:

    # create new instance, constructor calls s.find_files()
    # to find all grid_n?????.slh
    s=analysis.Sequence(analysis.SLH)

    # plot mean Mach number
    s.plot("g.mach.mean()", color="red")

    # update the list of files
    s.find_files()
    s.replot()
    """

    def __init__(self, gridclass, dir='.', auto_cache=True, quiet=False, openkwargs={}, **findkwargs):

        self.dir = os.path.abspath(dir)

        self.gridclass = gridclass

        self.quiet = quiet

        self.filenumbers = []

        self.findkwargs = findkwargs
        self.openkwargs = openkwargs

        self.cached = True
        self.clear_cache()

        self.find_files()

        self.auto_cache = auto_cache
        if len(findkwargs) == 0:
            self.cache_filename = 'turbanalysis.cache'
        else:
            h = hashlib.sha256(str(findkwargs))
            self.cache_filename = 'turbanalysis_' + h.hexdigest() + '.cache'
        h = hashlib.sha256()
        h.update('0'.encode('utf-8'))  # internal version number
        h.update(dir.encode('utf-8'))
        h.update(str(openkwargs).encode('utf-8'))
        h.update(str(findkwargs).encode('utf-8'))
        self.cache_filename_unique = h.hexdigest() + '.cache'

        self.allplots = []

        if self.auto_cache:
            try:
                self.load_cache()
            except Exception as e:
                print("could not load cache file (%s)" % str(e), file=sys.stderr)

    def __del__(self):
        if self.auto_cache:
            self.save_cache()

    def clear_cache(self):
        self.cache = {}

    def save_cache(self):
        try:
            import cPickle as p
        except ImportError:
            import pickle as p

        try:
            with open(os.path.join(self.dir, self.cache_filename), 'wb') as fh:
                if not self.quiet:
                    print('writing cache file ' + fh.name)
                p.dump(self.cache, fh, protocol=2)
        except IOError:
            cachedir = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.join(os.environ['HOME'], '.cache')), 'turbanalysis')
            if not os.path.exists(cachedir):
                if not self.quiet:
                    print('creating directory ' + cachedir)
                os.makedirs(cachedir)
            with open(os.path.join(cachedir, self.cache_filename_unique), 'wb') as fh:
                if not self.quiet:
                    print('writing cache file ' + fh.name)
                p.dump(self.cache, fh, protocol=2)

    def load_cache(self):
        try:
            import cPickle as p
        except ImportError:
            import pickle as p

        try:
            with open(os.path.join(self.dir, self.cache_filename), 'rb') as fh:
                if not self.quiet:
                    print('reading cache file ' + fh.name)
                self.cache = p.load(fh)
        except IOError:
            try:
                with open(os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.join(os.environ['HOME'], '.cache')), 'turbanalysis', self.cache_filename_unique), 'rb') as fh:
                    if not self.quiet:
                        print('reading cache file ' + fh.name)
                    self.cache = p.load(fh)
            except IOError:
                if not self.quiet:
                    print('no cache file found')

    def find_files(self):
        self.filenumbers, self.filenames = self.gridclass.find_files(self.dir, **self.findkwargs)

    def get_time(self):

        try:
            if len(self.time) == len(self.filenumbers):
                return self.time
        except:
            pass

        time = []

        cached = self.cached and ("<time>" in self.cache)
        if self.cached and ("<time>" not in self.cache):
            self.cache["<time>"] = {}

        for i, n in zip(self.filenumbers, self.filenames):

            from_cache = False
            if cached:
                if i in self.cache["<time>"]:
                    t = self.cache["<time>"][i]
                    from_cache = True

            if not from_cache:
                g = self.gridclass(n, **self.openkwargs)
                t = g.time
                if cached:
                    self.cache["<time>"][i] = t

            time.append(t)

        self.time = np.array(time)
        return self.time

    def nearest_snapshot(self, t):
        time = self.get_time()
        return int(abs(time-t).argmin())

    def get_data(self, expr):
        time = []
        data = []

        cached = self.cached and expr in self.cache
        if self.cached and expr not in self.cache:
            self.cache[expr] = {}

        if self.cached and ("<time>" not in self.cache):
            self.cache["<time>"] = {}

        for i, n in zip(self.filenumbers, self.filenames):

            from_cache = False

            if cached:
                if i in self.cache[expr]:
                    t = self.cache[expr][i][0]
                    d = self.cache[expr][i][1]
                    from_cache = True

            if not from_cache:
                g = self.gridclass(n, **self.openkwargs)
                t = g.time
                d = eval(expr)

                if self.cached:
                    self.cache[expr][i] = [t, d]

            if self.cached:
                self.cache["<time>"][i] = t

            time.append(t)
            data.append(d)

        time = np.array(time)
        data = np.array(data)
        return time, data

    def get_tdiff(self, expr):
        time, data = self.get_data(expr)
        return time[:-1], (data[1:]-data[:-1])/(time[1:]-time[:-1])

    def get_tdiff_avg(self, expr, t1, t2):
        time, data = self.get_tdiff(expr)

        i1 = -1
        i2 = -1

        for i in range(time.shape[0]):
            if i1 < 0 and time[i] >= t1:
                i1 = i
            if i2 < 0 and time[i] >= t2:
                i2 = i

        return data[i1:i2].mean()

    def power_spectrum(self, expr, n=None):
        time, data = self.get_data(expr)
        dt = np.diff(time)
        if np.any(dt[0] != dt):
            warnings.warn("Time steps are not equal. Using average")
        dt = dt.mean()

        spec = np.fft.rfft(data, n=n)
        spec = np.abs(spec)**2
        if n is None:
            m = data.shape[0]
        else:
            m = n
        if m % 2 == 0:
            spec = spec[:-1]
        f = np.fft.fftfreq(m, d=dt)[:spec.shape[0]]
        return f, spec

    def plot(self, expr, tdiff=False, smooth=0, tunit=1., ax=None, **kwargs):
        """plot single value from every slh_grid versus time

        expr: string with python code that evaluates to the desired value
              the current grid object can be referenced with g

              Example: plot mean Mach number versus time
              plot("g.mach().mean()")

        **kwargs: additional arguments to the plot command
        """
        if tdiff:
            time, data = self.get_tdiff(expr)
        else:
            time, data = self.get_data(expr)
        if smooth > 0:
            data = self.smooth(data, smooth)
        time = time / tunit
        if ax is None:
            ax = plt.gca()
        line, = ax.plot(time, data, **kwargs)
        opt = {'tdiff': tdiff, 'smooth': smooth, 'tunit': tunit}
        self.allplots.append((line, expr, opt))

        plt.draw_if_interactive()

    def smooth(self, data, n):
        r = np.arange(data.shape[0])

        ndata = np.zeros(data.shape)

        for i in range(-n, n+1):
            ndata += data.take(r+i, mode='clip')

        return ndata/(2.*n+1.)

    def tavg(self, expr, dt=None):
        time, data = self.get_data(expr)
        res = []
        if dt is None:
            return time, data.mean(axis=0)
        else:
            for t in time:
                tselect = np.abs(t - time) <= dt
                res.append(data[tselect].mean(axis=0))
            return time, res

    def replot(self):
        oldplots = self.allplots
        self.allplots = []
        for l, expr, opt in oldplots:
            style = {'c': l.get_c(), 'ls': l.get_ls(), 'lw': l.get_lw(), 'marker': l.get_marker(), 'markersize': l.get_markersize(), 'label': l.get_label()}
            style.update(opt)
            ax = l.get_axes()
            ax.lines.remove(l)
            self.plot(expr, ax=ax, **style)
        plt.draw_if_interactive()

    def semilogy(self, *args, **kwargs):
        """similar to plot with logarithmic y-axis"""
        self.plot(*args, **kwargs)
        plt.yscale('log')


def read_mmap(f, shape, **kwargs):
    if not isinstance(f, mmap.mmap):
        raise TypeError('argument f must be of type mmap.mmap')
    kwargs.setdefault('order', 'F')
    kwargs.setdefault('offset', f.tell())
    res = np.ndarray(shape, buffer=f, **kwargs)
    f.seek(len(res.data), os.SEEK_CUR)
    return res
