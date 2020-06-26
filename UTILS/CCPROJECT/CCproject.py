import UTILS.PROMPI.PROMPI_data as prd
import numpy as np
import os
# https://docs.sympy.org/dev/modules/physics/vector/api/fieldfunctions.html#curl
# from sympy.physics.vector import ReferenceFrame
# from sympy.physics.vector import curl
import sys


class CCproject:

    # this is the output function for profiles
    def __init__(self, filename):

        # available bindata #
        # density, velx, vely, velz, energy,
        # press, temp, gam1, gam2, enuc1, enuc2,
        # 0001, 0002

        dat = ['density', 'velx', 'vely', 'velz', 'energy', 'press', 'temp', '0001', '0002']
        block = prd.PROMPI_bindata(filename, dat)

        # get time and x-grid
        time = block.datadict['time']
        xzn0 = block.datadict['xzn0']

        # get density-weighted horizontal averages

        # fh_press = self.calcFHdata(block, 'press')
        # fh_temp = self.calcFHdata(block, 'temp')
        
        eh_rho = self.calcEHdata(block, 'density')
        eh_press = self.calcEHdata(block, 'press')
        eh_temp = self.calcEHdata(block, 'temp')
        eh_Avalue = self.calcEhAvalue(block)
        fh_x0002 = self.calcFHdata(block, '0002')
        fh_vel = self.calcFHvel(block)

        minq_rho = self.minQ(block,'density')
        maxq_rho = self.maxQ(block,'density')
        stdevq_rho = self.stDevQ(block, 'density')

        minq_x2 = self.minQ(block,'0002')
        maxq_x2 = self.maxQ(block,'0002')
        stdevq_x2 = self.stDevQ(block, '0002')

        minq_press = self.minQ(block,'press')
        maxq_press = self.maxQ(block,'press')
        stdevq_press = self.stDevPress(block, 'press')

        minq_temp = self.minQ(block,'temp')
        maxq_temp = self.maxQ(block,'temp')
        stdevq_temp = self.stDevQ(block, 'temp')

        nx = block.datadict['qqx']
        minq_Avalue = self.minAvalue(nx, self.calcAvalue(block))
        maxq_Avalue = self.maxAvalue(nx, self.calcAvalue(block))
        stdevq_Avalue = self.stDevAvalue(nx, self.calcAvalue(block))

        nx = block.datadict['qqx']
        minq_Vmag = self.minVmag(nx, block)
        maxq_Vmag = self.maxVmag(nx, block)
        stdevq_Vmag = self.stdevVmag(nx, block)

        fh_velx = self.calcFHdata(block, 'velx')
        fh_vely = self.calcFHdata(block, 'vely')
        fh_velz = self.calcFHdata(block, 'velz')

        velxmag = self.calcFH_Vxmag(block)
        velhormag = self.calcFH_Vhormag(block)

        # fh_x0001 = self.calcFHdata(block, '0001')
        # fh_x0002 = self.calcFHdata(block, '0002')
        # eh_gam1 = self.calcEHdata(block, 'gam1')
        # eh_gam2 = self.calcEHdata(block, 'gam2')
        # fh_enuc1 = self.calcFHdata(block, 'enuc1')

        # vorticitymag = self.calcVorticityMagnitude(block)

        self.time = time
        self.eh_rho = eh_rho
        self.eh_press = eh_press
        self.eh_temp = eh_temp
        self.eh_Avalue = eh_Avalue
        self.fh_x0002 = fh_x0002
        self.vel = fh_vel

        self.minq_rho = minq_rho
        self.maxq_rho = maxq_rho
        self.stdevq_rho = stdevq_rho

        self.minq_x2 = minq_x2
        self.maxq_x2 = maxq_x2
        self.stdevq_x2 = stdevq_x2

        self.minq_press = minq_press
        self.maxq_press = maxq_press
        self.stdevq_press = stdevq_press

        self.minq_temp = minq_temp
        self.maxq_temp = maxq_temp
        self.stdevq_temp = stdevq_temp

        self.minq_Avalue = minq_Avalue
        self.maxq_Avalue = maxq_Avalue
        self.stdevq_Avalue = stdevq_Avalue

        self.minq_Vmag = minq_Vmag
        self.maxq_Vmag = maxq_Vmag
        self.stdevq_Vmag = stdevq_Vmag

        self.velx = fh_velx
        self.vely = fh_vely
        self.velz = fh_velz

        self.velxmag = velxmag
        self.velhormag = velhormag

        self.Hflux = self.calcFH_Hflux(block)
        self.KEflux = self.calcFH_KEflux(block)

        # self.fh_x0001 = fh_x0001
        # self.eh_gam1 = eh_gam1
        # self.eh_gam2 = eh_gam2
        # self.fh_enuc1 = fh_enuc1

        self.xzn0 = xzn0

    def getData(self):
        return {'x': self.xzn0, 'time': self.time, 'density': self.eh_rho, 'pressure': self.eh_press,
                'temp': self.eh_temp, 'velx': self.velx, 'vely': self.vely, 'velz': self.velz,
                'x2': self.fh_x0002, 'vel': self.vel, 'Avalue': self.eh_Avalue,
                'minq_rho': self.minq_rho, 'maxq_rho': self.maxq_rho, 'stdevq_rho': self.stdevq_rho,
                'minq_x2': self.minq_x2, 'maxq_x2': self.maxq_x2, 'stdevq_x2': self.stdevq_x2,
                'minq_press': self.minq_press, 'maxq_press': self.maxq_press, 'stdevq_press': self.stdevq_press,
                'minq_temp': self.minq_temp, 'maxq_temp': self.maxq_temp, 'stdevq_temp': self.stdevq_temp,
                'minq_Avalue': self.minq_Avalue, 'maxq_Avalue': self.maxq_Avalue, 'stdevq_Avalue': self.stdevq_Avalue,
                'minq_Vmag': self.minq_Vmag, 'maxq_Vmag': self.maxq_Vmag, 'stdevq_Vmag': self.stdevq_Vmag,
                'velxmag' : self.velxmag, 'velhormag': self.velhormag, 'Hflux': self.Hflux, 'KEflux': self.KEflux}

    def write_output(self, p, t, columns, filename, outputnames=[], n=1, units=[]):
        """
            a function that writes profiles into a file, according
            to the format specified in the CodeComparison Project

            Input:
            p: a yt profile instance
            t: the time of the current snapshot
            columns: the list of fields that are written to file.
                     They have to be available in p
            filename: the name of the file. Will be apended by '-#.rprof',
                      where # is the dump number n

            Optional:
            outputnames: a list of alternative fieldnames,
                        that will be printed in the file.
                        Defaults to the columns list.
            n : the dump number, defaults to 1
            units: np array of conversion factors to get to the specified dimensionless units.
                   defaults to 1
        """
        from numpy.lib import recfunctions

        if len(outputnames) != len(columns):
            print("""Incomplete set of output names provided - 
                  We will use the field names in the output""")
            outputnames = columns
        if len(units) != len(columns):
            print("""Incomplete set of unit conversions - 
                  We will ignore ALL conversions""")
            units = np.ones(len(columns))
        if isinstance(units, list):
            units = np.array(units)

        try:
            t = t.to_ndarray()
        except:
            pass

        array = np.recarray(len(p['x']), dtype=[
            ('IR', 'int'), ])

        array['IR'] = np.arange(len(p['x']), 0, -1)

        for c, o in zip(columns, outputnames):
            array = recfunctions.append_fields(array, o, p[c][::-1])

        outputnames = ['IR'] + outputnames

        # dir = os.path.join('..','DATA')
        fdir = os.path.join('DATA', 'CCP')
        ffdir = os.path.join(fdir, os.path.basename(filename))

        filename = ffdir + '-{:04d}.rprof'.format(n)
        print(filename)
        with open(filename, 'wb') as f:
            f.write(('DUMP  {},t = {:.8e}\n'.format(n, t )).encode('ascii'))
            f.write(('Nx = {} \n\n\n'.format(len(p['x']))).encode('ascii'))
            f.write((' '.join('{:13s}  '.format(k) for k in outputnames)).encode('ascii'))
            f.write(('\n\n').encode('ascii'))
            for i in range(len(p['x'])):
                profline = array[i].tolist()
                nline = profline[0]
                profline = np.array(profline[1:]) / units
                txt = ' '.join(' {:.8e}'.format(k) for k in profline) + '\n'
                txt = '{0:<5}'.format(nline) + txt
                f.write(txt.encode('ascii'))

    def calcFHdata(self, block, field):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        rho = block.datadict['density']
        data = block.datadict[field]

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate mean density
        eh_rho = []
        for i in range(nx):
            hdata = rho[i, :, :]
            eh_rho.append(np.sum(hdata * sterad) / steradtot)

        eh_rho = np.asarray(eh_rho)

        # calculate density-weighted horizontal average
        eh_data = []
        for i in range(nx):
            hdata = rho[i, :, :] * data[i, :, :]
            eh_data.append(np.sum(hdata * sterad) / steradtot)

        fh_data = np.asarray(eh_data / eh_rho)

        return fh_data

    def calcEHdata(self, block, field):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        data = block.datadict[field]

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate horizontal average
        eh_data = []
        for i in range(nx):
            hdata = data[i, :, :]
            eh_data.append(np.sum(hdata * sterad) / steradtot)

        return eh_data

    def calcFHvel(self, block):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        rho = block.datadict['density']
        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate mean density
        eh_rho = []
        for i in range(nx):
            hdata = rho[i, :, :]
            eh_rho.append(np.sum(hdata * sterad) / steradtot)

        eh_rho = np.asarray(eh_rho)

        # get velocity
        vel = np.sqrt(velx ** 2. + vely ** 2. + velz ** 2)

        # calculate density-weighted horizontal average
        eh_data = []
        for i in range(nx):
            hdata = rho[i, :, :] * vel[i, :, :]
            eh_data.append(np.sum(hdata * sterad) / steradtot)

        fh_data = np.asarray(eh_data / eh_rho)

        return fh_data

    def calcAvalue(self, block):

        rho = block.datadict['density']
        press = block.datadict['press']

        # calculate A = p/rho**5/3
        Avalue = press / (rho ** (5. / 3.))


        return Avalue

    def calcEhAvalue(self, block):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        rho = block.datadict['density']
        press = block.datadict['press']

        # calculate A = p/rho**5/3
        Avalue = press / (rho ** (5. / 3.))

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate mean density
        eh_Avalue = []
        for i in range(nx):
            hdata = Avalue[i, :, :]
            eh_Avalue.append(np.sum(hdata * sterad) / steradtot)

        return eh_Avalue


    def calcVorticityMagnitude(self, block):

        # CALCULATE THE CURVE OF VECTOR FIELD a WHICH HAS COMPONENTS ax, ay, az #
        # source: https://github.com/CaseyAMeakin/PROMPI/blob/Develop/root/analysis/idl/curl.pro

        qqx = block.datadict['qqx']
        qqy = block.datadict['qqy']
        qqz = block.datadict['qqz']

        xl = block.datadict['xznl']
        x0 = block.datadict['xzn0']
        xr = block.datadict['xznr']

        yl = block.datadict['yznl']
        y0 = block.datadict['yzn0']
        yr = block.datadict['yznr']

        zl = block.datadict['zznl']
        z0 = block.datadict['zzn0']
        zr = block.datadict['zznr']

        ax = block.datadict['velx']
        ay = block.datadict['vely']
        az = block.datadict['velz']

        # original declaration:: has different dimension order

        # curl1 = dblarr(qqy, qqz, qqx)
        # curl2 = dblarr(qqy, qqz, qqx)
        # curl3 = dblarr(qqy, qqz, qqx)

        curl1 = np.zeros((qqx,qqy,qqz))
        curl2 = np.zeros((qqx,qqy,qqz))
        curl3 = np.zeros((qqx,qqy,qqz))

        for i in range(1,qqx-2):
            for j in range(0,qqy-1):
                for k in range(0,qqz-1):

                    dels1 = xr[i] - xl[i]
                    dels2 = x0[i] * (yr[j] - yl[j])
                    dels3 = x0[i] * (zr[k] - zl[k])

                    delA1 = dels2 * dels3
                    delA2 = dels1 * dels3
                    delA3 = dels1 * dels2

                    # 1 - Curl
                    ip = i + 1
                    im = i - 1

                    if j == 0:
                        jm = qqy-1
                    else:
                        jm = j-1

                    if j == qqy-1:
                        jp = 0
                    else:
                        jp = j+1

                    if k == 0:
                        km = qqz-1
                    else:
                        km = k-1

                    if k == qqz-1:
                        kp = 0
                    else:
                        kp = k+1

                    v2p = 0.5 * (ay[j, k, i] + ay[j, kp, i])
                    v2m = 0.5 * (ay[j, k, i] + ay[j, km, i])
                    v3p = 0.5 * (az[j, k, i] + az[jm, k, i])
                    v3m = 0.5 * (az[j, k, i] + az[jp, k, i])

                    curl1[j, k, i] = (dels2 * (v2p - v2m) + dels3 * (v3p - v3m)) / delA1

                    # 2 - Curl
                    if k == 0:
                        km = qqz-1
                    else:
                        km = k-1

                    if k == qqz-1:
                        kp = 0
                    else:
                        kp = k+1

                    v1p = 0.5 * (ax[j, k, i] + ax[j, kp, i])
                    v1m = 0.5 * (ax[j, k, i] + ax[j, km, i])
                    v3p = 0.5 * (az[j, k, i] + az[j, k, im])
                    v3m = 0.5 * (az[j, k, i] + az[j, k, ip])

                    curl2[j, k, i] = (dels1 * (v1p - v1m) + dels3 * (v3p - v3m)) / delA2

                    # 3 - Curl
                    if j == 0:
                        jm = qqy-1
                    else:
                        jm = j-1

                    if j == qqy-1:
                        jp = 0
                    else:
                        jp = j+1

                    v1p = 0.5 * (ax[j, k, i] + ax[jm, k, i])
                    v1m = 0.5 * (ax[j, k, i] + ax[jp, k, i])
                    v2p = 0.5 * (ay[j, k, i] + ay[j, k, ip])
                    v2m = 0.5 * (ay[j, k, i] + ay[j, k, im])

                    curl3[j, k, i] = (dels1 * (v1p - v1m) + dels2 * (v2p - v2m)) / delA3

        curlmag = np.sqrt(curl1**2.+curl2**2.+curl3**2.)

        return curlmag

    def minQ(self, block, field):

        nx = block.datadict['qqx']
        data = block.datadict[field]

        # get min value
        min_data = []
        for i in range(nx):
            hdata = data[i, :, :]
            min_data.append(np.min(hdata))

        return min_data

    def maxQ(self, block, field):

        nx = block.datadict['qqx']
        data = block.datadict[field]

        # get max value
        max_data = []
        for i in range(nx):
            hdata = data[i, :, :]
            max_data.append(np.max(hdata))

        return max_data

    def stDevQ(self, block, field):

        nx = block.datadict['qqx']
        data = block.datadict[field]

        # calculate standard deviation
        stdev_data = []
        for i in range(nx):
            hdata = data[i, :, :]
            stdev_data.append(np.std(hdata.astype(np.float64)))

        return stdev_data

    def stDevPress(self, block, field):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']
        data = block.datadict[field]

        # calculate standard deviation
        stdev_data = []

        for i in range(nx):
            mmm = np.mean(data[i,:,:])
            var = []
            for j in range(ny):
                for k in range(nz):
                        var.append((data[i,j,k]-mmm)**2)
            varmean = np.mean(var)
            stdev = np.sqrt(varmean)
            stdev_data.append(stdev)

        return stdev_data

    def minAvalue(self, nx, Avalue):

        data = Avalue

        # get min value
        min_data = []
        for i in range(nx):
            hdata = data[i, :, :]
            min_data.append(np.min(hdata))

        return min_data


    def maxAvalue(self, nx, Avalue):

        data = Avalue

        # get min value
        max_data = []
        for i in range(nx):
            hdata = data[i, :, :]
            max_data.append(np.max(hdata))

        return max_data

    def stDevAvalue(self, nx, Avalue):

        data = Avalue

        # calculate standard deviation
        stdev_data = []
        for i in range(nx):
            hdata = data[i, :, :]
            stdev_data.append(np.std(hdata.astype(np.float64)))

        return stdev_data

    def minVmag(self, nx, block):
        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # get velocity
        vel = np.sqrt(velx ** 2. + vely ** 2. + velz ** 2)

        # get min value
        min_vmag = []
        for i in range(nx):
            hdata = vel[i, :, :]
            min_vmag.append(np.min(hdata))

        return min_vmag

    def maxVmag(self, nx, block):
        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # get velocity
        vel = np.sqrt(velx ** 2. + vely ** 2. + velz ** 2)

        # get max value
        max_vmag = []
        for i in range(nx):
            hdata = vel[i, :, :]
            max_vmag.append(np.max(hdata))

        return max_vmag

    def stdevVmag(self, nx, block):
        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # get velocity
        vel = np.sqrt(velx ** 2. + vely ** 2. + velz ** 2)

        # get stdev value
        stdev_vmag = []
        for i in range(nx):
            hdata = vel[i, :, :]
            stdev_vmag.append(np.std(hdata))

        return stdev_vmag

    def calcFH_Vxmag(self, block):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        rho = block.datadict['density']
        velx = block.datadict['velx']

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate mean density
        eh_rho = []
        for i in range(nx):
            hdata = rho[i, :, :]
            eh_rho.append(np.sum(hdata * sterad) / steradtot)

        eh_rho = np.asarray(eh_rho)

        # get velocity
        vel = np.sqrt(velx ** 2.)

        # calculate density-weighted horizontal average
        eh_data = []
        for i in range(nx):
            hdata = rho[i, :, :] * vel[i, :, :]
            eh_data.append(np.sum(hdata * sterad) / steradtot)

        fh_data = np.asarray(eh_data / eh_rho)

        return fh_data

    def calcFH_Vhormag(self, block):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        rho = block.datadict['density']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate mean density
        eh_rho = []
        for i in range(nx):
            hdata = rho[i, :, :]
            eh_rho.append(np.sum(hdata * sterad) / steradtot)

        eh_rho = np.asarray(eh_rho)

        # get velocity
        vel = np.sqrt(vely ** 2. + velz ** 2.)

        # calculate density-weighted horizontal average
        eh_data = []
        for i in range(nx):
            hdata = rho[i, :, :] * vel[i, :, :]
            eh_data.append(np.sum(hdata * sterad) / steradtot)

        fh_data = np.asarray(eh_data / eh_rho)

        return fh_data


    def calcFH_Hflux(self, block):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        rho = block.datadict['density']
        press = block.datadict['press']
        energy = block.datadict['energy']
        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # calculate enthalpy
        ek = 0.5*(velx**2.+vely*2.+velz*2.)
        ei = energy - ek
        hh = ei + press/rho
        qdata = hh

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate mean density
        eh_rho = []
        for i in range(nx):
            hdata = rho[i, :, :]
            eh_rho.append(np.sum(hdata * sterad) / steradtot)

        eh_rho = np.asarray(eh_rho)

        # calculate mean rhoQvelx
        eh_rhoQvelx = []
        rhoQvelx = rho*qdata*velx
        for i in range(nx):
            hdata = rhoQvelx[i, :, :]
            eh_rhoQvelx.append(np.sum(hdata * sterad) / steradtot)

        eh_rhoQvelx = np.asarray(eh_rhoQvelx)

        # calculate mean rhoQ
        eh_rhoQ = []
        rhoQ = rho*qdata
        for i in range(nx):
            hdata = rhoQ[i, :, :]
            eh_rhoQ.append(np.sum(hdata * sterad) / steradtot)

        eh_rhoQ = np.asarray(eh_rhoQ)

        # caluclate mean rhovelx
        eh_rhovelx = []
        rhovelx = rho*velx
        for i in range(nx):
            hdata = rhovelx[i, :, :]
            eh_rhovelx.append(np.sum(hdata * sterad) / steradtot)

        eh_rhovelx = np.asarray(eh_rhovelx)

        # get flux
        flux = eh_rhoQvelx - eh_rhoQ*eh_rhovelx/eh_rho

        return flux

    def calcFH_KEflux(self, block):

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        rho = block.datadict['density']
        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # calculate kinetic energy
        ek = 0.5*(velx**2.+vely*2.+velz*2.)
        qdata = ek

        # for cartesian geometry
        sterad = np.ones((nz, ny))
        steradtot = np.sum(sterad)

        # calculate mean density
        eh_rho = []
        for i in range(nx):
            hdata = rho[i, :, :]
            eh_rho.append(np.sum(hdata * sterad) / steradtot)

        eh_rho = np.asarray(eh_rho)

        # calculate mean rhoQvelx
        eh_rhoQvelx = []
        rhoQvelx = rho*qdata*velx
        for i in range(nx):
            hdata = rhoQvelx[i, :, :]
            eh_rhoQvelx.append(np.sum(hdata * sterad) / steradtot)

        eh_rhoQvelx = np.asarray(eh_rhoQvelx)

        # calculate mean rhoQ
        eh_rhoQ = []
        rhoQ = rho*qdata
        for i in range(nx):
            hdata = rhoQ[i, :, :]
            eh_rhoQ.append(np.sum(hdata * sterad) / steradtot)

        eh_rhoQ = np.asarray(eh_rhoQ)

        # caluclate mean rhovelx
        eh_rhovelx = []
        rhovelx = rho*velx
        for i in range(nx):
            hdata = rhovelx[i, :, :]
            eh_rhovelx.append(np.sum(hdata * sterad) / steradtot)

        eh_rhovelx = np.asarray(eh_rhovelx)

        # get flux
        flux = eh_rhoQvelx - eh_rhoQ*eh_rhovelx/eh_rho



        return flux
