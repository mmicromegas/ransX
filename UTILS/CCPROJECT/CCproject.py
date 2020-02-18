import UTILS.PROMPI.PROMPI_data as prd
import numpy as np
import os
import sys


class CCproject:

    # this is the output function for profiles
    def __init__(self, filename):

        dat = ['temp', 'velx', 'density', '0002']
        block = prd.PROMPI_bindata(filename, dat)

        # get time and x-grid
        time = block.datadict['time']
        xzn0 = block.datadict['xzn0']

        # get density-weighted horizontal averages
        # available data (velx, temp, density, 0001, 0002)
        fh_dens = self.calcFHdata(block, 'density')
        fh_x0002 = self.calcFHdata(block, '0002')

        self.time = time
        self.fh_dens = fh_dens
        self.fh_x0002 = fh_x0002
        self.xzn0 = xzn0

    def getData(self):
        return {'x': self.xzn0, 'time': self.time, 'density': self.fh_dens, 'x1': self.fh_x0002}

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
            f.write(('DUMP  {},t = {:.8e}\n'.format(n, t / 0.7951638)).encode('ascii'))
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

        fh_data = np.asarray(eh_data/eh_rho)

        return fh_data
