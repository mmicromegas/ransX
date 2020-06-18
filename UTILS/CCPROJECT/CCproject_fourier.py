import UTILS.PROMPI.PROMPI_data as prd
import numpy as np
import os
# https://docs.sympy.org/dev/modules/physics/vector/api/fieldfunctions.html#curl
# from sympy.physics.vector import ReferenceFrame
# from sympy.physics.vector import curl
import sys


class CCproject_fourier:

    # this is the output function for profiles
    def __init__(self, filename):

        # available bindata #
        # density, velx, vely, velz, energy,
        # press, temp, gam1, gam2, enuc1, enuc2,
        # 0001, 0002

        dat = ['velx', 'vely', 'velz']
        block = prd.PROMPI_bindata(filename, dat)

        # get time and x-grid
        time = block.datadict['time']
        xzn0 = block.datadict['xzn0']

        velx = block.datadict['velx']
        vely = block.datadict['vely']
        velz = block.datadict['velz']

        # lhc == 1.5*y i.e. 6.e8 cm
        lhc = 6.e8
        xlm = np.abs(np.asarray(xzn0) - np.float(lhc))
        ilhc = int(np.where(xlm == xlm.min())[0][0])

        nx = block.datadict['qqx']
        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        # initialize
        ig = 1 # cartesian geometry

        if (ig == 1):
            sterad = np.ones((nz, ny))
            steradtot = np.sum(sterad)
        elif (ig == 2):
            print("ERROR (SpectrumTurbulentKineticEnergy.py): Spherical Geometry not implemented.")
            # sterad =
            # steradtot =
        else:
            print("ERROR (SpectrumTurbulentKineticEnergy.py): Geometry not implemented")

        # get horizontal data
        hvelx = velx[ilhc][:][:]
        hvely = vely[ilhc][:][:]
        hvelz = velz[ilhc][:][:]

        # calculate horizontal mean value
        eh_ux = np.sum(hvelx * sterad) / steradtot
        eh_uy = np.sum(hvely * sterad) / steradtot
        eh_uz = np.sum(hvelz * sterad) / steradtot

        # calculate Reynolds fluctuations
        uxf_r = hvelx - eh_ux
        uyf_r = hvely - eh_uy
        uzf_r = hvelz - eh_uz

        # calculate Fourier coefficients (a_ and b_)
        uxfr_fft = np.fft.fft2(uxf_r)
        uyfr_fft = np.fft.fft2(uyf_r)
        uzfr_fft = np.fft.fft2(uzf_r)

        a_uxff = np.real(uxfr_fft)
        b_uxff = np.imag(uxfr_fft)

        a_uyff = np.real(uyfr_fft)
        b_uyff = np.imag(uyfr_fft)

        a_uzff = np.real(uzfr_fft)
        b_uzff = np.imag(uzfr_fft)

        # calculate energy contribution to total variance
        # from real and imaginary parts of Fourier transforms

        energy_uxff = a_uxff * a_uxff + b_uxff * b_uxff
        energy_uyff = a_uyff * a_uyff + b_uyff * b_uyff
        energy_uzff = a_uzff * a_uzff + b_uzff * b_uzff

        # define wave numbers (in 2pi/N units)
        if (ny == nz):
            nn = ny
            nnmax = np.round(np.sqrt(2. * (nn ** 2.)))

        # array of horizontal wave numbers
        kh = np.arange((nnmax / 2))
        khmax = int(max(kh))

        # calculate distance from nearest corners in nn x nn matrix
        # and use it later to computer mask for integration over
        # spherical shells
        aa = self.dist(nn)

        # integrate over radial shells and calculate total TKE spectrum
        spect_tke = []
        for ishell in kh:
            mask = np.where((aa >= float(ishell)) & (aa < float(ishell + 1)), 1., 0.)
            integrand_tke = mask * 0.5 * (energy_uxff + energy_uyff + energy_uzff)
            spect_tke.append(np.sum(integrand_tke) / (float(nn) * float(nn)))
            # fig, (ax1) = plt.subplots(figsize=(3, 3))
            # pos = ax1.imshow(mask,interpolation='bilinear',origin='lower',extent=(0,nn,0,nn))
            # fig.colorbar(pos, ax=ax1)
            # plt.show(block=False)

        # calculate mean TKE spectrum
        spect_tke_mean = []
        i = -1
        for ishell in kh:
            i += 1
            spect_tke_mean.append(spect_tke[i] / (float(ny) * float(nz)))

        # check Parseval's theorem

        total_tke = (uxf_r * uxf_r + uyf_r * uyf_r + uzf_r * uzf_r) / 2.0
        # print(uxf_r*uxf_r,uyf_r*uyf_r,uzf_r*uzf_r,uxf_r*uxf_r+uyf_r*uyf_r+uzf_r*uzf_r)
        # print(np.sum(uxf_r*uxf_r+uyf_r*uyf_r+uzf_r*uzf_r))
        # print(np.sum(total_tke), np.sum(spect_tke))

        # share stuff across class
        self.spect_tke_mean = spect_tke_mean
        self.kh = kh
        self.time = time
        # print(kh)

    def getData(self):
        return {'wavenumber': self.kh, 'time': self.time, 'tkespect': self.spect_tke_mean}

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

        array = np.recarray(len(p['wavenumber']), dtype=[
            ('IR', 'int'), ])

        #array = np.array(len(p['wavenumber']))

        array['IR'] = np.arange(0, len(p['wavenumber']))
        #print(array['IR'])

        for c, o in zip(columns, outputnames):
            array = recfunctions.append_fields(array, o, p[c])

        outputnames = ['k'] + outputnames

        # dir = os.path.join('..','DATA')
        fdir = os.path.join('DATA', 'CCP')
        ffdir = os.path.join(fdir, os.path.basename(filename))

        filename = ffdir + '-{:04d}.fourier'.format(n)
        print(filename)
        with open(filename, 'wb') as f:
            f.write(('DUMP  {},t = {:.8e}\n'.format(n, t )).encode('ascii'))
            # f.write(('Nx = {} \n\n\n'.format(len(p['x']))).encode('ascii'))
            f.write(('\n\n'.format().encode('ascii')))
            f.write((' '.join('{:13s}  '.format(k) for k in outputnames)).encode('ascii'))
            f.write(('\n\n').encode('ascii'))
            for i in range(len(p['wavenumber'])):
                profline = array[i].tolist()
                nline = profline[0]
                profline = np.array(profline[1:]) / units
                txt = ' '.join(' {:.8e}'.format(k) for k in profline) + '\n'
                txt = '{0:<5}'.format(nline) + txt
                f.write(txt.encode('ascii'))

    # source: https://gist.github.com/abeelen/453de325dd9787ea2aa7fad495f4f018
    def dist(self, NAXIS):
        """Returns a rectangular array in which the value of each element is proportional to its frequency.
        >>> dist(3)
        array([[ 0.        ,  1.        ,  1.        ],
               [ 1.        ,  1.41421356,  1.41421356],
               [ 1.        ,  1.41421356,  1.41421356]])
        >>> dist(4)
        array([[ 0.        ,  1.        ,  2.        ,  1.        ],
               [ 1.        ,  1.41421356,  2.23606798,  1.41421356],
               [ 2.        ,  2.23606798,  2.82842712,  2.23606798],
               [ 1.        ,  1.41421356,  2.23606798,  1.41421356]])
        """
        axis = np.linspace(-NAXIS / 2 + 1, NAXIS / 2, NAXIS)
        result = np.sqrt(axis ** 2 + axis[:, np.newaxis] ** 2)
        return np.roll(result, NAXIS / 2 + 1, axis=(0, 1))
