import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.PROMPI.PROMPI_data as pd


class SpectrumUzVariance():

    def __init__(self, filename, data_prefix, ig, lhc):

        block = pd.PROMPI_bindata(filename, ['velz'])

        xzn0 = block.datadict['xzn0']
        velz = block.datadict['velz']

        xlm = np.abs(np.asarray(xzn0) - np.float(lhc))
        ilhc = int(np.where(xlm == xlm.min())[0][0])

        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        # initialize 
        ig = int(ig)

        if (ig == 1):
            sterad = np.ones((nz, ny))
            steradtot = np.sum(sterad)
        elif (ig == 2):
            print("ERROR (SpectrumUyVariance.py): Spherical Geometry not implemented.")
            # sterad =
            # steradtot =
        else:
            print("ERROR (SpectrumUyVariance.py): Geometry not implemented")

        # get horizontal data
        hvelz = velz[ilhc][:][:]

        # calculate horizontal mean value
        eh_uz = np.sum(hvelz * sterad) / steradtot

        # calculate Reynolds fluctuations
        uzf_r = hvelz - eh_uz

        # calculate Fourier coefficients (a_ and b_)
        uzfr_fft = np.fft.fft2(uzf_r)

        a_uzff = np.real(uzfr_fft)
        b_uzff = np.imag(uzfr_fft)

        # calculate energy contribution to total variance 
        # from real and imaginary parts of Fourier transforms  

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
        spect_uzvar = []
        for ishell in kh:
            mask = np.where((aa >= float(ishell)) & (aa < float(ishell + 1)), 1., 0.)
            integrand_uzvar = mask * energy_uzff
            spect_uzvar.append(np.sum(integrand_uzvar) / (float(nn) * float(nn)))
            # fig, (ax1) = plt.subplots(figsize=(3, 3))
            # pos = ax1.imshow(mask,interpolation='bilinear',origin='lower',extent=(0,nn,0,nn))
            # fig.colorbar(pos, ax=ax1)
            # plt.show(block=False)

        # calculate mean TKE spectrum  
        spect_uzvar_mean = []
        i = -1
        for ishell in kh:
            i += 1
            spect_uzvar_mean.append(spect_uzvar[i] / (float(ny) * float(nz)))

        # check Parseval's theorem

        total_uzuz = (uzf_r * uzf_r)
        # print(uxf_r*uxf_r,uyf_r*uyf_r,uzf_r*uzf_r,uxf_r*uxf_r+uyf_r*uyf_r+uzf_r*uzf_r)
        # print(np.sum(uxf_r*uxf_r+uyf_r*uyf_r+uzf_r*uzf_r))
        print(np.sum(total_uzuz), np.sum(spect_uzvar))

        # share stuff across class
        self.spect_uzvar = spect_uzvar
        self.spect_uzvar_mean = spect_uzvar_mean
        self.kh = kh
        self.data_prefix = data_prefix

    def plot_UZspectrum(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot UZ spectrum"""

        # load horizontal wave number GRID
        grd1 = self.kh

        # load spectrum to plot
        plt1 = self.spect_uzvar_mean

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   						
        plt.axis([xbl, xbr, ybu, ybd])

        # plot DATA 
        plt.title('uz variance "energy" spectrum')
        plt.loglog(grd1, plt1, color='brown', label=r"$u'_z u'_z$")
        plt.loglog(grd1, max(plt1) * grd1 ** (-5. / 3.), color='r', linestyle='--', label=r"k$^{-5/3}$")

        setxlabel = r'k$_h$'
        setylabel = r"$u_z$ variance (erg g$^{-1}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=0, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'uzvariancespectrum.png')

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
