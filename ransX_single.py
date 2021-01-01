###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_single.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: December/2020
# Desc: plots mean fields from single ransdat
# Usage: run ransX_single.py

from UTILS.PROMPI.PROMPI_single import PROMPI_single
from UTILS.SINGLE.ReadParamsSingle import ReadParamsSingle

import warnings
import os
import sys
import errno


def main():
    warnings.filterwarnings("ignore")

    # check python version
    if sys.version_info[0] < 3:
        print("Python " + str(sys.version_info[0]) + "  is not supported. EXITING.")
        sys.exit()

    # read input parameters
    paramFile = os.path.join('PARAMS', 'param.single')
    params = ReadParamsSingle(paramFile)

    datafile = params.getForSingle('single')['datafile']
    endianness = params.getForSingle('single')['endianness']
    precision = params.getForSingle('single')['precision']

    xbl = params.getForSingle('single')['xbl']
    xbr = params.getForSingle('single')['xbr']

    q2plot = params.getForSingle('single')['q']

    ransdat = PROMPI_single(datafile, endianness, precision)

    ransdat.SetMatplotlibParams()

    # check/create RESULTS folder
    try:
        os.makedirs('RESULTS')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # ransdat.plotMonstarIni()

    for q in q2plot:
        ransdat.plot_lin_q1(xbl, xbr, q, r'r (cm)', q, q)

    # for q in q2plot:
    #    ransdat.plot_log_q1(xbl,xbr,q,r'r (cm)',q,q)

    # ransdat.plot_check_heq1()
    # ransdat.plot_check_heq2(xbl,xbr)
    # ransdat.plot_check_heq3()
    # ransdat.plot_check_ux(xbl,xbr)
    # ransdat.plot_nablas(xbl,xbr)
    # ransdat.plot_dx(xbl,xbr)
    # ransdat.plot_mm(xbl,xbr)
    ransdat.PlotNucEnergyGen(xbl, xbr)
    # ransdat.PlotNucEnergyGen2(xbl,xbr)
    # ransdat.PlotTrippleAlphaNucEnergyGen(xbl,xbr)

    # ransdat.plot_lin_q1q2(xbl,xbr,'dd','tt',\
    #                      r'r (cm)',\
    #                      r'log $\rho$ (g cm$^{-3}$)',r'log T (K)',\
    #                      r'$\rho$',r'$T$')

    # ransdat.plot_log_q1q2(xbl,xbr,'enuc1','ei',\
    #                       r'r (cm)',\
    #                       r'$\varepsilon_{nuc}$ (erg $s^{-1}$)',\
    #                       r'$\epsilon$ (ergs)',\
    #                       r'$\varepsilon_{nuc}$',r'$\epsilon$')


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END
