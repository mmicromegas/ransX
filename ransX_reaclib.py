###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_reaclib.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: December/2020
# Desc: compare transport and nuclear timescales
# Usage: run ransX_reaclib.py

from UTILS.RANSX.Properties import Properties
from UTILS.REACLIB.ReadParamsReaclib import ReadParamsReaclib
from UTILS.REACLIB.ReaclibMasterPlot import ReaclibMasterPlot

import ast
import os
import sys
import errno


def main():
    # check python version
    if sys.version_info[0] < 3:
        print("Python " + str(sys.version_info[0]) + "  is not supported. EXITING.")
        sys.exit()

    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.reaclib')
    params = ReadParamsReaclib(paramFile)

    # get input parameters
    filename = params.getForProp('prop')['eht_data']
    ig = params.getForProp('prop')['ig']
    nsdim = params.getForProp('prop')['nsdim']
    plabel = params.getForProp('prop')['plabel']
    ieos = params.getForProp('prop')['ieos']
    intc = params.getForProp('prop')['intc']
    laxis = params.getForProp('prop')['laxis']
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']

    # calculate properties
    ransP = Properties(filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

    # instantiate master plot
    plt = ReaclibMasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # check/create RESULTS folder
    try:
        os.makedirs('RESULTS')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # load network
    network = params.getNetwork()

    # COMPARE TRANSPORT and NUCLEAR TIMESCALES
    for elem in network[1:]:  # skip network identifier in the list
        inuc = params.getInuc(network, elem)
        if str2bool(params.getForEqs('xTimescales_' + elem)['plotMee']):
            plt.execXtransportVSnuclearTimescales(inuc, elem, 'xTimescales_' + elem, prp['xzn0inc'], prp['xzn0outc'],
                                                  prp['tc'])


# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END
