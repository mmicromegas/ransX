###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_reaclib.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: August/2019
# Desc: compare transport and nuclear timescales
# Usage: run ransX_reaclib.py

import UTILS.RANSX.Properties as pRop
import UTILS.REACLIB.ReadParamsReaclib as rP
import UTILS.REACLIB.ReaclibMasterPlot as mPlot
import ast
import os


def main():
    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.reaclib')
    params = rP.ReadParamsReaclib(paramFile)

    # get input parameters
    filename = params.getForProp('prop')['eht_data']
    ig = params.getForProp('prop')['ig']
    ieos = params.getForProp('prop')['ieos']
    intc = params.getForProp('prop')['intc']
    laxis = params.getForProp('prop')['laxis']
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']

    # calculate properties
    ransP = pRop.Properties(filename, ig, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

    # instantiate master plot
    plt = mPlot.ReaclibMasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

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
