###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_canuto1997.py
# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: June/2020
# Desc: controls selection, hence calculation and plotting of terms in RANS equations
# Usage: run ransX_canuto1997.py

import UTILS.CANUTO1997.Properties as pRop
import UTILS.CANUTO1997.ReadParamsRansX as rP
import UTILS.CANUTO1997.MasterPlot as mPlot
import ast
import os


def main():
    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.canuto1997')
    params = rP.ReadParamsRansX(paramFile)

    # get input parameters
    filename = params.getForProp('prop')['eht_data']
    plabel = params.getForProp('prop')['plabel']
    ig = params.getForProp('prop')['ig']
    ieos = params.getForProp('prop')['ieos']
    intc = params.getForProp('prop')['intc']
    laxis = params.getForProp('prop')['laxis']
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']

    # calculate properties
    ransP = pRop.Properties(filename, plabel, ig, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

    # instantiate master plot
    plt = mPlot.MasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # CONTINUITY EQUATION
    if str2bool(params.getForEqs('rho')['plotMee']):
        plt.execRho(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('conteq')['plotMee']):
        plt.execContEq(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqsBar('conteqBar')['plotMee']):
        plt.execContEqBar()

    # MOMENTUM X EQUATION
    if str2bool(params.getForEqs('momex')['plotMee']):
        plt.execMomx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('momxeq')['plotMee']):
        plt.execMomxEq(prp['xzn0inc'], prp['xzn0outc'])

    # MOMENTUM Y EQUATION
    if str2bool(params.getForEqs('momey')['plotMee']):
        plt.execMomy(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('momyeq')['plotMee']):
        plt.execMomyEq(prp['xzn0inc'], prp['xzn0outc'])

    # MOMENTUM Z EQUATION
    if str2bool(params.getForEqs('momez')['plotMee']):
        plt.execMomz(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('momzeq')['plotMee']):
        plt.execMomzEq(prp['xzn0inc'], prp['xzn0outc'])

    # TURBULENT KINETIC ENERGY FLUX
    if str2bool(params.getForEqs('keflx')['plotMee']):
        plt.execKeflx(prp['xzn0inc'], prp['xzn0outc'])

# True/False strings to proper boolean
def str2bool(param):
    return ast.literal_eval(param)


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END
