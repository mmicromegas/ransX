###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_evol.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: November/2019
# Desc: plot time-evolution based on results from calcX_evol.py
# Usage: run ransX_evol.py

import UTILS.EVOL.EvolReadParams as rP
import UTILS.EVOL.EvolMasterPlot as mPlot
import ast
import os


def main():
    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.evol')
    params = rP.EvolReadParams(paramFile)

    # instantiate master plot
    plt = mPlot.EvolMasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # TURBULENT KINETIC ENERGY EVOLUTION
    if str2bool(params.getForEvol('tkeevol')['plotMee']):
        plt.execEvolTKE()

    # CONVECTION BOUNDARIES POSITION EVOLUTION
    if str2bool(params.getForEvol('cnvzbndry')['plotMee']):
        plt.execEvolCNVZbnry()

    # TOTAL ENERGY SOURCE TERM EVOLUTION
    if str2bool(params.getForEvol('enesource')['plotMee']):
        plt.execEvolTenuc()

    # X0002 EVOLUTION
    if str2bool(params.getForEvol('x0002evol')['plotMee']):
        plt.execEvolX0002()


# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END
