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

    # MACH NUMBER MAX EVOLUTION
    if str2bool(params.getForEvol('machmxevol')['plotMee']):
        plt.execEvolMachMax()

    # MACH NUMBER MEAN EVOLUTION
    if str2bool(params.getForEvol('machmeevol')['plotMee']):
        plt.execEvolMachMean()

    # CONVECTION BOUNDARIES POSITION EVOLUTION
    if str2bool(params.getForEvol('cnvzbndry')['plotMee']):
        plt.execEvolCNVZbnry()

    # TOTAL ENERGY SOURCE TERM EVOLUTION
    if str2bool(params.getForEvol('enesource')['plotMee']):
        plt.execEvolTenuc()

    # CONVECTIVE RMS VELOCITIES
    if str2bool(params.getForEvol('convelrms')['plotMee']):
        plt.execEvolConvVelRMS()

    # CONVECTIVE TURNOVER TIMESCALE
    if str2bool(params.getForEvol('convturn')['plotMee']):
        plt.execEvolConvTurnoverTime()

    # MEAN RESIDUAL IN CONTINUITY EQUATION WITH MASS FLUX
    if str2bool(params.getForEvol('contresmean')['plotMee']):
        plt.execContResMean()

    # MAX RESIDUAL IN CONTINUITY EQUATION WITH MASS FLUX
    if str2bool(params.getForEvol('contresmax')['plotMee']):
        plt.execContResMax()

    # MEAN RESIDUAL IN TOTAL ENERGY EQUATION
    if str2bool(params.getForEvol('teeresmean')['plotMee']):
        plt.execTeeResMean()

    # MAX RESIDUAL IN TOTAL ENERGY EQUATION
    if str2bool(params.getForEvol('teeresmax')['plotMee']):
        plt.execTeeResMax()

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
