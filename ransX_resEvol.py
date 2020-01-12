###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_resEvol.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: November/2019
# Desc: resolution study of temporal evolution
# Usage: run ransX_resEvol.py


import UTILS.EVOL.FOR_RESOLUTION_STUDY.ResEvolReadParams as rP
import UTILS.EVOL.FOR_RESOLUTION_STUDY.ResEvolMasterPlot as mPlot
import ast
import os


def main():
    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.evol.res')
    params = rP.ResEvolReadParams(paramFile)

    # instantiate master plot
    plt = mPlot.ResEvolMasterPlot(params)

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
