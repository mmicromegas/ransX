###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_evol.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: December/2020
# Desc: resolution study of temporal evolution
# Usage: run ransX_evol.py


from UTILS.EVOL.EvolReadParams import EvolReadParams
from UTILS.EVOL.EvolMasterPlot import EvolMasterPlot
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
    paramFile = os.path.join('PARAMS', 'param.evol')
    params = EvolReadParams(paramFile)

    # instantiate master plot
    plt = EvolMasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # check/create RESULTS folder
    try:
        os.makedirs('RESULTS')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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
