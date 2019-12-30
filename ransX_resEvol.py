# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: November/2019
from typing import Union

import UTILS.EVOL.FOR_RESOLUTION_STUDY.ResEvolReadParams as rp
import UTILS.EVOL.FOR_RESOLUTION_STUDY.ResEvolMasterPlot as plot
import UTILS.EVOL.PropertiesEvolution as propevol
import ast
import os

# create os independent path and read parameter file
paramFile = os.path.join('PARAMS', 'param.evol.res')
params = rp.ResEvolReadParams(paramFile)

# instantiate master plot
plt = plot.ResEvolMasterPlot(params)

# obtain publication quality figures
plt.SetMatplotlibParams()

# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)

# TURBULENT KINETIC ENERGY EVOLUTION
if str2bool(params.getForEvol('tkeevol')['plotMee']): plt.execEvolTKE()

# MACH NUMBER MAX EVOLUTION
if str2bool(params.getForEvol('machmxevol')['plotMee']): plt.execEvolMachMax()

# MACH NUMBER MEAN EVOLUTION
if str2bool(params.getForEvol('machmeevol')['plotMee']): plt.execEvolMachMean()

# CONVECTION BOUNDARIES POSITION EVOLUTION
if str2bool(params.getForEvol('cnvzbndry')['plotMee']): plt.execEvolCNVZbnry()

# TOTAL ENERGY SOURCE TERM EVOLUTION
if str2bool(params.getForEvol('enesource')['plotMee']): plt.execEvolTenuc()

# X0002 EVOLUTION
if str2bool(params.getForEvol('x0002evol')['plotMee']): plt.execEvolX0002()
