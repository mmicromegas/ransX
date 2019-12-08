# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: November/2019

import UTILS.EVOL.EvolReadParams as rp
import UTILS.EVOL.EvolMasterPlot as plot
import UTILS.EVOL.PropertiesEvolution as propevol
import ast 

paramFile = 'param.evol'
params = rp.EvolReadParams(paramFile)

# instantiate master plot  	 
plt = plot.EvolMasterPlot(params) 

# obtain publication quality figures
plt.SetMatplotlibParams()

# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)

# PLOT

# TURBULENT KINETIC ENERGY EVOLUTION
if str2bool(params.getForEvol('tkeevol')['plotMee']): plt.execEvolTKE()

# CONVECTION BOUNDARIES POSITION EVOLUTION
if str2bool(params.getForEvol('cnvzbndry')['plotMee']): plt.execEvolCNVZbnry()

# TOTAL ENERGY SOURCE TERM EVOLUTION
if str2bool(params.getForEvol('enesource')['plotMee']): plt.execEvolTenuc()

# X0002 EVOLUTION
if str2bool(params.getForEvol('x0002evol')['plotMee']): plt.execEvolX0002()