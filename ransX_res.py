###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: January/2019

import EQUATIONS.ResProperties as prop
import UTILS.ResReadParamsRansX as rp
import UTILS.ResMasterPlot as plot
import ast 

paramFile = 'param.res'
params = rp.ResReadParamsRansX(paramFile)

# get simulation properties
#ransP = prop.ResProperties(params)
#properties = ransP.execute()

# instantiate master plot 								 
plt = plot.ResMasterPlot(params)								 

# obtain publication quality figures
plt.SetMatplotlibParams()

# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)

# PLOT

# BRUNT-VAISALLA FREQUENCY
if str2bool(params.getForEqs('nsq')['plotMee']): plt.execBruntV()