###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: January/2019

import UTILS.RES.ResProperties as prop
import UTILS.RES.ResReadParamsRansX as rp
import UTILS.RES.ResMasterPlot as plot
import ast 	

paramFile = 'param.res'
params = rp.ResReadParamsRansX(paramFile)

# get list with data source files
filename = params.getForProp('prop')['eht_data']

# get simulation properties
eht = []		
for file in filename:
    ransP = prop.ResProperties(params,file)
    properties = ransP.execute()

# instantiate master plot 								 
plt = plot.ResMasterPlot(params)								 

# obtain publication quality figures
plt.SetMatplotlibParams()

# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)

# PLOT

# TEMPERATURE
if str2bool(params.getForEqs('temp')['plotMee']): plt.execTemp()

# BRUNT-VAISALLA FREQUENCY
if str2bool(params.getForEqs('nsq')['plotMee']): plt.execBruntV()

# TURBULENT KINETIC ENERGY
if str2bool(params.getForEqs('tkie')['plotMee']): plt.execTke()

# INTERNAL ENERGY FLUX
if str2bool(params.getForEqs('eintflx')['plotMee']): plt.execEiFlx()

# PRESSURE FLUX
if str2bool(params.getForEqs('pressxflx')['plotMee']): plt.execPPxflx()

# TEMPERATURE FLUX EQUATION
if str2bool(params.getForEqs('tempflx')['plotMee']): plt.execTTflx()

# ENTHALPY FLUX EQUATION
if str2bool(params.getForEqs('enthflx')['plotMee']): plt.execHHflx()

# ENTROPY FLUX
if str2bool(params.getForEqs('entrflx')['plotMee']): plt.execSSflx()

# TURBULENT MASS FLUX
if str2bool(params.getForEqs('tmsflx')['plotMee']): plt.execTMSflx()

# load network
network = params.getNetwork() 

# COMPOSITION MASS FRACTION AND FLUX
for elem in network[1:]: # skip network identifier in the list 
    inuc = params.getInuc(network,elem) 	
    if str2bool(params.getForEqs('xrho_'+elem)['plotMee']): plt.execXrho(inuc,elem,'xrho_'+elem)
    if str2bool(params.getForEqs('xflxx_'+elem)['plotMee']): plt.execXflxx(inuc,elem,'xflxx_'+elem)	




