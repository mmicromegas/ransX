###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: August/2019

import EQUATIONS.Properties as prop
import UTILS.REACLIB.ReadParamsReaclib as rp
import UTILS.REACLIB.ReaclibMasterPlot as plot
import ast

paramFile = 'param.reaclib'
params = rp.ReadParamsReaclib(paramFile)

# get simulation properties
ransP = prop.Properties(params)
properties = ransP.execute()

# instantiate master plot 								 
plt = plot.ReaclibMasterPlot(params)

# obtain publication quality figures
plt.SetMatplotlibParams()


# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)


# load network
network = params.getNetwork()

# COMPARE TRANSPORT and NUCLEAR TIMESCALES 
for elem in network[1:]:  # skip network identifier in the list
    inuc = params.getInuc(network, elem)
    if str2bool(params.getForEqs('xTimescales_' + elem)['plotMee']): plt.execXtransportVSnuclearTimescales(inuc, elem,
                                                                                                           'xTimescales_' + elem,
                                                                                                           properties[
                                                                                                               'xzn0inc'],
                                                                                                           properties[
                                                                                                               'xzn0outc'],
                                                                                                           properties[
                                                                                                               'tc'])
