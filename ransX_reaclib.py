###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: August/2019

import EQUATIONS.Properties as prop
import UTILS.ReadParamsReaclib as rp
#import UTILS.REACLIB_data as rd
#import UTILS.TRANSPORT_data as td
import UTILS.ReaclibMasterPlot as plot
import ast

paramFile = 'param.reaclib'
params = rp.ReadParamsReaclib(paramFile)

# get RANS data source file
filename_rans = params.getForProp('prop')['eht_data']

# get simulation properties
ransP = prop.Properties(params,filename_rans)
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
for elem in network[1:]: # skip network identifier in the list 
    inuc = params.getInuc(network,elem) 	
    if str2bool(params.getForEqs('xTimescales_'+elem)['plotMee']): plt.execXtransportVSnuclearTimescales(inuc,elem,'xTimescales_'+elem,properties['xzn0inc'],properties['xzn0outc'],properties['tc'])


# get nuclear rate 
#nr_ne20 = reaclib.get_nuclear_rate_per_reaction("ne20") # this is a dictionary :: nameOfReaction/rate

# get nuclear timescale
#t_nucl_ne20 = reaclib.get_nuclear_timescale(nr_ne20)

# get transport timescale
#t_trans_ne20 = ranstse.get_transport_timescale("ne20")
 
# get convective turnover timescale
#tc = ranstse.get_tconv()

# get X-grid
#xgrid = ranstse.get_xgrid()

# plot transport timescale
#plt.plot(xgrid,t_trans_ne20)

# plot convective turnover timescale 
#plt.plot(xgrid,tc)

# plot nuclear timescales
#for i_ne20_nt in nt_ne20:
#    plt.plot(xgrid,i_ne20_nt)
