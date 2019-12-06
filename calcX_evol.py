# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: November/2019

import numpy as np
import UTILS.EVOL.EvolReadParams as rp
import UTILS.EVOL.EvolMasterPlot as plot
import UTILS.EVOL.PropertiesEvolution as propevol
import ast 
import sys

paramFile = 'param.evol'
params = rp.EvolReadParams(paramFile)

filename = params.getForProp('prop')['eht_data']
dataout = params.getForProp('prop')['dataout']

# load data to structured array
eht = np.load(filename)	

# load availabe central times
t_timec   = np.asarray(eht.item().get('timec')) 
ntc   = np.asarray(eht.item().get('ntc')) 


# check imposed boundary limits 
xbl = params.getForProp('prop')['xbl']
xbr = params.getForProp('prop')['xbr']
xzn0 = np.asarray(eht.item().get('xzn0'))

if ((xbl < xzn0[0]) or (xbr > xzn0[-1])):
    #print(xbl,xbr,xzn0[0],xzn0[-1])
    print("ERROR(calcX_evol.py): imposed boundary limit in param.evol exceeds the grid limits.")
    sys.exit()

t_xzn0inc = []
t_xzn0outc = []	
t_TKEsum = []
t_epsD = []
t_tD = []
t_tc = [] 
t_tenuc = []
t_pturb_o_pgas = []	
t_x0002mean_cnvz = []	
	
#print(t_timec.size,ntc)		
		
for intc in range(0,t_timec.size):	
    ransPevol = propevol.PropertiesEvolution(params,intc)
    properties = ransPevol.execute()
    t_xzn0inc.append(properties['xzn0inc'])
    t_xzn0outc.append(properties['xzn0outc'])
    t_TKEsum.append(properties['TKEsum'])
    t_epsD.append(properties['epsD'])
    t_tD.append(properties['tD'])
    t_tc.append(properties['tc'])
    t_tenuc.append(properties['tenuc'])	
    t_pturb_o_pgas.append(properties['pturb_o_pgas'])
    t_x0002mean_cnvz.append(properties['x0002mean_cnvz'])	
    print(properties['xzn0inc'],properties['xzn0outc'])


# store time-evolution
tevol = {}

t_c = {'t_timec': t_timec}
tevol.update(t_c)

t_x0inc = {'t_xzn0inc': t_xzn0inc}
tevol.update(t_x0inc)

t_x0outc = {'t_xzn0outc': t_xzn0outc}
tevol.update(t_x0outc)

t_turbke = {'t_TKEsum': t_TKEsum}
tevol.update(t_turbke)

t_epsilonD = {'t_epsD': t_epsD}
tevol.update(t_epsilonD)

t_dissTimeScale = {'t_tD': t_tD}
tevol.update(t_dissTimeScale)

t_cturn = {'t_tc': t_tc}
tevol.update(t_cturn)

t_nucl = {'t_tenuc': t_tenuc}
tevol.update(t_nucl)

t_pt_o_pg = {'t_pturb_o_pgas': t_pturb_o_pgas}
tevol.update(t_pt_o_pg)

xznl = {'xznl' : properties['xznl']}
tevol.update(xznl)

xznr = {'xznr' : properties['xznr']}
tevol.update(xznr)

x0002mc = {'t_x0002mean_cnvz' : t_x0002mean_cnvz}
tevol.update(x0002mc)

# STORE 
np.save(dataout,tevol)

# END
