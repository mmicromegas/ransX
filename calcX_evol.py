###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: calcX_evol.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: December/2020
# Desc: calculates temporal evolution of several quantities based on RANS equations
# Usage: run calcX_evol.py

import numpy as np
import sys
import os

from UTILS.EVOL.CalcxReadParams import CalcxReadParams
from UTILS.RANSX.Properties import Properties
from UTILS.Tools import Tools
from UTILS.Errors import Errors


def main():
    # check python version
    if sys.version_info[0] < 3:
        print("Python " + str(sys.version_info[0]) + "  is not supported. EXITING.")
        sys.exit()

    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.calcx')
    params = CalcxReadParams(paramFile)

    # get input parameters
    filename = params.getForProp('prop')['eht_data']
    plabel = params.getForProp('prop')['plabel']
    dataout = params.getForProp('prop')['dataout']
    ig = params.getForProp('prop')['ig']
    nsdim = params.getForProp('prop')['nsdim']
    ieos = params.getForProp('prop')['ieos']
    laxis = params.getForProp('prop')['laxis']

    # instantiate tools
    tools = Tools()

    # load data to structured array
    eht = tools.customLoad(filename)

    # load available central times
    t_timec = tools.getRAdata(eht, 'timec')

    # check imposed boundary limits
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']
    xzn0 = tools.getRAdata(eht, 'xzn0')

    # instantiate errors
    errors = Errors()

    if (xbl < xzn0[0]) or (xbr > xzn0[-1]):
        print(xbl, xbr, xzn0[0], xzn0[-1])
        print("ERROR(calcX_evol.py):" + errors.errorOutOfBoundary())
        sys.exit()

    t_xzn0inc, t_xzn0outc, t_TKEsum, t_epsD, t_tD, t_urms = [], [], [], [], [], []
    t_tc, t_tenuc, t_pturb_o_pgas, t_machMax, t_machMean = [], [], [], [], []
    t_resContMean, t_resContMax, t_resTeeMean, t_resTeeMax = [], [], [], []
    t_x0002mean_cnvz = []

    intc = 0
    for intc in range(0, t_timec.size):
        ransP = Properties(filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
        prp = ransP.properties()
        t_xzn0inc.append(prp['xzn0inc'])
        t_xzn0outc.append(prp['xzn0outc'])
        t_TKEsum.append(prp['TKEsum'])
        t_epsD.append(prp['epsD'])
        t_tD.append(prp['tD'])
        t_tc.append(prp['tc'])
        t_tenuc.append(prp['tenuc'])
        t_pturb_o_pgas.append(prp['pturb_o_pgas'])
        t_machMax.append(prp['machMax_1'])
        t_machMean.append(prp['machMean_1'])
        t_urms.append(prp['urms'])
        t_resContMean.append(prp['resContMean'])
        t_resContMax.append(prp['resContMax'])
        t_resTeeMean.append(prp['resTeeMean'])
        t_resTeeMax.append(prp['resTeeMax'])
        t_x0002mean_cnvz.append(prp['x0002mean_cnvz'])
        print(prp['xzn0inc'], prp['xzn0outc'], prp['resContMean'], prp['resContMax'], prp['resTeeMean'],
              prp['resTeeMax'])

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

    t_machMx = {'t_machMax': t_machMax}
    tevol.update(t_machMx)

    t_machMn = {'t_machMean': t_machMean}
    tevol.update(t_machMn)

    t_velrms = {'t_urms': t_urms}
    tevol.update(t_velrms)

    t_resCntMx = {'t_resContMax': t_resContMax}
    tevol.update(t_resCntMx)

    t_resCntMn = {'t_resContMean': t_resContMean}
    tevol.update(t_resCntMn)

    t_resTeeMx = {'t_resTeeMax': t_resTeeMax}
    tevol.update(t_resTeeMx)

    t_resTeeMn = {'t_resTeeMean': t_resTeeMean}
    tevol.update(t_resTeeMn)

    x0002mc = {'t_x0002mean_cnvz': t_x0002mean_cnvz}
    tevol.update(x0002mc)

    # get and store grid via redundant call to properties
    # otherwise you get "prp may be as assigned before reference"
    ransP = Properties(filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

    tavg = {'tavg': prp['tavg']}
    tevol.update(tavg)

    xznl = {'xznl': prp['xznl']}
    tevol.update(xznl)

    xznr = {'xznr': prp['xznr']}
    tevol.update(xznr)

    nx = {'nx': prp['nx']}
    tevol.update(nx)

    ny = {'ny': prp['ny']}
    tevol.update(ny)

    nz = {'nz': prp['nz']}
    tevol.update(nz)

    # STORE dictionary tevol
    np.save(dataout, tevol)


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END
