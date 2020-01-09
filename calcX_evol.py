###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: calcX_evol.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: November/2019
# Desc: calculates temporal evolution of several quantities based on RANS equations
# Usage: run calcX_evol.py

import numpy as np
import sys
import os
import UTILS.EVOL.EvolReadParams as rP
import UTILS.RANSX.Properties as pRop
import UTILS.Tools as uT
import UTILS.Errors as eR


def main():
    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.evol')
    params = rP.EvolReadParams(paramFile)

    # get input parameters
    filename = params.getForProp('prop')['eht_data']
    dataout = params.getForProp('prop')['dataout']
    ig = params.getForProp('prop')['ig']
    ieos = params.getForProp('prop')['ieos']
    laxis = params.getForProp('prop')['laxis']

    # load data to structured array
    eht = np.load(filename)

    # instantiate tools
    tools = uT.Tools()

    # load available central times
    t_timec = tools.getRAdata(eht, 'timec')

    # check imposed boundary limits
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']
    xzn0 = tools.getRAdata(eht, 'xzn0')

    # instantiate errors
    errors = eR.Errors()

    if (xbl < xzn0[0]) or (xbr > xzn0[-1]):
        print(xbl, xbr, xzn0[0], xzn0[-1])
        print("ERROR(calcX_evol.py):" + errors.errorOutOfBoundary())
        sys.exit()

    # TODO: add size of cnvz in Hp, RMS velocities, convective turnover timescales #
    # TODO: add here amplitude of residuals in the convection zone (to find out optimum tavg)

    t_xzn0inc, t_xzn0outc, t_TKEsum, t_epsD, t_tD = [], [], [], [], []
    t_tc, t_tenuc, t_pturb_o_pgas, t_machMax, t_machMean = [], [], [], [], []
    t_x0002mean_cnvz = []

    intc = 0
    for intc in range(0, t_timec.size):
        ransP = pRop.Properties(filename, ig, ieos, intc, laxis, xbl, xbr)
        prp = ransP.properties()
        t_xzn0inc.append(prp['xzn0inc'])
        t_xzn0outc.append(prp['xzn0outc'])
        t_TKEsum.append(prp['TKEsum'])
        t_epsD.append(prp['epsD'])
        t_tD.append(prp['tD'])
        t_tc.append(prp['tc'])
        t_tenuc.append(prp['tenuc'])
        t_pturb_o_pgas.append(prp['pturb_o_pgas'])
        t_machMax.append(prp['machMax'])
        t_machMean.append(prp['machMean'])
        t_x0002mean_cnvz.append(prp['x0002mean_cnvz'])
        print(prp['xzn0inc'], prp['xzn0outc'])

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

    x0002mc = {'t_x0002mean_cnvz': t_x0002mean_cnvz}
    tevol.update(x0002mc)

    # get and store grid via redundant call to properties
    ransP = pRop.Properties(filename, ig, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

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
