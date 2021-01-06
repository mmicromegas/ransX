###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX_tseries.py
# Author: Miroslav Mocak
# Email: miroslav.mocak@gmail.com
# Date: December/2020
# Desc: calculates time-averages over tavg
# Usage: run ransX_tseries.py

from UTILS.PROMPI.PROMPI_data import PROMPI_ransdat
from UTILS.TSERIES.ReadParamsTseries import ReadParamsTseries
from UTILS.Errors import Errors
import numpy as np
import os
import sys


def main():
    # check python version
    if sys.version_info[0] < 3:
        print("Python " + str(sys.version_info[0]) + "  is not supported. EXITING.")
        sys.exit()

    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.tseries')
    params = ReadParamsTseries(paramFile)

    # read input parameters
    datadir = params.getForTseries('tseries')['datadir']
    endianness = params.getForTseries('tseries')['endianness']
    precision = params.getForTseries('tseries')['precision']

    dataout = params.getForTseries('tseries')['dataout']

    trange_beg = params.getForTseries('tseries')['trange_beg']
    trange_end = params.getForTseries('tseries')['trange_end']
    trange = [trange_beg, trange_end]

    tavg = params.getForTseries('tseries')['tavg']

    ransdat = [filee for filee in sorted(os.listdir(datadir)) if "ransdat" in filee]
    ransdat = [filee.replace(filee, datadir + filee) for filee in ransdat]

    filename = ransdat[0]
    ts = PROMPI_ransdat(filename, endianness, precision)

    time = []
    dt = []

    for filename in ransdat:
        print(filename)
        ts = PROMPI_ransdat(filename, endianness, precision)
        rans_tstart, rans_tend, rans_tavg = ts.rans_header()
        time.append(rans_tend)
        dt.append(rans_tavg)
        # print(rans_tend,rans_tavg)

    # convert to array
    time = np.asarray(time)
    dt = np.asarray(dt)
    nt = len(ransdat)

    print('Number of snapshots: ', nt)
    print('Available time range:', min(time), round(max(time), 3))
    print('Restrict data to time range:', trange[0], trange[1])

    # limit snapshot list to time range of interest
    idx = np.where((time > trange[0]) & (time < trange[1]))
    time = time[idx]
    dt = dt[idx]

    # time averaging window
    timecmin = min(time) + tavg / 2.0
    timecmax = max(time) - tavg / 2.0
    itc = np.where((time >= timecmin) & (time <= timecmax))
    timec = time[itc]
    ntc = len(timec)

    # print useful information
    print('Number of time averaged snapshots: ', ntc)
    print('Averaged time range: ', round(timecmin, 3), round(timecmax, 3))
    print('nx', ts.rans()['nx'])

    #   instantiate errors
    errors = Errors()
    if ntc == 0:
        print("----------")
        print("Error(rans_tseries.py) " + errors.errorAveragedSnapshots())
        print("----------")
        sys.exit()

    # READ IN DATA
    eh = []
    for i in idx[0]:
        filename = ransdat[i]
        ts = PROMPI_ransdat(filename, endianness, precision)
        field = [[data for data in ts.rans()[s]] for s in ts.ransl]
        eh.append(field)

    # eh = eh(r,time,quantity)
    # plt.plot(eh[:][nt-1][2])
    # plt.show(block=False)

    # TIME AVERAGING

    eht = {}

    for s in ts.ransl:
        idx = ts.ransl.index(s)
        tmp2 = []
        for i in range(ntc):
            itavg = np.where((time >= (timec[i] - tavg / 2.)) & (time <= (timec[i] + tavg / 2.)))
            sumdt = np.sum(dt[itavg])
            tmp1 = np.zeros(ts.rans()['nx'])
            for j in itavg[0]:
                tmp1 += np.asarray(eh[:][j][idx]) * dt[j]
            tmp2.append(tmp1 / sumdt)
        field = {str(s): tmp2}
        eht.update(field)

    # store grid
    nx = {'nx': ts.rans()['nx']}
    eht.update(nx)

    ny = {'ny': ts.rans()['ny']}
    eht.update(ny)

    nz = {'nz': ts.rans()['nz']}
    eht.update(nz)

    xzn0 = {'xzn0': ts.rans()['xzn0']}
    eht.update(xzn0)

    xznl = {'xznl': ts.rans()['xznl']}
    eht.update(xznl)

    xznr = {'xznr': ts.rans()['xznr']}
    eht.update(xznr)

    yzn0 = {'yzn0': ts.rans()['yzn0']}
    eht.update(yzn0)

    yznl = {'yznl': ts.rans()['yznl']}
    eht.update(yznl)

    yznr = {'yznr': ts.rans()['yznr']}
    eht.update(yznr)

    zzn0 = {'zzn0': ts.rans()['zzn0']}
    eht.update(zzn0)

    zznl = {'zznl': ts.rans()['zznl']}
    eht.update(zznl)

    zznr = {'zznr': ts.rans()['zznr']}
    eht.update(zznr)

    nnuc = {'nnuc': ts.rans()['nnuc']}
    eht.update(nnuc)

    ntc = {'ntc': ntc}
    eht.update(ntc)

    # store central times
    timec = {'timec': timec}
    eht.update(timec)

    tavg = {'tavg': tavg}
    eht.update(tavg)

    # store time-averaging window
    trange = {'trange': trange}
    eht.update(trange)

    # STORE TIME-AVERAGED DATA i.e the EHT dictionary   
    np.save(dataout + '.npy', eht)

    # STORE TIME-AVERAGED DATA i.e the EHT dictionary
    # fix_imports=True does not fix python 2 vs python 3 compatibility due to unsupported protocols in python 2 greater than 3
    # np.save(dataout + '.npy', eht, allow_pickle=True, fix_imports=True )


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END

# OBSOLETE CODE 

# print(ts.ransl)

# fld = 'ux'
# a = eht[fld]
# intc = ntc - 1
# intc = 11
# b = a[:][intc]
# xx = eht['xzn0']

# print(b)

# fig, ax1 = plt.subplots(figsize=(7,6))

# for i in range(nt):
#    filename = ransdat[i]
#    ts = pt.PROMPI_ransdat(filename,'little_endian','double')
#    plt.plot(xx,ts.rans()[fld])

# ax1.plot(xx,b,color='k')

# plt.show()
