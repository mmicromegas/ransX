import PROMPI_data as pt
import numpy as np
import os
import sys
import matplotlib.pyplot as plt       
	   
#datadir = 'C:\Users\mmocak\Desktop\simonX\MREZ\\'
datadir = 'C:\Users\mmocak\Desktop\cyrilX\\'
dataout = 'DATA\\tseries_ransout_nelrez'

#trange = [210. ,650.]
#tavg = 430.

#trange = [90. ,198.]
#tavg = 100.

trange = [200. ,450.]
tavg = 200.

ransdat = [file for file in os.listdir(datadir) if "ransdat" in file]
ransdat = [file.replace(file,datadir+file) for file in ransdat]	

filename = ransdat[0]
ts = pt.PROMPI_ransdat(filename)

qqx = ts.rans_qqx()
qqy = ts.rans_qqy()
qqz = ts.rans_qqz()

xznl = ts.rans_xznl()
xznr = ts.rans_xznr()

ransl = ts.rans_list()
		
nstep = []
time  = []
dt  = []	
		
for filename in ransdat:
    print(filename)
    ts = pt.PROMPI_ransdat(filename)
    rans_tstart, rans_tend, rans_tavg = ts.rans_header()
    time.append(rans_tend)
    dt.append(rans_tavg)

# convert to array

time = np.asarray(time)
dt = np.asarray(dt)		
nt = len(ransdat)

print('Numer of snapshots: ', nt)
print('Available time range:',min(time),round(max(time),3))
print('Restrict data to time range:', trange[0],trange[1])

#   limit snapshot list to time range of interest

idx = np.where((time > trange[0]) & (time < trange[1]))
time = time[idx]
dt = dt[idx]		

#  time averaging window

timecmin = min(time)+tavg/2.0
timecmax = max(time)-tavg/2.0
itc      = np.where((time >= timecmin) & (time <= timecmax))
timec    = time[itc]
ntc      = len(timec)

print('Number of time averaged snapshots: ', ntc)
print('Averaged time range: ',round(timecmin,3), round(timecmax,3))
print('qqx',qqx)

if ntc == 0:
    print("----------")
    print("rans_tseries.py ERROR: Zero time-averaged snapshots.")
    print("rans_tseries.py Adjust your trange and averaging window.")
    print("rans_tseries.py EXITING ... ")
    print("----------")
    sys.exit()

# READ IN DATA

eh = []
for i in idx[0]:
    filename = ransdat[i]
    ts = pt.PROMPI_ransdat(filename) 
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
        itavg = np.where((time >= (timec[i]-tavg/2.)) & (time <= (timec[i]+tavg/2.)))
        sumdt = np.sum(dt[itavg])
        tmp1 = np.zeros(qqx)
        for j in itavg[0]:   
            tmp1 += np.asarray(eh[:][j][idx])*dt[j]
        tmp2.append(tmp1/sumdt)
    field = {str(s) : tmp2}  		
    eht.update(field)     

# store radial grid 
	
grid = {'xzn0' : ts.rans()['xzn0']}
eht.update(grid)

xznl = {'xznl' : xznl}
eht.update(xznl)

xznr = {'xznr' : xznr}
eht.update(xznr)

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

# store number of grid points in simulation

nx = {'nx': qqx}
eht.update(nx)

ny = {'ny': qqy}
eht.update(ny)

nz = {'nz': qqz}
eht.update(nz)


# STORE TIME-AVERAGED DATA i.e the EHT dictionary 

np.save(dataout+'.npy',eht)

# END

# OBSOLETE CODE 
	
#print(ts.ransl)

#fld = 'enuc1'
#a = eht[fld]
#intc = ntc - 1
#b = a[:][intc]
#xx = eht['rr']

#print(b)

#fig, ax1 = plt.subplots(figsize=(7,6))

#for i in range(nt):
#    filename = ransdat[i]
#    ts = pt.PROMPI_ransdat(filename)
#    plt.plot(xx,ts.rans()[fld])

#ax1.plot(xx,b,color='k')

