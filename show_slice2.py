import UTILS.PROMPI.PROMPI_data as prd
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os


dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
#filename_blck = dataloc+'ccptwo.res128cubed.fixedmu.opto3.01014.bindata'

bindata = [filee for filee in sorted(os.listdir(dataloc)) if "bindata" in filee]

for filename in bindata:
    print(filename)

    dat = ['temp']

    block = prd.PROMPI_bindata(dataloc+filename,dat)

    #grid = block.grid()

    #nx = grid['nx']
    #ny = grid['ny']
    #nz = grid['nz']

    #xzn0 = grid['xzn0']
    #yzn0 = grid['yzn0']
    #zzn0 = grid['zzn0']

    nx =  block.datadict['qqx']
    ny = block.datadict['qqy']
    nz = block.datadict['qqz']

    xzn0 = block.datadict['xzn0']
    yzn0 = block.datadict['yzn0']
    zzn0 = block.datadict['zzn0']

    time = block.datadict['time']

    #velx = block.datadict['velx']
    temp = block.datadict['temp']
    #rho  = block.datadict['density']

    sterad = np.ones((nz,ny))
    steradtot = np.sum(sterad)

    for i in range(nx):
        htemp = temp[i][:][:]
        eh_temp = np.sum(htemp*sterad)/steradtot
        temp_r = htemp-eh_temp
        temp[i][:][:] = temp_r[:][:]

    mtemp = 5.e6
    temp[temp > mtemp] = mtemp
    temp[temp < -mtemp] = -mtemp

    temp = np.asarray(temp[100][:][:])

    #print(nx, ny, nz)
    #print(yzn0[0],yzn0[nx-1])

    vmaxv = 5.e6
    vminv = -5.e6

    # create FIGURE
    plt.figure(figsize=(7, 6))

    im = plt.imshow(temp, interpolation='bilinear', cmap=cm.inferno,
                    origin='lower', extent=[yzn0[0], yzn0[ny-1], zzn0[0], zzn0[nz-1]],
                    vmax=vmaxv, vmin=vminv)

    plt.colorbar()
    plt.title("TempFlct - time: " + str(time) + " s")

    #plt.show(block=False)

    # save PLOT
    plt.savefig('RESULTS/'+ filename + '_ccptwo_temp3D.png')