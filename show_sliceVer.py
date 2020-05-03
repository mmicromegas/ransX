import UTILS.PROMPI.PROMPI_data as prd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os

# dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
# filename_blck = dataloc+'ccptwo.res128cubed.fixedmu.opto3.01014.bindata'
dataloc = 'D:\\simonX\\galileo\\'

bindata = [filee for filee in sorted(os.listdir(dataloc)) if "bindata" in filee]

for filename in bindata:
    print(filename)

    dat = ['temp', 'velx', 'density', '0002']

    lhc = 4.e8

    block = prd.PROMPI_bindata(dataloc + filename, dat)

    # grid = block.grid()

    # nx = grid['nx']
    # ny = grid['ny']
    # nz = grid['nz']

    # xzn0 = grid['xzn0']
    # yzn0 = grid['yzn0']
    # zzn0 = grid['zzn0']

    nx = block.datadict['qqx']
    ny = block.datadict['qqy']
    nz = block.datadict['qqz']

    xzn0 = block.datadict['xzn0']
    yzn0 = block.datadict['yzn0']
    zzn0 = block.datadict['zzn0']

    time = block.datadict['time']

    velx = block.datadict['velx']
    temp = block.datadict['temp']
    dens = block.datadict['density']
    x0002 = block.datadict['0002']

    xlm = np.abs(np.asarray(xzn0) - np.float(lhc))
    ilhc = int(np.where(xlm == xlm.min())[0][0])

    temp = np.asarray(temp[:,:,ilhc])
    vtmaxv = 1.3e9
    vtminv = 1.2e9
    temp[temp > vtmaxv] = vtmaxv
    temp[temp < vtminv] = vtminv

    velx = np.asarray(velx[:,:,ilhc])
    vvmaxv = 2.e7
    vvminv = -2.e7
    velx[velx > vvmaxv] = vvmaxv
    velx[velx < vvminv] = vvminv

    dens = np.asarray(dens[:,:,ilhc])
    vdmaxv = 0.4e6
    vdminv = 0.2e6
    dens[dens > vdmaxv] = vdmaxv
    dens[dens < vdminv] = vdminv

    x0002 = np.asarray(x0002[:,:,ilhc])
    vxmaxv = 0.9
    vxminv = 0.1
    x0002[x0002 > vxmaxv] = vxmaxv
    x0002[x0002 < vxminv] = vxminv

    # create FIGURE
    # plt.figure(figsize=(7, 6))

    # https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111

    fig = plt.figure(figsize=(14, 14))
    ax1 = fig.add_subplot(221)

    fig.suptitle("Temp (UP-LEFT), VelX (UP-RIGHT), Rho(DOWN-LEFT), X2(DOWN-RIGHT) - time: " + str(time) + " s, y = " + str(lhc) + " cm")
    im1 = ax1.imshow(temp, interpolation='bilinear', cmap=cm.inferno,
                     origin='lower', extent=[yzn0[0], yzn0[ny - 1], xzn0[0], xzn0[nx - 1]],
                     vmax=vtmaxv, vmin=vtminv)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(222)

    im2 = ax2.imshow(velx, interpolation='bilinear', cmap=cm.bwr,
                     origin='lower', extent=[yzn0[0], yzn0[ny - 1], xzn0[0], xzn0[nx - 1]],
                     vmax=vvmaxv, vmin=vvminv)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical');

    ax3 = fig.add_subplot(223)

    im3 = ax3.imshow(dens, interpolation='bilinear', cmap=cm.copper,
                     origin='lower', extent=[yzn0[0], yzn0[ny - 1], xzn0[0], xzn0[nx - 1]],
                     vmax=vdmaxv, vmin=vdminv)

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')

    ax4 = fig.add_subplot(224)

    # if you want to reverse colormap add _r at the end
    im4 = ax4.imshow(x0002, interpolation='bilinear', cmap=cm.binary_r,
                     origin='lower', extent=[yzn0[0], yzn0[ny - 1], xzn0[0], xzn0[nx - 1]],
                     vmax=vxmaxv, vmin=vxminv)

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax, orientation='vertical')

    plt.show(block=False)

    # save PLOT
    plt.savefig('RESULTS/' + filename + '_ccptwo_2DcutsVer.png')

    #plt.close('all')
