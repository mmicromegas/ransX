import UTILS.PROMPI.PROMPI_data as prd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


def SetMatplotlibParams():
    """ This routine sets some standard values for matplotlib """
    """ to obtain publication-quality figures """

    # plt.rc('text',usetex=True)
    # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
    plt.rc('font', size=16.)
    plt.rc('lines', linewidth=2, markeredgewidth=2., markersize=12)
    plt.rc('axes', linewidth=1.5)
    plt.rcParams['xtick.major.size'] = 8.
    plt.rcParams['xtick.minor.size'] = 8.
    plt.rcParams['figure.subplot.bottom'] = 0.15
    plt.rcParams['figure.subplot.left'] = 0.17
    plt.rcParams['figure.subplot.right'] = 0.85
    plt.rcParams.update({'figure.max_open_warning': 0})

def getFileID(ii):
    iid = 0
    if ii < 10:
        iid = '0000' + str(ii)
    if 10 <= ii < 100:
        iid = '000' + str(ii)
    if 100 <= ii < 1000:
        iid = '00' + str(ii)
    if 1000 <= ii < 10000:
        iid = '0' + str(ii)
    return iid

SetMatplotlibParams()

# dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
# filename_blck = dataloc+'ccptwo.res128cubed.fixedmu.opto3.01014.bindata'
dataloc3D = "D:\\ransX\\DATA_D\\BINDATA\\ccptwo_dev\\test\\3D\\"
dataloc2D = "D:\\ransX\\DATA_D\\BINDATA\\ccptwo_dev\\test\\2D\\"

#bindata2D = [filee for filee in sorted(os.listdir(dataloc2D)) if "bindata" in filee]
#bindata3D = [filee for filee in sorted(os.listdir(dataloc3D)) if "bindata" in filee]

dat2D = ['velx', 'vely']
dat3D = ['velx', 'vely', 'velz']


tkemaxv = 15.
tkeminv = 12.5

for ii in range(363,364):

    filename2D = "ccptwo.2D.r512.dev."+ getFileID(ii)+".bindata"
    filename3D = "ccptwo.r512x512x512.cosma."+ getFileID(ii)+".bindata"

    block2D = prd.PROMPI_bindata(dataloc2D + filename2D, dat2D)

    nx_2D = block2D.datadict['qqx']
    ny_2D = block2D.datadict['qqy']
    nz_2D = block2D.datadict['qqz']

    xzn0_2D = block2D.datadict['xzn0']
    yzn0_2D = block2D.datadict['yzn0']
    zzn0_2D = block2D.datadict['zzn0']

    time_2D = block2D.datadict['time']

    velx_2D = block2D.datadict['velx']
    vely_2D = block2D.datadict['vely']

    ilhc = 0
    velx2D = np.asarray(velx_2D[:,:,ilhc])
    vely2D = np.asarray(vely_2D[:,:,ilhc])

    tke2D = 0.5*(velx2D**2. + vely2D**2.)
    tke2D = np.log10(tke2D)

    block3D = prd.PROMPI_bindata(dataloc3D + filename3D, dat3D)

    nx_3D = block3D.datadict['qqx']
    ny_3D = block3D.datadict['qqy']
    nz_3D = block3D.datadict['qqz']

    xzn0_3D = block3D.datadict['xzn0']
    yzn0_3D = block3D.datadict['yzn0']
    zzn0_3D = block3D.datadict['zzn0']

    time_3D = block3D.datadict['time']

    velx_3D = block3D.datadict['velx']
    vely_3D = block3D.datadict['vely']
    velz_3D = block3D.datadict['velz']

    lhc = 8.e8
    xlm = np.abs(np.asarray(xzn0_3D) - np.float(lhc))
    ilhc = int(np.where(xlm == xlm.min())[0][0])

    velx3D = np.asarray(velx_3D[:,:,ilhc])
    vely3D = np.asarray(vely_3D[:,:,ilhc])
    velz3D = np.asarray(velz_3D[:,:,ilhc])

    tke3D = 0.5*(velx3D**2. + vely3D**2. + velz3D**2.)
    tke3D = np.log10(tke3D)

    yb = 6.5e8
    xlm = np.abs(np.asarray(xzn0_2D) - np.float(yb))
    ib = int(np.where(xlm == xlm.min())[0][0])

    yt = 9.5e8
    xlm = np.abs(np.asarray(xzn0_2D) - np.float(yt))
    it = int(np.where(xlm == xlm.min())[0][0])

    # override
    ib = 0
    it = nx_2D-1

    print(ib,it)

    #ib = 20
    #it = 100

    # create FIGURE
    # plt.figure(figsize=(7, 6))

    # https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111

    # fig = plt.figure(figsize=(14, 14))
    fig = plt.figure(figsize=(7, 7))

    #spec = gridspec.GridSpec(ncols=2, nrows=2,
    #                         width_ratios=[2, 2])

    #ax1 = fig.add_subplot(221)

    #fig.suptitle("2D (" + str(round(time_2D,1)) +" s) - LEFT vs. 3D (" + str(round(time_3D,1)) + " s) - RIGHT (y = " + str(lhc) + " cm)")
    #im1 = ax1.imshow(velx2D[ib:it,:], interpolation='bilinear', cmap=cm.bwr,
    #                 origin='lower', extent=[yzn0_2D[0], yzn0_2D[ny_2D - 1], xzn0_2D[ib], xzn0_2D[it]],
    #                 vmax=vvmaxv, vmin=vvminv)


    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im1, cax=cax, orientation='vertical')
    #ax1.set_title("2D (velx)")

    #ax2 = fig.add_subplot(222)

    #im2 = ax2.imshow(velx3D[ib:it,:], interpolation='bilinear', cmap=cm.bwr,
    #                 origin='lower', extent=[yzn0_3D[0], yzn0_3D[ny_3D - 1], xzn0_3D[ib], xzn0_3D[it]],
    #                 vmax=vvmaxv, vmin=vvminv)

    #divider = make_axes_locatable(ax2)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im2, cax=cax, orientation='vertical');
    #ax2.set_title("3D (velx)")

    #ax3 = fig.add_subplot(111)

    #im3 = ax3.imshow(tke2D[ib:it,:], interpolation='bilinear', cmap=cm.bwr,
    #                 origin='lower', extent=[yzn0_2D[0], yzn0_2D[ny_2D - 1], xzn0_2D[ib], xzn0_2D[it]],
    #                 vmax=tkemaxv, vmin=tkeminv)

    #divider = make_axes_locatable(ax3)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im3, cax=cax, orientation='vertical')
    #ax3.set_title("2D (" + str(round(time_2D,1)) +" s)")

    #ax3.set_xlabel('y (cm)')
    #ax3.set_ylabel('x (cm)')


    # ax4 = fig.add_subplot(212)
    ax4 = fig.add_subplot(111)

    # if you want to reverse colormap add _r at the end
    im4 = ax4.imshow(tke3D[ib:it,:], interpolation='bilinear', cmap=cm.bwr,
                     origin='lower', extent=[yzn0_3D[0], yzn0_3D[ny_3D - 1], xzn0_3D[ib], xzn0_3D[it]],
                     vmax=tkemaxv, vmin=tkeminv)

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax, orientation='vertical')
    ax4.set_title("3D (" + str(round(time_3D,1)) +" s)")

    ax4.set_xlabel('y (cm)')
    ax4.set_ylabel('x (cm)')

    plt.show(block=False)

    # save PLOT
    plt.savefig('RESULTS/' + filename3D + '_ccptwo_3Dv_tke.eps')

    #plt.close('all')


