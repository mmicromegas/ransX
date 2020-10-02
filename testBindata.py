import UTILS.PROMPI.PROMPI_data as prd
import numpy as np
import matplotlib.pyplot as plt

#dataloc = 'D:\\ransX\\DATA_D\\BINDATA\\cflash_nucboost10x\\'
#filename_blck = dataloc + 'c3d.128x128x128.nucb10x.00159.bindata'

dataloc = 'D:\\ransX\\DATA_D\\BINDATA\\ccp_two_layers\\cosma\\RAPHAEL\\ccptwo.r256x16x16.cosma.00002.bindata'

dat = ['density','velx','vely','velz','energy','press','temp','gam1','gam2','enuc1','enuc2','0001','0002']

block = prd.PROMPI_bindata(dataloc, dat)

grid = block.grid()

nx = grid['nx']
ny = grid['ny']
nz = grid['nz']

xznl = block.datadict['xznl']
xznr = block.datadict['xznr']

yznl = block.datadict['yznl']
yznr = block.datadict['yznr']

zznl = block.datadict['zznl']
zznr = block.datadict['zznr']

print(nx, ny, nz)

density = block.datadict['density']
velx = block.datadict['velx']
vely = block.datadict['vely']
velz = block.datadict['velz']
energy = block.datadict['energy']
press = block.datadict['press']
temp = block.datadict['temp']
gam1 = block.datadict['gam1']
gam2 = block.datadict['gam2']

x0001 = block.datadict['0001']
x0002 = block.datadict['0002']
#x0003 = block.datadict['0003']
#x0004 = block.datadict['0004']
#x0005 = block.datadict['0005']
#x0006 = block.datadict['0006']

print('MAXdensity',np.max(density))
print('MAXvelx',np.max(velx))
print('MAXvely',np.max(vely))
print('MAXvelz',np.max(velz))
print('MAXenergy',np.max(energy))
print('MAXpress',np.max(press))
print('MAXtemp',np.max(temp))
print('MAXgam1',np.max(gam1))
print('MAXgam2',np.max(gam2))

print('MAX0001',np.max(x0001))
print('MAX0002',np.max(x0002))
#print('MAX0003',np.max(x0003))
#print('MAX0004',np.max(x0004))
#print('MAX0005',np.max(x0005))
#print('MAX0006',np.max(x0006))

# energy is velx
# gam2 is vely ? check
# x0002 is velz
# x0006 energy ? check

plt.plot(velx[:,1,1])
#print(x0002[:,63,63])
plt.show()

#print(vely)


