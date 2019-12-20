import UTILS.PROMPI_data as prd
import numpy as np


def threed(a, b, c, param):
    lst = [[[param for col in range(a)] for col in range(b)] for row in range(c)]
    return lst


# dataloc = '/home/miro/ccp_one_layer/'
# filename_blck = dataloc+'ccpone.lres.00001.bindata'

# dataloc = '/home/miro/ransX/DATA/BINDATA/ccp_two_layers/'
# filename_blck = dataloc+'ccptwo.res128cubed.00315.bindata'

dataloc = '/home/miro/ransX/DATA/BINDATA/ccp_two_layers/'
filename_blck = dataloc + 'ccptwo.ideal.lres2.01871.bindata'

dat = ['density', 'enuc1']

block = prd.PROMPI_bindata(filename_blck, dat)

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

rho = block.datadict['density']
enuc1 = block.datadict['enuc1']

source = rho * enuc1

volone = (xznr[1] - xznr[0]) * (yznr[1] - yznr[0]) * (zznr[1] - zznr[0])
Vol = np.asarray(threed(nx, ny, nz, volone))

totlum = (Vol * source).sum()

print("Total LUminosity in ergs/s is: ", totlum)
