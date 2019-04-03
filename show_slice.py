import UTILS.PROMPI_data as prd
import matplotlib.pyplot as plt
import numpy as np

dataloc = 'DATA/BINDATA/'
filename_blck = dataloc+'ob3d.45.nnuc25.lrez.4cpu.00004.bindata'

dat = 'density'

ob_blck = prd.PROMPI_bindata(filename_blck,dat)


#plt.plot(ob_blck.test())

plt.plot(ob_blck.test()[:,10,10])
#plt.plot(ob_blck.test()[:,200,200])
#plt.plot(ob_blck.test()[:,150,200])

plt.show(block=False)
