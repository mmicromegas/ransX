import UTILS.PROMPI_data as prd
import matplotlib.pyplot as plt
import numpy as np

dataloc = 'DATA/BLOCKDAT/'
filename_blck = dataloc+'ob3d.45.lrez.01137.blockdat'

dat = 'temp'

ob_blck = prd.PROMPI_blockdat(filename_blck,dat)


#plt.plot(ob_blck.test())

plt.plot(ob_blck.test()[:,120,120])
#plt.plot(ob_blck.test()[:,200,200])
#plt.plot(ob_blck.test()[:,150,200])

plt.show(block=False)
