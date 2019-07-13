import UTILS.PROMPI_data as prd
#import matplotlib.pyplot as plt
import numpy as np
import yt

dataloc = 'D:\simonX\ob-mres-newbindata-5Apr19\ob-mres-newbindata-5Apr19'
filename_blck = dataloc+'ob3d.45.hrez.00617.bindata'

dat = ['density','temp']

block = prd.PROMPI_bindata(filename_blck,dat)


#plt.plot(ob_blck.test())

#plt.plot(ob_blck.test()[:,10,10])
#plt.plot(ob_blck.test()[:,200,200])
#plt.plot(ob_blck.test()[:,150,200])

#ds = yt.load_uniform_grid({}, [128, 128, 128],
#                          bbox=np.array([[0.0, 1.0], [0.0, np.pi], [0.0, 2*np.pi]]),
#                          geometry="spherical")

#s = ds.slice(2, np.pi/2)
#p = s.to_pw("funfield", origin="native")
#p.set_zlim("all", 0.0, 4.0)
#p.show()						  

grid = block.grid()

nx = grid['nx']
ny = grid['ny']
nz = grid['nz']

print(nx,ny,nz)

rho = block.datadict['density']

data = dict(density = (rho, "g/cm**3"))

ds = yt.load_uniform_grid(data, rho.shape,
                          bbox=np.array([[3.e8, 1.e9], [0.0, np.pi], [0.0, 2*np.pi]]),
                          geometry="spherical")



#arr = np.random.random(size=(64,64,64))

#data = dict(density = (arr, "g/cm**3"))
#bbox = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
#ds = yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", bbox=bbox, nprocs=1)

s = ds.slice(2, np.pi/2)
#p = s.to_pw("funfield", origin="native")
#s.set_zlim("all", 0.0, 4.0)
s.save('test')
s.show()



#plt.show(block=False)
