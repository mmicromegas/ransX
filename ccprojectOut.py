import UTILS.CCPROJECT.CCproject as ccp
import numpy as np
import os

# dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
# dataloc = '/cosma/home/dp040/dc-moca1/ccp_two_layers_256x256x256/'
# dataloc = '/cosma/home/dp040/dc-moca1/ccp_two_layers/'
dataloc = '/cosma6/data/dp040/dc-moca1/PROMPI_for_ccptwo_2048x128x128/setups/ccp_two_layers/'
bindata = [filee for filee in sorted(os.listdir(dataloc)) if "bindata" in filee]

# update 7/March/2020 due to new Rgascons

onetu = 0.7920256 # ccp one time unit
onerho = 1.820940e+06 # ccp one density unit
onelu = 4.e8 # cp one length unit
onete = 3.401423e+09 # ccp one temperature unit
onepr = 4.644481e+23 # ccp one pressure unit dyne/cm**2
oneve = onelu/onetu
onea = onepr/(onerho**(5./3.))

one = 1.
onex = one
oneenuc = one

icnt = 0
chp = 100
for fff in bindata:
    filename = os.path.join(dataloc, fff)
    print(filename)
    icnt += 1
    q = ccp.CCproject(filename)
    data = q.getData()
    time = data['time']/onetu
    print(data['time'],time,onetu)
    q.write_output(data, time, ['x','density','pressure','temp','Avalue','x1','vel','velx','vely','velz'],
                   filename[0:chp], ['Y','RHO','P','TEMP','A','X1','V','VX','VY','VZ'], icnt,
                   [onelu,onerho,onepr,onete,onea,onex,oneve,oneve,oneve,oneve])
