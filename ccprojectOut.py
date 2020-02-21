import UTILS.CCPROJECT.CCproject as ccp
import numpy as np
import os

dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
bindata = [filee for filee in sorted(os.listdir(dataloc)) if "bindata" in filee]

onetu = 0.7951638 # ccp one time unit
onerho = 1.820940e+06 # ccp one density unit
onelu = 4.e8 # cp one length unit
onete = 3.401423e+09 # ccp one temperature unit
onepr = 4.607893e+23 # ccp one pressure unit dyne/cm**2
oneve = onelu/onetu
onea = onepr/(onerho**(5./3.))

one = 1.
onex = one
oneenuc = one

icnt = 0
for fff in bindata:
    filename = os.path.join(dataloc, fff)
    print(filename)
    icnt += 1
    q = ccp.CCproject(filename)
    data = q.getData()
    time = data['time']/onetu
    q.write_output(data, time, ['x','density','pressure','temp','velx','vely','velz','vel','x1','x2','Avalue'],
                   filename, ['Y','RHO','P','T','UX','UY','UZ','|U|','X1','X2','A'], icnt,
                   [onelu,onerho,onepr,onete,oneve,oneve,oneve,oneve,onex,onex,onea])
