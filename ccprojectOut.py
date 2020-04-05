import UTILS.CCPROJECT.CCproject as ccp
import numpy as np
import os

# dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
# dataloc = '/cosma/home/dp040/dc-moca1/ccp_two_layers_256x256x256/'
# dataloc = '/cosma/home/dp040/dc-moca1/ccp_two_layers/'
# dataloc = '/cosma6/data/dp040/dc-moca1/PROMPI_for_ccptwo_2048x128x128/setups/ccp_two_layers/'
dataloc = 'D:\\ransX\\DATA_D\\BINDATA\\ccp_two_layers\\cosma\\'
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

oneeu = 2.972468e+49 # one energy unit erg
oneflx = oneeu/(onetu*(onelu**2))

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
    # for X1 the source is x2, for X0 the souce is x1
    q.write_output(data, time, ['x','density','pressure','temp','Avalue','x2','vel',
                                'minq_rho','maxq_rho','stdevq_rho',
                                'minq_press', 'maxq_press', 'stdevq_press',
                                'minq_temp', 'maxq_temp', 'stdevq_temp',
                                'minq_Avalue', 'maxq_Avalue', 'stdevq_Avalue',
                                'minq_Vmag', 'maxq_Vmag', 'stdevq_Vmag',
                                'vely','velx','velz', # vely and velx are switched, according to specs, Y is vertical/gravity direction
                                'velxmag','velhormag',
                                'Hflux','KEflux'],
                   filename[0:chp], ['Y','RHO','P','TEMP','A','X1','V',
                                     'MIN_RHO','MAX_RHO','STDEV_RHO',
                                     'MIN_PRESS', 'MAX_PRESS', 'STDEV_PRESS',
                                     'MIN_TEMP', 'MAX_TEMP', 'STDEV_TEMP',
                                     'MIN_A','MAX_A','STDEV_A',
                                     'MIN_V', 'MAX_V', 'STDEV_V',
                                     'VX','VY','VZ','|VY|','VXZ','FH','FK'], icnt,
                   [onelu,onerho,onepr,onete,onea,onex,oneve,
                    onerho,onerho,onerho,
                    onepr, onepr, onepr,
                    onete, onete, onete,
                    onea, onea, onea,
                    oneve, oneve, oneve, oneve, oneve,
                    oneve, oneve, oneve, oneflx, oneflx])
