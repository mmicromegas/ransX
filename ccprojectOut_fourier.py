import UTILS.CCPROJECT.CCproject_fourier as ccp
import numpy as np
import os

# dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
# dataloc = '/cosma/home/dp040/dc-moca1/ccp_two_layers_256x256x256/'
# dataloc = '/cosma/home/dp040/dc-moca1/ccp_two_layers/'
# dataloc = '/cosma6/data/dp040/dc-moca1/PROMPI_for_ccptwo_128x128x128/setups/ccp_two_layers/'
dataloc = 'D:\\ransX\\DATA_D\\BINDATA\\ccp_two_layers\\cosma\\'
bindata = [filee for filee in sorted(os.listdir(dataloc)) if "bindata" in filee]

# update 7/March/2020 due to new Rgascons

onetu = 0.7920256 # ccp one time unit
onerho = 1.820940e+06 # ccp one density unit
onelu = 4.e8 # cp one length unit
oneeu = 2.972468e+49 # one energy unit erg
onemu = 1.165402e+32 # ccp one mass unit in grams
onetke = oneeu/onemu

one = 1.

icnt = 0
chp = 100
for fff in bindata:
    filename = os.path.join(dataloc, fff)
    #print(filename)
    icnt += 1
    q = ccp.CCproject_fourier(filename)
    data = q.getData()
    time = data['time']/onetu
    # print(data['time'],time,onetu)
    # for X1 the source is x2, for X0 the souce is x1
    q.write_output(data, time, ['tkespect'],
                   filename[0:chp], ['H2'], icnt,
                   [onetke])
