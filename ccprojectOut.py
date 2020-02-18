import UTILS.CCPROJECT.CCproject as ccp
import numpy as np
import os

dataloc = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\DATA\\BINDATA\\ccp_two_layers\\'
bindata = [filee for filee in sorted(os.listdir(dataloc)) if "bindata" in filee]

onetu = 0.7951638 # ccp one time unit
onerho = 1.820940e+06 # ccp one density unit
onex1 = 1.

icnt = 0
for fff in bindata:
    filename = os.path.join(dataloc, fff)
    print(filename)
    icnt += 1
    q = ccp.CCproject(filename)
    data = q.getData()
    time = data['time']/onetu
    q.write_output(data, time, ['density','x1'], filename, ['RHO','X1'], icnt, [onerho,onex1])
