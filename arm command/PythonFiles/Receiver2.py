#!/usr/bin/env python3
from oscpy.server import OSCThreadServer
from oscpy.client import OSCClient
import time

# Communication INIT with Optitrack
osc = OSCThreadServer()
osc.listen(address="172.20.10.10", port=8000, default=True)

# objectData = classObject()

def callbackOSC(*values):
    print(values)
    # data = np.zeros(9)
    # data[0:9] = values
    # objectData.update(data)
            
osc.bind(b"/Optitrack", callbackOSC)

while True :
    time.sleep(0.001)