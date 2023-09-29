#!/usr/bin/env python3
import numpy as np
from osc4py3.as_allthreads import *
from osc4py3 import oscmethod as osm

osc_startup()
ip = "172.20.10.10"
port = 8000
osc_udp_server(ip, port, "aservername")

# objectData = classObject()
def callback(address, *args):
    # objectData.update(args)
    print("OK")
osc_method("/Optitrack", callback,
            argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK)

while True:
    osc_process()
