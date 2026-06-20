import inspect
import os
import sys

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


from env.autoscaling_v1.lib.buildDAGfromXML import buildGraph
from env.autoscaling_v1.lib.get_DAGlongestPath import get_longestPath_nodeWeighted
import copy

app_type_s = "Montage_5"
app_type_t = "Test_1"
app_type_app6 = "App_6"
app_type_app11 = "App_11"
app_type_app12 = "App_12"
app_type_app13 = "App_13"
app_type_app14 = "App_14"

dataset_dict = {'T': app_type_t, "A6": app_type_app6, "A11": app_type_app11, \
                 "A12": app_type_app12, "A13": app_type_app13, "A14": app_type_app14}
class dataset:
    def __init__(self, arg):
        if arg not in dataset_dict:
            raise NotImplementedError
        self.wset = []

        self.wsetTotProcessTime = []
        dag, wsetProcessTime = buildGraph(f'{0}', parentdir + f'/autoscaling_v1/dax/{dataset_dict[arg]}.xml')
        self.wset.append(dag)

        
        
        self.wsetTotProcessTime.append(wsetProcessTime)

        self.wsetSlowestT = []
        for app in self.wset:
            self.wsetSlowestT.append(get_longestPath_nodeWeighted(app))

        self.wsetBeta = []
        for app in self.wset:
            self.wsetBeta.append(2)

        self.vm_vcpu = [4, 8, 16, 32, 48]

        self.vm_mem = [16, 32, 64, 128, 192]

        self.request = np.array([1]) * 0.01

        self.datacenter = [(0, 'East, USA', 0.096)]

        self.vm_price = {4: 0.192, 8: 0.384, 16: 0.768, 32: 1.536, 48: 2.304}
