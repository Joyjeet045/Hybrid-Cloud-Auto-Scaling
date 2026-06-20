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

# dataset_30 = ['CyberShake_30', 'Montage_25', 'Inspiral_30', 'Sipht_30']  # test instance 1
# dataset_50 = ['CyberShake_50', 'Montage_50', 'Inspiral_50', 'Sipht_60']  # test instance 2
# dataset_100 = ['CyberShake_100', 'Montage_100', 'Inspiral_100', 'Sipht_100']  # test instance 3
# dataset_1000 = ['CyberShake_1000', 'Montage_1000', 'Inspiral_1000', 'Sipht_1000']  # test instance 4
# dataset_test = ['Montage_3', 'Montage_5', 'Montage_6', 'Montage_25']
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

        # the maximum processing time of a DAG from entrance to exit
        self.wsetSlowestT = []
        for app in self.wset:
            self.wsetSlowestT.append(get_longestPath_nodeWeighted(app))

        self.wsetBeta = []
        for app in self.wset:
            self.wsetBeta.append(2)

        # Five Amazon EC2 m5 VM types exactly as listed in STAR Table 2
        # (Fang et al., 2026): m5.xlarge..m5.12xlarge = 4/8/16/32/48 vCPU.
        self.vm_vcpu = [4, 8, 16, 32, 48]  # EC2 m5 (STAR Table 2)

        self.vm_mem = [16, 32, 64, 128, 192]

        self.request = np.array([1]) * 0.01  # the default is 0.01, lets test 10.0 1.0 and 0.1

        self.datacenter = [(0, 'East, USA', 0.096)]

        # Hourly on-demand price per VM type (STAR Table 2).
        self.vm_price = {4: 0.192, 8: 0.384, 16: 0.768, 32: 1.536, 48: 2.304}
