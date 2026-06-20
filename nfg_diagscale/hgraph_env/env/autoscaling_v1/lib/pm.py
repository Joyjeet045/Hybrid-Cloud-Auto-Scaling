import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.autoscaling_v1.lib.simqueue import SimQueue
import math
import heapq
from env.autoscaling_v1.lib.vm import VM


class PM:
    def __init__(self, id, vcpu, t, vm_list: list): 
        self.pmid = id
        self.used_vcpu = 0
        self.vcpu = vcpu
        self.max_vcpu = vcpu
        self.pmQueue = SimQueue()
        self.currentTimeStep = t
        self.rentStartTime = t
        self.rentEndTime = t
        self.currentQlen = 0

        self.vm_list = vm_list
        self.aver_resptime = 0
        
        self.active = True

    def get_aver_resptime(self):
        total_resptime = 0
        for vm in self.vm_list:
            total_resptime += vm.get_aver_resptime()

        self.aver_resptime = total_resptime / len(self.vm_list)

        return self.aver_resptime

    def get_utilization(self, app, task):
        numOfTask = self.totalProcessTime / (app.get_taskProcessTime(task)/self.vcpu)
        util = numOfTask/self.get_capacity(app, task) 
        return util

    def get_capacity(self, app, task):
        return 60*60 / (app.get_taskProcessTime(task)/self.vcpu)

    def get_pmid(self):
        return self.pmid

    def get_vcpu(self):
        return self.vcpu

    def get_maxvcpu(self):
        return self.max_vcpu
    
    def get_vm_list(self):
        return self.vm_list
    
    def get_util(self):
        return self.used_vcpu / self.max_vcpu

    def add_vm(self, vm):
        if vm in self.vm_list:
            raise ValueError(f"{vm} is already deployed in the VM")
        
        if vm.get_maxvcpu() <= self.vcpu:
            self.vm_list.append(vm)
            for con in vm.container_list:
                self.used_vcpu += con.vcpu
                con.update_pm(self)
            self.vcpu -= vm.get_maxvcpu()
            vm.update_pm(self)
        else:
            raise ValueError(f"{vm} connot be deployed on this PM")
        

    def remove_vm(self, VM):
        if VM not in self.vm_list:
            raise ValueError(f"{VM} is not deployed in this PM")
        
        self.vcpu += VM.get_vcpu()
        self.vm_list.remove(VM)
        is_empty = False
        if self.vm_list == []:
            is_empty = True
            self.active = False

        return is_empty

        



















    def pmQueueTime(self): 
        return max(round(self.pendingTaskTime,3), 0)

    def pmTotalTime(self): 
        return self.totalProcessTime
    
    def pmLatestTime(self): 
        return self.currentTimeStep + self.pendingTaskTime
    
    def get_pmRentEndTime(self):
        return self.rentEndTime
    
    def update_pmRentEndTime(self, time):
        self.rentEndTime += time

