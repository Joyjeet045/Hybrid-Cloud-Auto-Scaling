import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.autoscaling_v1.lib.simqueue import SimQueue
import math
import heapq


class VM:
    def __init__(self, vm_id, vcpu, start_t, end_t, price, container_list: list): 
        self.vmid = vm_id
        self.vcpu = vcpu
        self.max_vcpu = vcpu
        self.vmQueue = SimQueue()
        self.rentStartTime = start_t
        self.rentEndTime = end_t
        self.currentQlen = 0

        self.pm = None

        self.aver_resptime = 0
        self.total_resptime = 0
        self.pending_num = 0

        self.price = price
        self.rental = 0

        self.container_list = container_list

        self.active = True

    def get_total_resptime(self):
        for con in self.container_list:
            self.total_resptime += con.aver_resptime
            self.pending_num += con.conQueue.qlen()

        return self.total_resptime, self.pending_num
    
    def get_aver_resptime(self):
        total_resptime = 0
        for con in self.container_list:
            total_resptime += con.total_resptime

        self.aver_resptime = total_resptime / len(self.container_list)

        return self.aver_resptime

    def get_utilization(self, app, task):
        numOfTask = self.totalProcessTime / (app.get_taskProcessTime(task)/self.vcpu)
        util = numOfTask/self.get_capacity(app, task) 
        return util

    def get_capacity(self, app, task):
        return 60*60 / (app.get_taskProcessTime(task)/self.vcpu)

    def get_vmid(self):
        return self.vmid

    def get_vcpu(self):
        return self.vcpu

    def get_maxvcpu(self):
        return self.max_vcpu
    
    def get_container_list(self):
        return self.container_list
    
    def get_pm(self):
        return self.pm

    def update_pm(self, pm):
        self.pm = pm

    def add_container(self, con):
        if con in self.container_list:
            raise ValueError(f"{con} is already deployed in the VM")
        
        if con.get_vcpu() <= self.vcpu:
            self.container_list.append(con)
            self.vcpu -= con.get_vcpu()
            for c in self.container_list:
                c.update_max_scal_vcpu(self.vcpu)
            con.update_vm(self)
            con.update_pm(self.pm)
            if self.pm != None:
                self.pm.used_vcpu += con.get_vcpu()
        else:
            raise ValueError(f"{con} cannot be deployed on this VM")
        
    def update_vcpu(self, num_vcpu, vm_map_id_vcpu):
        """
        update the remaing vcpu during scaling
        """
        self.vcpu -= num_vcpu
        self.pm.used_vcpu += num_vcpu
        vm_map_id_vcpu[self.vmid] = self.vcpu
        for c in self.container_list:
            c.update_max_scal_vcpu(self.vcpu)

    def remove_container(self, con, num_add, map_con_type_id, PrenextTimeStep):
        if con not in self.container_list:
            raise ValueError(f"{con} is not deployed in the VM")

        self.container_list.remove(con)
        self.vcpu -= num_add
        map_con_type_id[con.get_contype()].remove(con.get_conid())
        is_empty = False
        if self.container_list == []:
            self.update_vmRentEndTime(PrenextTimeStep)
            self.rental = self.get_rental()
            is_empty = self.pm.remove_vm(self)

        return self.rental, is_empty
    
    def get_step_rental(self, pre_timestape):
        return (self.rentEndTime / 3600000 - pre_timestape / 3600000) * self.price


        



















    def vmQueueTime(self): 
        return max(round(self.pendingTaskTime,3), 0)

    def vmTotalTime(self): 
        return self.totalProcessTime
    
    def get_vmRentEndTime(self):
        return self.rentEndTime
    
    def update_vmRentEndTime(self, time):
        self.rentEndTime = time

    def get_rental(self):
        self.rental += (self.rentEndTime / 3600000 - self.rentStartTime / 3600000) * self.price

        return self.rental

