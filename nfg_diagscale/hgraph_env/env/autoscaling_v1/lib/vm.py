# import numpy as np
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.autoscaling_v1.lib.simqueue import SimQueue
# from workflow_scheduling.env.poissonSampling import one_sample_poisson
import math
import heapq


class VM:
    def __init__(self, vm_id, vcpu, start_t, end_t, price, container_list: list): 
        ##self, vmID, vmCPU, dcID, dataset.datacenter[dcid][0], self.nextTimeStep, task_selection_rule
        self.vmid = vm_id
        self.vcpu = vcpu        # remaining vcpu of VM
        self.max_vcpu = vcpu
        self.vmQueue = SimQueue()  # store the apps waiting to be processed
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
        return util  ## == self.totalProcessTime / 60*60

    def get_capacity(self, app, task):
        return 60*60 / (app.get_taskProcessTime(task)/self.vcpu)  # how many tasks can processed in one hour

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
        
        # print(con.get_vcpu(), self.vcpu)
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


    # ## self-defined
    # def cal_priority(self, task, app):
        
    #     if self.taskSelectRule is None:     # use the FIFO principal
    #         enqueueTime = app.get_enqueueTime(task)
    #         return enqueueTime
    #     else:   
    #         ## task_selection_rule Terminals: ET, WT, TIQ, NIQ, NOC, NOR, RDL
    #         task_ExecuteTime_real = app.get_taskProcessTime(task)/self.vcpu                # ET
    #         task_WaitingTime = self.get_taskWaitingTime(app, task)                        # WT
    #         vm_TotalProcessTime = self.vmQueueTime()                                      # TIQ
    #         vm_NumInQueue = self.currentQlen                                              # NIQ； 
    #                         # not self.vmQueue.qlen(), because in self.task_enqueue(resort = Ture), it will changes with throwout
    #         task_NumChildren = app.get_NumofSuccessors(task)                              # NOC
    #         workflow_RemainTaskNum = app.get_totNumofTask() - app.get_completeTaskNum()   # NOR
    #         RemainDueTime = app.get_Deadline() - self.currentTimeStep #- task_ExecuteTime_real # RDL

    #         priority = self.taskSelectRule(ET = task_ExecuteTime_real, WT = task_WaitingTime, TIQ = vm_TotalProcessTime, 
    #                 NIQ = vm_NumInQueue, NOC = task_NumChildren, NOR = workflow_RemainTaskNum, RDL= RemainDueTime)
    #         return priority


    # def get_firstTaskEnqueueTimeinVM(self):
    #     if self.processingApp is None:
    #         return math.inf
    #     return self.processingApp.get_enqueueTime(self.processingtask)

    # def get_firstTaskDequeueTime(self):
    #     if self.get_pendingTaskNum() > 0:
    #         return self.currentTimeStep
    #     else:
    #         return math.inf

    # def get_firstDequeueTask(self):
    #     return self.processingApp, self.processingtask

    # # how long a new task needs to wait if it is assigned
    # def get_pendingTaskNum(self):
    #     if self.processingApp is None:
    #         return 0
    #     else:
    #         return self.vmQueue.qlen()+1  # 1 is needed

    # def task_enqueue(self, task, enqueueTime, app, resort=False):
    #     temp = app.get_taskProcessTime(task)/self.vcpu       # execute the task in app
    #     self.totalProcessTime += temp
    #     self.pendingTaskTime += temp        
    #     self.currentQlen = self.get_pendingTaskNum()        # number of pending queue + 1

    #     app.update_executeTime(temp, task)
    #     app.update_enqueueTime(enqueueTime, task, self.vmid)
    #     self.vmQueue.enqueue(app, enqueueTime, task, self.vmid, enqueueTime) # last is priority

    #     if self.processingApp is None:
    #         self.process_task()

    #     return temp

    # def task_dequeue(self, resort=True):
    #     task, app = self.processingtask, self.processingApp

    #     # self.currentTimeStep always == dequeueTime(env.nextTimeStep)

    #     qlen = self.vmQueue.qlen()
    #     if qlen == 0:
    #         self.processingApp = None
    #         self.processingtask = None
    #     else:
    #         if resort:  
    #             tempvmQueue = SimQueue()
    #             for _ in range(qlen):
    #                 oldtask, oldapp = self.vmQueue.dequeue()        # Take out the tasks in self.vmQueue in turn and recalculate
    #                 priority = self.cal_priority(oldtask, oldapp)   # re-calculate priority
    #                 heapq.heappush(tempvmQueue.queue, (priority, oldtask, oldapp))
    #             self.vmQueue.queue = tempvmQueue.queue

    #         self.process_task()
    #         self.currentQlen-=1

    #     return task, app 

    # def process_task(self): #
    #     self.processingtask, self.processingApp = self.vmQueue.dequeue() 
    #         # Pop and return the smallest item from the heap, the popped item is deleted from the heap
    #     enqueueTime = self.processingApp.get_enqueueTime(self.processingtask)
    #     processTime = self.processingApp.get_executeTime(self.processingtask)

    #     taskStratTime = max(enqueueTime , self.currentTimeStep)
    #     leaveTime = taskStratTime +processTime

    #     self.processingApp.update_enqueueTime(taskStratTime, self.processingtask, self.vmid)
    #     self.pendingTaskTime -= processTime
    #     self.processingApp.update_pendingIndexVM(self.processingtask, self.pendingTaskNum)
    #     self.pendingTaskNum+=1
    #     self.currentTimeStep = leaveTime

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

    # ## real_waitingTime in dual-tree = currentTime - enqueueTime
    # def get_taskWaitingTime(self, app, task): 
    #     waitingTime = self.currentTimeStep - app.get_enqueueTime(task)
    #     return waitingTime
