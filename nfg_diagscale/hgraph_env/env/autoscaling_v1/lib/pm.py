# import numpy as np
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.autoscaling_v1.lib.simqueue import SimQueue
# from workflow_scheduling.env.poissonSampling import one_sample_poisson
import math
import heapq
from env.autoscaling_v1.lib.vm import VM


class PM:
    def __init__(self, id, vcpu, t, vm_list: list): 
        ##self, pmID, pmCPU, dcID, dataset.datacenter[dcid][0], self.nextTimeStep, task_selection_rule
        self.pmid = id
        self.used_vcpu = 0
        self.vcpu = vcpu            # remaining vcpu of PM
        self.max_vcpu = vcpu
        self.pmQueue = SimQueue()  # store the apps waiting to be processed
        self.currentTimeStep = t  # record the leave time of the first processing app
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
        return util  ## == self.totalProcessTime / 60*60

    def get_capacity(self, app, task):
        return 60*60 / (app.get_taskProcessTime(task)/self.vcpu)  # how many tasks can processed in one hour

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

    # ## self-defined
    # def cal_priority(self, task, app):
        
    #     if self.taskSelectRule is None:     # use the FIFO principal
    #         enqueueTime = app.get_enqueueTime(task)
    #         return enqueueTime
    #     else:   
    #         ## task_selection_rule Terminals: ET, WT, TIQ, NIQ, NOC, NOR, RDL
    #         task_ExecuteTime_real = app.get_taskProcessTime(task)/self.vcpu                # ET
    #         task_WaitingTime = self.get_taskWaitingTime(app, task)                        # WT
    #         pm_TotalProcessTime = self.pmQueueTime()                                      # TIQ
    #         pm_NumInQueue = self.currentQlen                                              # NIQ； 
    #                         # not self.pmQueue.qlen(), because in self.task_enqueue(resort = Ture), it will changes with throwout
    #         task_NumChildren = app.get_NumofSuccessors(task)                              # NOC
    #         workflow_RemainTaskNum = app.get_totNumofTask() - app.get_completeTaskNum()   # NOR
    #         RemainDueTime = app.get_Deadline() - self.currentTimeStep #- task_ExecuteTime_real # RDL

    #         priority = self.taskSelectRule(ET = task_ExecuteTime_real, WT = task_WaitingTime, TIQ = pm_TotalProcessTime, 
    #                 NIQ = pm_NumInQueue, NOC = task_NumChildren, NOR = workflow_RemainTaskNum, RDL= RemainDueTime)
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
    #         return self.pmQueue.qlen()+1  # 1 is needed

    # def task_enqueue(self, task, enqueueTime, app, resort=False):
    #     temp = app.get_taskProcessTime(task)/self.vcpu       # execute the task in app
    #     self.totalProcessTime += temp
    #     self.pendingTaskTime += temp        
    #     self.currentQlen = self.get_pendingTaskNum()        # number of pending queue + 1

    #     app.update_executeTime(temp, task)
    #     app.update_enqueueTime(enqueueTime, task, self.pmid)
    #     self.pmQueue.enqueue(app, enqueueTime, task, self.pmid, enqueueTime) # last is priority

    #     if self.processingApp is None:
    #         self.process_task()

    #     return temp

    # def task_dequeue(self, resort=True):
    #     task, app = self.processingtask, self.processingApp

    #     # self.currentTimeStep always == dequeueTime(env.nextTimeStep)

    #     qlen = self.pmQueue.qlen()
    #     if qlen == 0:
    #         self.processingApp = None
    #         self.processingtask = None
    #     else:
    #         if resort:  
    #             temppmQueue = SimQueue()
    #             for _ in range(qlen):
    #                 oldtask, oldapp = self.pmQueue.dequeue()        # Take out the tasks in self.pmQueue in turn and recalculate
    #                 priority = self.cal_priority(oldtask, oldapp)   # re-calculate priority
    #                 heapq.heappush(temppmQueue.queue, (priority, oldtask, oldapp))
    #             self.pmQueue.queue = temppmQueue.queue

    #         self.process_task()
    #         self.currentQlen-=1

    #     return task, app 

    # def process_task(self): #
    #     self.processingtask, self.processingApp = self.pmQueue.dequeue() 
    #         # Pop and return the smallest item from the heap, the popped item is deleted from the heap
    #     enqueueTime = self.processingApp.get_enqueueTime(self.processingtask)
    #     processTime = self.processingApp.get_executeTime(self.processingtask)

    #     taskStratTime = max(enqueueTime , self.currentTimeStep)
    #     leaveTime = taskStratTime +processTime

    #     self.processingApp.update_enqueueTime(taskStratTime, self.processingtask, self.pmid)
    #     self.pendingTaskTime -= processTime
    #     self.processingApp.update_pendingIndexVM(self.processingtask, self.pendingTaskNum)
    #     self.pendingTaskNum+=1
    #     self.currentTimeStep = leaveTime

    def pmQueueTime(self): 
        return max(round(self.pendingTaskTime,3), 0)

    def pmTotalTime(self): 
        return self.totalProcessTime
    
    def pmLatestTime(self): 
        # return self.totalProcessTime+self.rentStartTime    
        return self.currentTimeStep + self.pendingTaskTime
    
    def get_pmRentEndTime(self):
        return self.rentEndTime
    
    def update_pmRentEndTime(self, time):
        self.rentEndTime += time

    # ## real_waitingTime in dual-tree = currentTime - enqueueTime
    # def get_taskWaitingTime(self, app, task): 
    #     waitingTime = self.currentTimeStep - app.get_enqueueTime(task)
    #     return waitingTime
