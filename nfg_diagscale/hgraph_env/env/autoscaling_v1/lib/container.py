# import numpy as np
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.autoscaling_v1.lib.simqueue import SimQueue
# from workflow_scheduling.env.poissonSampling import one_sample_poisson
import math
import heapq

import copy

import networkx as nx
import numpy as np

from config.param import configs



class Container:
    def __init__(self, id, con_type, vcpu, t, rule): 
        ##self, conID, conCPU, dcID, dataset.datacenter[dcid][0], self.nextTimeStep, task_selection_rule
        self.conid = id
        self.con_type = con_type
        self.vcpu = vcpu
        self.max_scal_vcpu = 0    # the max vcpu that can be used for scaling up
        self.conQueue = SimQueue()  # store the apps waiting to be processed
        self.currentTimeStep = t  # record the leave time of the first processing app
        self.rentStartTime = t
        self.rentEndTime = t
        self.pm = None      # the PM that this container is deployed on
        self.vm = None      # the VM that this container is deployed on
        self.processingApp = None  # store the app with the highest priority
        self.processingtask = None  # the task associated with the highest priority app
        self.totalProcessTime = 0  # record the total processing time required for all queuing tasks
        self.pendingTaskTime = 0
        self.pendingTaskNum = 0
        self.taskSelectRule = rule
        self.currentQlen = 0
        self.aver_resptime = 0
        self.total_resptime = 0
        self.finished_num = 0

        self.active = True

        # used to calculate workload (#request / s) of previous time slot 
        self.request_num = 0
        self.workload_his = np.array([])
    
    def get_utilization(self, app):
        num = len(self.conQueue.queue)
        # print(num)
        if num < 1:
            return 0
        else:
            total_proctime = 0
            for task in self.conQueue.queue:
                # print(task)
                run_time = app.get_taskProcessTime(task[1])
                total_proctime += run_time
            # print(f"pending = {self.conQueue.qlen()}")

            return num / (self.vcpu * (total_proctime / num))


    # def get_utilization(self, app, task):
    #     numOfTask = self.totalProcessTime / (app.get_taskProcessTime(task) / self.vcpu)
    #     util = numOfTask / self.get_capacity(app, task) 
    #     return util  ## == self.totalProcessTime / 60*60

    # def get_capacity(self, app, task):
    #     return 60 * 60 / (app.get_taskProcessTime(task) / self.vcpu)  # how many of the input tasks can be processed within an hour.

    def get_conid(self):
        return self.conid

    def get_contype(self):
        return self.con_type

    def get_pm(self):
        return self.pm

    def get_vm(self):
        return self.vm

    def update_pm(self, pm):
        self.pm = pm

    def update_vm(self, vm):
        self.vm = vm

    def get_vcpu(self):
        return self.vcpu

    def get_max_scal_vcpu(self):
        return self.max_scal_vcpu

    ## self-defined
    def cal_priority(self, task, app):
        
        if self.taskSelectRule is None:     # use the FIFO principal
            enqueueTime = app.get_enqueueTime(task)
            return enqueueTime
        else:   
            ## task_selection_rule Terminals: ET, WT, TIQ, NIQ, NOC, NOR, RDL
            task_ExecuteTime_real = app.get_taskProcessTime(task)/self.vcpu                # ET
            task_WaitingTime = self.get_taskWaitingTime(app, task)                        # WT
            con_TotalProcessTime = self.conQueueTime()                                      # TIQ
            con_NumInQueue = self.currentQlen                                              # NIQ； 
                            # not self.conQueue.qlen(), because in self.task_enqueue(resort = Ture), it will changes with throwout
            task_NumChildren = app.get_NumofSuccessors(task)                              # NOC
            workflow_RemainTaskNum = app.get_totNumofTask() - app.get_completeTaskNum()   # NOR
            RemainDueTime = app.get_Deadline() - self.currentTimeStep #- task_ExecuteTime_real # RDL

            priority = self.taskSelectRule(ET = task_ExecuteTime_real, WT = task_WaitingTime, TIQ = con_TotalProcessTime, 
                    NIQ = con_NumInQueue, NOC = task_NumChildren, NOR = workflow_RemainTaskNum, RDL= RemainDueTime)
            return priority


    def get_firstTaskEnqueueTimeinCON(self):
        if self.processingApp is None:
            return math.inf
        return self.processingApp.get_enqueueTime(self.processingtask)

    def get_firstTaskDequeueTime(self):
        if self.get_pendingTaskNum() > 0:
            return self.currentTimeStep
        else:
            return math.inf

    def get_firstDequeueTask(self):
        return self.processingApp, self.processingtask

    # how long a new task needs to wait if it is assigned
    def get_pendingTaskNum(self):
        if self.processingApp is None:
            return 0
        else:
            return self.conQueue.qlen()+1  # 1 is needed

    def update_history_workload(self):
        # workload = self.request_num / configs.time_slot
        self.workload_his = np.append(self.workload_his, self.request_num)

        self.request_num = 0


    def task_enqueue(self, task, enqueueTime, app, resort=False):
        temp = app.get_taskProcessTime(task)/self.vcpu       # execute the task in app
        
        self.request_num += 1   # add one request

        self.totalProcessTime += temp
        self.pendingTaskTime += temp        
        self.currentQlen = self.get_pendingTaskNum()        # number of pending queue + 1

        app.update_executeTime(temp, task)
        app.update_enqueueTime(enqueueTime, task, self.conid)
        self.conQueue.enqueue(app, enqueueTime, task, self.conid, enqueueTime) # last is priority

        if self.processingApp is None:
            self.process_task()

        return temp

    def task_dequeue(self, resort=True):
        task, app = self.processingtask, self.processingApp

        # self.currentTimeStep always == dequeueTime(env.nextTimeStep)

        qlen = self.conQueue.qlen()
        if qlen == 0:
            self.processingApp = None
            self.processingtask = None
        else:
            if resort:  
                tempconQueue = SimQueue()
                for _ in range(qlen):
                    oldtask, oldapp = self.conQueue.dequeue()        # Take out the tasks in self.conQueue in turn and recalculate
                    priority = self.cal_priority(oldtask, oldapp)   # re-calculate priority
                    heapq.heappush(tempconQueue.queue, (priority, oldtask, oldapp))
                self.conQueue.queue = tempconQueue.queue

            self.process_task()
            self.currentQlen-=1

        return task, app 

    def process_task(self): #
        self.processingtask, self.processingApp = self.conQueue.dequeue() 
            # Pop and return the smallest item from the heap, the popped item is deleted from the heap
        enqueueTime = self.processingApp.get_enqueueTime(self.processingtask)
        processTime = self.processingApp.get_executeTime(self.processingtask)

        taskStratTime = max(enqueueTime , self.currentTimeStep)
        leaveTime = taskStratTime + processTime
        
        self.finished_num += 1
        self.aver_resptime = (leaveTime - enqueueTime) / self.finished_num
        self.total_resptime += (leaveTime - enqueueTime)
        # self.aver_resptime = self.total_resptime / self.finished_num

        self.processingApp.update_enqueueTime(taskStratTime, self.processingtask, self.conid)
        self.pendingTaskTime -= processTime
        self.processingApp.update_pendingIndexCON(self.processingtask, self.pendingTaskNum)
        self.pendingTaskNum+=1
        self.currentTimeStep = leaveTime

    def conQueueTime(self): 
        return max(round(self.pendingTaskTime,3), 0)

    def conTotalTime(self): 
        return self.totalProcessTime
    
    def conLatestTime(self): 
        # return self.totalProcessTime+self.rentStartTime    
        return self.currentTimeStep + self.pendingTaskTime
    
    def get_conRentEndTime(self):
        return self.rentEndTime
    
    def update_conRentEndTime(self, time):
        self.rentEndTime += time

    ## real_waitingTime in dual-tree = currentTime - enqueueTime
    def get_taskWaitingTime(self, app, task): 
        waitingTime = self.currentTimeStep - app.get_enqueueTime(task)
        return waitingTime
    
    """
    Auto_scaling
    """
    def get_container_id(self):
        return self.con_type

    def update_max_scal_vcpu(self, max_scal_vcpu):
        self.max_scal_vcpu = max_scal_vcpu


    def v_scaling(self, num_add, con_queues, map_con_type_id, vm_map_id_vcpu, PrenextTimeStep, G):
        # scale up / down
        if self.vcpu + num_add < 0 or num_add > self.max_scal_vcpu:
            raise ValueError(f"Invalid scaling number")
        
        self.vcpu += num_add
        self.vm.update_vcpu(num_add, vm_map_id_vcpu)
        self.max_scal_vcpu = self.vm.get_vcpu()

        rental = 0
        is_empty = False

        if self.vcpu == 0:
            # this container can be released
            # update the corresponding VM's container_list
            self.active = False
            G.remove_node(self.conid)
            rental, is_empty = self.vm.remove_container(self, num_add, map_con_type_id, PrenextTimeStep)

        return rental, is_empty


    def h_scaling(self, in_out, new_vcpu, con_queues, con_queues_id, map_con_type_id, PrenextTimeStep, firstconWrfLeaveTime, G: nx):
        # scale out / in
        if in_out == 0:
            # scale out
            new_id = max(con_queues_id) + 1
            con_queues_id.append(new_id)
            if new_vcpu == None:
                replica_container = self.__class__(new_id, self.con_type, self.vcpu, PrenextTimeStep, self.taskSelectRule)
            else:
                replica_container = self.__class__(new_id, self.con_type, new_vcpu, PrenextTimeStep, self.taskSelectRule)
            con_queues[new_id] = (replica_container)
            map_con_type_id[self.con_type].append(new_id)

            half_size = len(self.conQueue.queue) // 2
            
            # copy node in networkx
            parents = list(G.predecessors(self.conid))
            children = list(G.successors(self.conid))
            G.add_node(new_id, processTime=G.nodes[self.conid]["processTime"])
            for p in parents:
                G.add_edge(p, new_id)
            for c in children:
                G.add_edge(new_id, c)
            
            for task in self.conQueue.queue[half_size:]:
                replica_container.task_enqueue(task[1], task[0], task[2])
            del self.conQueue.queue[half_size:]
            firstconWrfLeaveTime[new_id] = replica_container.get_firstTaskDequeueTime()

            # print(f"after = {len(self.conQueue.queue)}")
            # print(f"after replica = {len(replica_container.conQueue.queue)}")

            return replica_container
        elif in_out == 1:
            # scale in
            self.active = False
            rental, is_empty = self.vm.remove_container(self, con_queues, map_con_type_id, PrenextTimeStep)
            
            return rental, is_empty