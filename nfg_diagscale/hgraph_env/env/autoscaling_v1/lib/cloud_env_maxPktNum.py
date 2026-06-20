import numpy as np
# import pandas as pd
import csv
import math
import os, sys, inspect, random, copy

from config.param import configs


class _BoxShim:
    """Minimal stand-in for ``gym.spaces.Box`` (only ``.shape`` is used)."""

    def __init__(self, low=0, high=0, shape=(0,)):
        self.low = low
        self.high = high
        self.shape = shape


class _DiscreteShim:
    """Minimal stand-in for ``gym.spaces.Discrete`` (only ``.n`` is used)."""

    def __init__(self, n=0):
        self.n = n


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.autoscaling_v1.lib.stats import Stats
from env.autoscaling_v1.lib.poissonSampling import one_sample_poisson
from env.autoscaling_v1.lib.vm import VM
from env.autoscaling_v1.lib.container import Container
from env.autoscaling_v1.lib.pm import PM
from env.autoscaling_v1.lib.workflow import Workflow
from env.autoscaling_v1.lib.simqueue import SimQueue
from env.autoscaling_v1.lib.simsetting import Setting
from env.autoscaling_v1.lib.cal_rank import calPSD

from utils.utils import graph_construct

action_list = []
conidRange = 10000

def ensure_dir_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv_header(file, header):
    ensure_dir_exist(file)
    with open(file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header)

def write_csv_data(file, data):
    ensure_dir_exist(file)
    with open(file, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(data)


class cloud_simulator(object):

    def __init__(self, args):

        self.set = Setting(args)
        self.budget = args["budget"]
        self.deadline = 500
        self.test = False

        self.app_type = args["app_types"]
        self.TaskRule = None                # input th task selection rule here, if has

        self.predicted_workload = [0, 0, 0, 0, 0]
        self.workload_next = 0

        if self.set.is_wf_trace_record:
            self.df = {}
            __location__ = os.getcwd() + r'\Saved_Results'
            self.pkt_trace_file = os.path.join(__location__, r'allocation_trace_%s_seed%s_arr%s_gamma%s.csv' % (args["algo"],  args["seed"], args["arrival rate"], args["gamma"]))
            write_csv_header(self.pkt_trace_file, ['Workflow ID', 'Workflow Pattern', 'Workflow Arrival Time', 'Workflow Finish Time', 'Workflow Deadline', 'Workflow Deadline Penalty',
                                                   'Task Index', 'Task Size', 'Task Execution Time', 'Task Ready Time', 'Task Start Time', 'Task Finish Time',
                                                   'CON ID', 'CON speed', 'Price', 'CON Rent Start Time', 'CON Rent End Time', 'CON Pending Index' ]) # 6 + 6 + 6 columns     


        self.observation_space = _BoxShim(low=0, high=10000, shape=(6 + self.set.history_len,))
        self.action_space = _DiscreteShim(n=100)  # n is a placeholder

        """
        VM and PM info
        """
        self.vm_vcpu_types = self.set.dataset.vm_vcpu
        self.vm_mem_types = self.set.dataset.vm_mem
        self.vm_prices = self.set.dataset.vm_price

    def close(self):
        print("Environment id %s is closed" % (self.set.envid))


    def _init(self, test=False):
        if test == True:
            self.test = True
        else:
            self.test = False

        self.num_app = 0
        self.replica_number = []

        self.app_instances = copy.deepcopy(self.set.dataset.wset[0])
        self.app_instances.add_node("start")
        self.app_instances.add_edge("start", 0)

        self.set.reset_workload(test=test)
        # metrics
        self.finished_req = 0
        self.response_time = np.array([])
        self.mean_step_resptime = np.array([])
        self.step_cost = np.array([])

        self.average_resptime = 0
        self.re_cost = 0
        self.total_cost = 0
        self.pre_total_cost = 0
        # vm list
        self.vm_queues = np.array([])
        self.vm_map_id_vcpu = {}
        self.vm_queues_vcpu = []
        self.vm_queues_mem = []
        # PM list
        self.pm_queues = np.array([])
        self.pm_map_id_vcpu = {}
        self.pm_queues_vcpu = []
        self.pm_queues_mem = []

        self.appSubDeadline = {}  # { app: {task: task_sub_deadline} } used as a state feature
        self.usr_queues = []            # [usr1:[workflows, ...], usr2:[workflows, ...]], e.g., user1 stores 30 workflows
        self.con_queues = {}             # [CON1, CON2, ...] each CON is a class
        self.con_queues_id = []          # the conid of each CON in self.con_queues
        self.con_queues_vcpu = []
        self.con_queues_rentEndTime = []


        self.map_con_type_id = {}
        self.usrNum = self.set.usrNum   ## useless for one cloud
        self.dcNum = self.set.dcNum     ## useless for one cloud
        self.wrfNum = self.set.wrfNum
        self.totWrfNum = self.set.totWrfNum
        self.CONtypeNum = len(self.set.dataset.vm_vcpu) ## number of con types
        self.numTimestep = 0            # indicate how many timesteps have been processed
        self.completedWF = 0
        self.CONRemainingTime = {}       # {conid1:time, conid2:time}
        self.CONRemainAvaiTime = {}      # reamin available time  = leased time period - con_total_execute_time
        self.CONrentInfos = {}           # {CONid: [rent start time, rent end time]}
        self.notNormalized_arr_hist = np.zeros((self.usrNum, self.wrfNum, self.set.history_len)) 
        self.CONcost = 0
        self.SLApenalty = 0
        self.wrfIndex = 0
        self.usrcurrentTime = np.zeros(self.usrNum)  # Used to record the current moment of the user
        self.remainWrfNum = 0           # Record the number of packets remained in CONs
        self.missDeadlineNum = 0
        self.CONrentHours = 0  
        self.CONexecHours = 0  

        # IMPORTANT: used to get the ready task for the next time step
        self.firstconWrfLeaveTime = {}   # Record the current timestamp on each CON
        self.firstusrWrfGenTime = np.zeros(self.usrNum)  # Arrival time of the first inactive workflow in each user's workflow set

        self.uselessAllocation = 0
        self.CONtobeRemove = None

        self.usr_respTime = np.zeros((self.usrNum, self.wrfNum)) 
        self.usr_received_wrfNum = np.zeros((self.usrNum, self.wrfNum)) 


        # upload all workflows with their arrival time to the 'self.firstusrWrfGenTime'
        self.num_req = 0
        for i in range(self.usrNum):                # generate some workflows for each user and put them into user queue
            self.usr_queues.append(SimQueue())      # a queue for each user
            for request in self.set.workload:
                self.num_req += 1
                self.workflow_generator(i, request)
            self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstPktEnqueueTime()    # the first workflow generated time for each user
        # print(self.usr_queues[0].enqueueTime)
        self.nextUsr, self.nextTimeStep = self.get_nextWrfFromUsr() # start from the first generated workflow within all users
        self.PrenextTimeStep = self.nextTimeStep     
        self.previous_time = self.nextTimeStep      
        self.nextisUsr = True
        self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt() # obtain the root task of the first workflow in the self.nextUsr
        temp = self.nextWrf.get_allnextTask(self.finishTask)   # Get all real successor tasks of the virtual workflow root task  
        self.dispatchParallelTaskNum = 0
        self.nextTask = temp[self.dispatchParallelTaskNum]

        # state information
        self.step_response = np.array([])
        self.req_rate = self.set.Workload[0]
        self.step_missdealine = 0
        self.step_finished_req = 0

        if len(temp) > 1:  # the next task has parallel successor tasks
            self.isDequeue = False
            self.isNextTaskParallel = True
        else:
            self.isDequeue = True  # decide whether the nextWrf should be dequeued
            self.isNextTaskParallel = False

        self.stat = Stats(self.set)

        # deploy the application
        self.init_deployment()

    # Generate one workflow at one time
    def workflow_generator(self, usr, next_arrivaltime):
        app_id = 0
        wrf = self.set.dataset.wset[0]
    
        self.remainWrfNum += 1
        # add workflow deadline to the workflow
        pkt = Workflow(self.usrcurrentTime[usr], wrf, app_id, usr, self.set.dataset.wsetSlowestT[app_id], self.set.dueTimeCoef[usr, app_id], self.wrfIndex) #self.set.gamma / max(self.set.dataset.vmVCPU))
        self.usr_queues[usr].enqueue(pkt, self.usrcurrentTime[usr], None, usr, app_id) # None means that workflow has not started yet
        self.usrcurrentTime[usr] = next_arrivaltime
        self.totWrfNum -= 1
        self.wrfIndex +=1


    def reset(self, seed, test=False):
        random.seed(seed)
        np.random.seed(seed)
        self._init(test=test)

    def input_task_rule(self, rule):
        self.TaskRule = rule



    def get_nextWrfFromUsr(self):       # Select the User with the smallest timestamp
        usrInd = np.argmin(self.firstusrWrfGenTime)
        firstPktTime = self.firstusrWrfGenTime[usrInd]
        return usrInd, firstPktTime     # Returns the user and arrival time of the minimum arrival time of the workflow in the current User queue.

    def get_nextWrfFromCON(self):        # Select the machine with the smallest timestamp
        if len(self.firstconWrfLeaveTime) > 0:
            conInd = min(self.firstconWrfLeaveTime, key=self.firstconWrfLeaveTime.get)
            firstPktTime = self.firstconWrfLeaveTime[conInd]
            return conInd, firstPktTime  # Returns con-id and the minimum end time of the current CON
        else:
            return None, math.inf

    def get_nextTimeStep(self):
        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep
        tempnextloc, tempnextTimeStep = self.get_nextWrfFromUsr()  
        tempnextloc1, tempnextTimeStep1 = self.get_nextWrfFromCON()
        if tempnextTimeStep > tempnextTimeStep1:  # task ready time > CON minimum time
            self.nextUsr, self.nextTimeStep = tempnextloc1, tempnextTimeStep1  
                                        # The next step is to process the CON and update it to the timestep of the CON.
            self.nextisUsr = False
            self.nextWrf, self.finishTask = self.con_queues[self.nextUsr].get_firstDequeueTask() # Only returns time, does not process task
        else:  # tempnextTimeStep <= tempnextTimeStep1
            if tempnextTimeStep == math.inf:   ## tempnextTimeStep：when self.usr_queues.queue is []
                self.nextTimeStep = None       ## tempnextTimeStep1：when self.firstconWrfLeaveTime is []
                self.nextUsr = None
                self.nextWrf = None
                self.nextisUsr = True
            else:
                self.nextUsr, self.nextTimeStep = tempnextloc, tempnextTimeStep # Next step is to process user & Update to user's timeStep
                self.nextisUsr = True    # Activate new Workflow from Usr_queue
                self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt() # The current first task in the selected user

        # print(tempnextTimeStep, tempnextTimeStep1, self.nextWrf)


    def record_a_completed_workflow(self, ddl_penalty):

        if self.set.is_wf_trace_record:        
            Workflow_Infos = [self.nextWrf.appArivalIndex, self.nextWrf.appID,
                            self.nextWrf.generateTime, self.nextTimeStep, self.nextWrf.deadlineTime, ddl_penalty]

            for task in range(len(self.nextWrf.executeTime)):

                Task_Infos = [task, self.nextWrf.app.nodes[task]['process_time'], self.nextWrf.executeTime[task], 
                            self.nextWrf.readyTime[task], self.nextWrf.enqueueTime[task], self.nextWrf.dequeueTime[task]]

                CON_Infos = self.CONrentInfos[self.nextWrf.processDC[task]] + [self.nextWrf.pendingIndexOnDC[task]]

                write_csv_data(self.pkt_trace_file, Workflow_Infos + Task_Infos + CON_Infos)


    # def step(self, action_step, pre_timestape, action=None):
    def step(self, pre_timestape, action=None):
        """
        get the response time of all request that generated when initialization
        """
        timestamp = False
        done = False
        reward = 0
        self.step_response = np.array([])
        # self.req_rate = self.set.NASAWorkload[action_step]
        self.step_missdealine = 0
        self.step_finished_req = 0

        """
        auto-sacling
        """
        self.hges_auto_scaling(pre_timestape, action)
        
        while timestamp == False and done == False:
            self.PrenextUsr, self.PrenextTask = self.nextUsr, self.nextTask 
            cand_con = self.map_con_type_id[self.PrenextTask]
            if len(cand_con) > 1:
                ratio = self.cwrr(cand_con)
                # min_index = ratio.index(min(ratio))
                # selected_conid = cand_con[min_index]
                selected_conid = random.choices(cand_con, weights=ratio, k=1)[0]
            else:
                selected_conid = cand_con[0]
            
            self.nextWrf.update_mapping(self.PrenextTask, self.con_queues[selected_conid].pm.get_pmid())
            # if the task is not in the same pm with its parents, will increase the communication time
            comm_time = self.nextWrf.get_comm_time(self.PrenextTask, self.con_queues)

            reward = 0

            self.nextTimeStep += comm_time
            self.PrenextTimeStep = self.nextTimeStep
            # dispatch nextWrf to selectedCON and update the wrfLeaveTime on selectedCON 
            parentTasks = self.nextWrf.get_allpreviousTask(self.PrenextTask)
            if len(parentTasks) == len(self.nextWrf.completeTaskSet(parentTasks)): # all its predecessor tasks have been done, just double-check
                process_time =  self.con_queues[selected_conid].task_enqueue(self.PrenextTask, self.PrenextTimeStep, self.nextWrf)
                self.CONexecHours += (process_time / 3600)    
                self.firstconWrfLeaveTime[selected_conid] = self.con_queues[selected_conid].get_firstTaskDequeueTime() # return currunt timestap on this machine

            # 2) Dequeue nextTask
            if self.isDequeue:      # True: the nextTask should be popped out 
                if self.nextisUsr:  # True: the nextTask to be deployed comes from the user queue
                    self.nextWrf.update_dequeueTime(self.PrenextTimeStep, self.finishTask)
                    _, _ = self.usr_queues[self.PrenextUsr].dequeue() # Here is the actual pop-up of the root task 
                    self.firstusrWrfGenTime[self.PrenextUsr] = self.usr_queues[self.PrenextUsr].getFirstPktEnqueueTime() 
                                                                # Updated with the arrival time of the next workflow

                    self.stat.add_app_arrival_rate(self.PrenextUsr, self.nextWrf.get_appID(), self.nextWrf.get_generateTime()) # record
                else:               # the nextTask to be deployed comes from the con queues
                    _, _ = self.con_queues[self.PrenextUsr].task_dequeue() # Here nextTask actually starts to run
                    self.firstconWrfLeaveTime[self.PrenextUsr] = self.con_queues[self.PrenextUsr].get_firstTaskDequeueTime()
                                                                # Update the current TimeStamp in this machine


            # 3) Update: self.nextTask, and maybe # self.nextWrf, self.finishTask, self.nextUsr, self.nextTimeStep, self.nextisUsr
            temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)   # all successor tasks of the current self.finishTask
                                        # and one successor task has already enqueued

            if len(temp_Children_finishTask) > 0:
                self.dispatchParallelTaskNum += 1
            
            
            while True: 
                
                # self.nextWrf is completed
                while len(temp_Children_finishTask) == 0:  # self.finishTask is the final task of self.nextWrf
                    if self.nextisUsr:  # for double-check: Default is False
                        # Because it corresponds to self.finishTask, if temp==0, it means it cannot be entry tasks
                        print('self.nextisUsr maybe wrong')
                    _, app = self.con_queues[self.nextUsr].task_dequeue()  
                    self.firstconWrfLeaveTime[self.nextUsr] = self.con_queues[self.nextUsr].get_firstTaskDequeueTime() 
                            # If there is no task on the CON, math.inf will be returned
                    if self.nextWrf.is_completeTaskSet(self.nextWrf.get_allTask()):     # self.nextWrf has been completed
                        """
                        finish a request
                        """
                        self.finished_req += 1
                        self.step_finished_req += 1
                        response_time = self.nextTimeStep - self.nextWrf.get_generateTime()

                        self.response_time = np.append(self.response_time, response_time)
                        self.step_response = np.append(self.step_response, response_time)

                        self.usr_respTime[app.get_originDC()][app.get_appID()] += response_time
                        self.usr_received_wrfNum[app.get_originDC()][app.get_appID()] += 1                    
                        self.completedWF += 1
                        self.remainWrfNum -= 1
                        ddl_penalty = self.calculate_penalty(app, response_time)
                        self.SLApenalty += ddl_penalty
                        self.record_a_completed_workflow(ddl_penalty)
                        del app, self.nextWrf

                    self.get_nextTimeStep()
                    if self.nextTimeStep is None:
                        break     
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                    temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

                if self.nextTimeStep is None:
                    break

                # Indicates that parallel tasks have not been allocated yet, and len(temp_Children_finishTask)>=1
                if len(temp_Children_finishTask) > self.dispatchParallelTaskNum: 
                    to_be_next = None
                    while len(temp_Children_finishTask) > self.dispatchParallelTaskNum:
                        temp_nextTask = temp_Children_finishTask[self.dispatchParallelTaskNum]
                        temp_parent_nextTask = self.nextWrf.get_allpreviousTask(temp_nextTask)
                        if len(temp_parent_nextTask) - len(self.nextWrf.completeTaskSet(temp_parent_nextTask)) > 0:
                            self.dispatchParallelTaskNum += 1
                        else: 
                            to_be_next = temp_nextTask
                            break

                    if to_be_next is not None: 
                        self.nextTask = to_be_next
                        if len(temp_Children_finishTask) - self.dispatchParallelTaskNum > 1:
                            self.isDequeue = False
                        else:
                            self.isDequeue = True
                        break

                    else: # Mainly to loop this part
                        _, _ = self.con_queues[self.nextUsr].task_dequeue() # Actually start running self.nextTask here
                        self.firstconWrfLeaveTime[self.nextUsr] = self.con_queues[self.nextUsr].get_firstTaskDequeueTime()
                        self.get_nextTimeStep() 
                     
                        self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask) 
                        self.dispatchParallelTaskNum = 0                     
                        if self.nextTimeStep is not None:
                            temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)                                

                else: # i.e., len(temp_Children_finishTask)<=self.dispatchParallelTaskNum
                    # self.nextTask is the last imcompleted successor task of the self.finishTask
                    if not self.isDequeue:      # Defaults to True
                        print('self.isDequeue maybe wrong')  
                    self.get_nextTimeStep()
                    # print(self.nextWrf)
                
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                    self.dispatchParallelTaskNum = 0 # Restart recording the number of successor tasks of self.finishTask
                    if self.nextTimeStep is not None:
                        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

            self.numTimestep = self.numTimestep + 1  ## useless for GP
            self.notNormalized_arr_hist = self.stat.update_arrival_rate_history() ## useless for GP
            
            if self.remainWrfNum == 0:
                if len(self.firstconWrfLeaveTime) == 0:
                    done = True
                elif next(iter(self.firstconWrfLeaveTime.values())) == math.inf and list(self.firstconWrfLeaveTime.values()).count(next(iter(self.firstconWrfLeaveTime.values()))) == len(self.firstconWrfLeaveTime):
                    done = True
   
            if (self.PrenextTimeStep - pre_timestape) >= 180000:  
                # print("=======================================")
                self.replica_number.append(len(self.con_queues) - self.num_app)
                cout_vm = 0
                for vm in self.vm_queues:
                    if vm.active == True:
                        cout_vm += 1
                        vm.update_vmRentEndTime(self.PrenextTimeStep)
                        self.total_cost += vm.get_step_rental(pre_timestape)

                for conid in self.con_queues.keys():
                    con = self.con_queues[conid]
                    con.update_history_workload()
                    
                reward = 0

                timestamp = True
                self.pre_total_cost = self.total_cost
                self.mean_step_resptime = np.append(self.mean_step_resptime, np.mean(self.step_response))
                self.step_cost = np.append(self.step_cost, self.total_cost)

            if done:
                # print(f"finished requests = {self.finished_req}")
                cout_vm = 0
                cout_pm = 0
                self.replica_number.append(len(self.con_queues) - self.num_app)

                replica_num = {}
                for key, con_list in self.map_con_type_id.items():
                    replica_num[key] = len(con_list) - 1
                # calculate energy consumption
                for pm in self.pm_queues:
                    if pm.active == True:
                        cout_pm += 1
                                                
                for vm in self.vm_queues:
                    if vm.active == True:
                        cout_vm += 1
                        vm.update_vmRentEndTime(self.PrenextTimeStep)
                        self.total_cost += vm.get_step_rental(pre_timestape)

                if self.test == True:
                    print(f"SLA violation (%): {self.missDeadlineNum / self.num_req}")
                    print(f"95th percentile response time (ms): {np.percentile(self.response_time, 95)}")
                    print(f"mean response time (ms): {np.mean(self.response_time)}")
                    print(f"total cost (USD): {self.total_cost}")
                step_cost = self.total_cost - self.pre_total_cost
                
                reward = - max(0, configs.penalty * (self.total_cost - self.budget)) - np.mean(self.response_time)

                self.mean_step_resptime = np.append(self.mean_step_resptime, np.mean(self.step_response))
                self.step_cost = np.append(self.step_cost, self.total_cost)
                
                # reward = - self.total_cost - np.mean(self.response_time)
                # reward = -step_cost-step_average_resp
                # reward = - self.average_resptime - max(0, (step_cost - 150))
                # print(f"number VMs = {cout_vm}")

                # print(action_list)
                self.episode_info = {"CON_execHour": self.CONexecHours, "average_resptime": np.mean(self.response_time), "99_percentile": np.percentile(self.response_time, 99), # CON_totHour is the total rent hours of all CONs
                        "VM_cost": self.total_cost, "SLA_penalty": self.SLApenalty, "missDeadlineNum": self.missDeadlineNum,
                        "response_list": self.response_time, "mean_step_resptime": self.mean_step_resptime,
                        "step_cost": self.step_cost, "replica_number": replica_num}
                # print('Useless Allocation has ----> ',self.uselessAllocation)
                # self._init()  ## cannot delete  
        

        return reward, done, self.response_time, self.total_cost

    # calculate the total CON cost during an episode
    def update_CONcost(self, dc, cpu, add=True):
        if add:
            temp = 1
        else:
            temp = 0
        self.CONcost += temp * self.set.dataset.conPrice[cpu]      # (self.set.dataset.datacenter[dc][-1])/2 * cpu
        self.CONrentHours += temp


    def calculate_penalty(self, app, respTime):
        threshold = self.deadline
        # threshold = app.get_Deadline() - app.get_generateTime()
        if respTime < threshold or round(respTime - threshold,5) == 0:
            return 0
        else:
            self.missDeadlineNum += 1
            self.step_missdealine += 1
            # return np.power(np.e, -(respTime - self.deadline))
            return (respTime / self.deadline)
            # return 0.0001 * (respTime-threshold)
            # return (threshold / respTime)
            # return np.power(np.e, -np.power((respTime - self.deadline) / self.deadline, 2))

    def cwrr(self, cand_con):
        """
        dispatching the request to containers
        """
        total_vcpu = 0
        for con_id in cand_con:
            total_vcpu += self.con_queues[con_id].get_vcpu()
        ratio = []
        for con_id in cand_con:
            ratio.append(self.con_queues[con_id].get_vcpu() / total_vcpu)
        # ratio = []
        # for con_id in cand_con:
        #     ratio.append(self.con_queues[con_id].conQueue.qlen())

        return ratio

    def hges_auto_scaling(self, pre_timestape, action):
        """
        action: 0: remain; 1: +1 vcpu; 2: -1 vcpu; 3: +1 replica; 4: -1 replica
        """ 
        selected_con = action[0]
        scaling = action[1]
        inverse_new_id_map = action[2]
        selected_con = inverse_new_id_map[selected_con]

        con = self.con_queues[selected_con]
        # scaling_num = scaling - 4
        scaling_num = scaling
        action_list.append(scaling_num)
        if scaling_num > 0:
            # scale up/out
            max_vcpu_add = con.max_scal_vcpu
            # print(selected_con, scaling_num, scaling, "scale up")
            if max_vcpu_add >= scaling_num:
                con.v_scaling(scaling_num, self.con_queues, self.map_con_type_id, self.vm_map_id_vcpu, self.PrenextTimeStep, self.app_instances)
            else:
                con.v_scaling(max_vcpu_add, self.con_queues, self.map_con_type_id, self.vm_map_id_vcpu, self.PrenextTimeStep, self.app_instances)
                replica_con = con.h_scaling(0, scaling_num - max_vcpu_add, self.con_queues, self.con_queues_id, self.map_con_type_id, self.PrenextTimeStep, self.firstconWrfLeaveTime, self.app_instances)
                self.deploy_con_vm(replica_con)

        elif scaling_num < 0:
            # scale down/in
            # reduce one vcpu
            util = con.get_utilization(self.nextWrf)
            # print(scaling, con.vcpu)
            if con.active == True and self.firstconWrfLeaveTime[selected_con] == math.inf and con.vcpu > -scaling_num:
                num_add = scaling_num
                rental, is_empty = con.v_scaling(num_add, self.con_queues, self.map_con_type_id, self.vm_map_id_vcpu, self.PrenextTimeStep, self.app_instances)
            elif util == 0 and len(self.map_con_type_id[con.get_contype()]) > 1 and \
                con.active == True and self.firstconWrfLeaveTime[selected_con] == math.inf:
                num_add = -con.get_vcpu()
                rental, is_empty = con.v_scaling(num_add, self.con_queues, self.map_con_type_id, self.vm_map_id_vcpu, self.PrenextTimeStep, self.app_instances)
                self.con_queues_id.remove(selected_con)
                del self.con_queues[selected_con]
                del self.firstconWrfLeaveTime[selected_con]
                if con.vm.container_list == []:
                    self.vm_map_id_vcpu[con.vm.get_vmid()] = 0
                    self.vm_queues[con.vm.get_vmid()].active = False
                    self.vm_queues[con.vm.get_vmid()].update_vmRentEndTime(self.PrenextTimeStep)
                    self.total_cost += self.vm_queues[con.vm.get_vmid()].get_step_rental(pre_timestape)
                if is_empty:
                    self.pm_queues[con.pm.get_pmid()].active = False
                    self.pm_map_id_vcpu[con.pm.get_pmid()] = 0
        
    def deploy_con_vm(self, container: Container):
        """
        Deploying new container to VM
        """
        vm_remaining = list(self.vm_map_id_vcpu.values())
        vm_remaining = np.array(vm_remaining + self.vm_vcpu_types)
        available_index = np.where(vm_remaining >= container.get_vcpu())

        # best-fit
        priority = vm_remaining[available_index] / container.get_vcpu()
        selected_vm_index = available_index[0][np.argmin(priority)]
        if selected_vm_index > len(self.vm_queues) - 1:
            selected_vm_id = selected_vm_index - len(self.vm_queues)
            new_vm_id = max(self.vm_map_id_vcpu.keys()) + 1
            new_vm_vcpu = self.vm_vcpu_types[selected_vm_id]
            new_vm_price = self.vm_prices[new_vm_vcpu]
            new_vm = VM(new_vm_id, new_vm_vcpu, self.PrenextTimeStep, self.PrenextTimeStep, new_vm_price, [])
            # deploy VM to PM
            new_vm.add_container(container)
            self.vm_queues = np.append(self.vm_queues, new_vm)
            
            self.vm_map_id_vcpu[new_vm_id] = new_vm.get_vcpu()
            self.vm_queues_vcpu.append(new_vm.get_vcpu())
            self.deploy_vm_pm(new_vm)
        else:
            selected_vm_id = selected_vm_index
            self.vm_queues[selected_vm_id].add_container(container)
            self.vm_queues_vcpu[selected_vm_id] = self.vm_queues[selected_vm_id].get_vcpu()
            self.vm_map_id_vcpu[selected_vm_id] = self.vm_queues[selected_vm_id].get_vcpu()

        

    def deploy_vm_pm(self, VM: VM):
        """
        deploying VM to PM
        """
        pm_remaining = list(self.pm_map_id_vcpu.values())
        pm_remaining = np.array(self.pm_queues_vcpu + [64])
        available_index = np.where(pm_remaining >= VM.get_maxvcpu())

        # best-fit
        priority = pm_remaining[available_index] / VM.get_maxvcpu()

        selected_pm_index = available_index[0][np.argmin(priority)]
        if selected_pm_index > len(self.pm_queues) - 1:
            selected_pm_id = selected_pm_index - len(self.pm_queues)
            new_pm_id = max(self.pm_map_id_vcpu.keys()) + 1
            new_pm_vcpu = 64
            new_pm = PM(new_pm_id, new_pm_vcpu, self.PrenextTimeStep, [])
            self.pm_queues = np.append(self.pm_queues, new_pm)
            new_pm.add_vm(VM)
            self.pm_queues_vcpu.append(new_pm.get_vcpu())
            self.pm_map_id_vcpu[new_pm_id] = new_pm.get_vcpu()
        else:
            selected_pm_id = selected_pm_index
            # print(len(self.pm_queues), selected_pm_id)
            self.pm_queues[selected_pm_id].add_vm(VM)
            self.pm_queues_vcpu[selected_pm_id] = self.pm_queues[selected_pm_id].get_vcpu()
            self.pm_map_id_vcpu[selected_pm_id] = self.pm_queues[selected_pm_id].get_vcpu()


    def init_deployment(self):
        """
        initial the deployment of containers to VMs
        """
        for i in range(2):
            new_pm = PM(i, 64, self.nextTimeStep, [])
            self.pm_queues = np.append(self.pm_queues, new_pm)
            self.pm_map_id_vcpu[i] = 64
            self.pm_queues_vcpu.append(64)

        if self.app_type == "T":
            for i in range(1):
                vm_vcpu = self.vm_vcpu_types[5]
                vm_price = self.vm_prices[16]
                new_vm = VM(i, vm_vcpu, self.PrenextTimeStep, self.PrenextTimeStep, vm_price, [])
                self.vm_queues = np.append(self.vm_queues, new_vm)
                self.vm_map_id_vcpu[i] = vm_vcpu
                self.vm_queues_vcpu.append(vm_vcpu)

            type = 0
            con_vcpu = 1
            for i in range(1):
                type = i
                new_container = Container(i, type, con_vcpu, self.nextTimeStep, self.TaskRule)
                if type in self.map_con_type_id:
                    self.map_con_type_id[type].append(i)
                else:
                    self.map_con_type_id[type] = [i]

                self.con_queues[i] = new_container
                self.firstconWrfLeaveTime[i] = new_container.get_firstTaskDequeueTime() #new CON is math.inf
                self.con_queues_id.append(i)
                self.con_queues_vcpu.append(con_vcpu)

            for i in range(1):
                self.vm_queues[0].add_container(self.con_queues[i])
                self.vm_queues_vcpu[0] = self.vm_queues[0].get_vcpu()
                self.vm_map_id_vcpu[0] = self.vm_queues[0].get_vcpu()

            for i in range(1):
                self.pm_queues[0].add_vm(self.vm_queues[i])
                self.pm_queues_vcpu[0] = self.pm_queues[0].get_vcpu()
                self.pm_map_id_vcpu[0] = self.pm_queues[0].get_vcpu()
                
        if self.app_type == "A12" or self.app_type == "A6" or self.app_type == "A11" or \
            self.app_type == "A13" or self.app_type == "A14":
            if self.app_type == "A12":
                app_type = 12
                vm_num = 3
            elif self.app_type == "A6":
                app_type = 6
                vm_num = 2
            elif self.app_type == "A11":
                app_type = 11
                vm_num = 3
            elif self.app_type == "A13":
                app_type = 13
                vm_num = 3
            elif self.app_type == "A14":
                app_type = 14
                vm_num = 3

            self.num_app = app_type
            for i in range(vm_num):
                # if i == 0:
                #     vm_vcpu = self.vm_vcpu_types[0]
                #     vm_price = self.vm_prices[48]
                # else:
                # Initial deployment uses m5.4xlarge (16 vCPU) VMs, fixed explicitly
                # so the 5-VM-type pool (which now includes smaller 4/8 vCPU types
                # for best-fit scale-out) does not shrink the initial host capacity.
                vm_vcpu = 16
                vm_price = self.vm_prices[16]
                new_vm = VM(i, vm_vcpu, self.PrenextTimeStep, self.PrenextTimeStep, vm_price, [])
                
                self.vm_queues = np.append(self.vm_queues, new_vm)
                self.vm_map_id_vcpu[i] = vm_vcpu
                self.vm_queues_vcpu.append(vm_vcpu)

            type = 0
            con_vcpu = 1
            for i in range(app_type):
                # if i == 11:
                #     con_vcpu = 1
                type = i
                new_container = Container(i, type, con_vcpu, self.nextTimeStep, self.TaskRule)
                if type in self.map_con_type_id:
                    self.map_con_type_id[type].append(i)
                else:
                    self.map_con_type_id[type] = [i]

                self.con_queues[i] = new_container
                self.firstconWrfLeaveTime[i] = new_container.get_firstTaskDequeueTime() #new CON is math.inf
                self.con_queues_id.append(i)
                self.con_queues_vcpu.append(con_vcpu) 

            for i in range(app_type):
                if i < 4:
                    self.vm_queues[0].add_container(self.con_queues[i])
                    self.vm_queues_vcpu[0] = self.vm_queues[0].get_vcpu()
                    self.vm_map_id_vcpu[0] = self.vm_queues[0].get_vcpu()
                elif i >= 4 and i < 8:
                    self.vm_queues[1].add_container(self.con_queues[i])
                    self.vm_queues_vcpu[1] = self.vm_queues[1].get_vcpu()
                    self.vm_map_id_vcpu[1] = self.vm_queues[1].get_vcpu()
                else:
                    self.vm_queues[2].add_container(self.con_queues[i])
                    self.vm_queues_vcpu[2] = self.vm_queues[2].get_vcpu()
                    self.vm_map_id_vcpu[2] = self.vm_queues[2].get_vcpu()

            for i in range(vm_num):
                if i == 0 or i == 1:
                    self.pm_queues[0].add_vm(self.vm_queues[i])
                    self.pm_queues_vcpu[0] = self.pm_queues[0].get_vcpu()
                    self.pm_map_id_vcpu[0] = self.pm_queues[0].get_vcpu()
                else:
                    self.pm_queues[1].add_vm(self.vm_queues[i])
                    self.pm_queues_vcpu[1] = self.pm_queues[1].get_vcpu()
                    self.pm_map_id_vcpu[1] = self.pm_queues[1].get_vcpu()

    def state_info_construct(self):
        '''
        Tao's DeepScale
        states:
        1.	The Current vCPU provision
        2.	The average CPU utilization
        3.	Predicted workload (to be done)
        '''
        req_rate = self.req_rate
        req_change = self.workload_next - self.req_rate
        ins_num = len(self.con_queues)
        vio_rate = self.step_missdealine / self.step_finished_req if self.step_finished_req != 0 else 0
        aver_response = np.mean(self.step_response) if len(self.step_response) != 0 else 0

        vcpu_provision = 0
        cpu_utilization = np.array([])
        for con_id in self.con_queues.keys():
            vcpu_provision += self.con_queues[con_id].get_vcpu()
        for pm in self.pm_queues:
            if (pm.max_vcpu - pm.vcpu) > 0:
                cpu_utilization = np.append(cpu_utilization, pm.get_util())

        average_util = np.mean(cpu_utilization)
        ob = np.array([vcpu_provision, average_util] + self.predicted_workload)
        # ob = np.array([req_rate, req_change, ins_num, vio_rate, aver_response, average_util])
        # ob = self.normalize_z_score(ob)

        return ob
    
    def layer_graph_construct(self):
        return graph_construct(self.pm_queues, self.vm_queues, self.con_queues, self.app_instances), self.set.Workload

    def normalize_z_score(self, arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        return (arr - mean) / std_dev
        

    # def state_info_construct(self):
    #     '''
    #     states:
    #     1.	Number of child tasks: childNum
    #     2.	Completion ratio: completionRatio
    #     3.	Workflow arrival rate: arrivalRate (a vector of historical arrivalRate)
    #     4.	Whether the CON can satisfy the deadline regardless the extra cost: meetDeadline (0:No, 1:Yes)
    #     5.	Total_overhead_cost = potential con rental fee + deadline violation penalty: extraCost
    #     6.  CON_remainTime: after allocation, currentRemainTime - taskExeTime ( + newCONrentPeriod if applicable)
    #     7.	BestFit - among all the CONs, whether the current one introduces the lowest extra cost? (0, 1)
    #     '''
    #     ob = []

    #     # task related state:
    #     childNum = len(self.nextWrf.get_allnextTask(self.nextTask))  # number of child tasks
    #     completionRatio = self.nextWrf.get_completeTaskNum() / self.nextWrf.get_totNumofTask()
    #     arrivalRate = np.sum(np.sum(self.notNormalized_arr_hist, axis=0), axis=0)
    #     task_ob = [childNum, completionRatio]
    #     task_ob.extend(list(copy.deepcopy(arrivalRate)))
    #     # calculate the sub-deadline for a task
    #     if self.nextWrf not in self.appSubDeadline:
    #         self.appSubDeadline[self.nextWrf] = {}
    #         deadline = self.nextWrf.get_maxProcessTime()*self.set.dueTimeCoef[self.nextWrf.get_originDC()][self.nextWrf.get_appID()]
    #         psd = calPSD(self.nextWrf, deadline, self.set.dataset.vm_vcpu)  # get deadline distribution based on upward rank, e.g., {task:sub_deadline}
    #         for key in psd:
    #             self.appSubDeadline[self.nextWrf][key] = psd[key]+self.nextTimeStep  # transform into absolute deadline, the task must be completed before appSubDeadline[self.nextWrf][key]

    #     # # con related state:
    #     # for con_ind in range(len(self.con_queues)):  # for currently rent CON
    #     #     task_est_startTime = self.con_queues[con_ind].conLatestTime() #get_taskWaitingTime(self.nextWrf,self.nextTask)   # relative time
    #     #     task_exe_time = self.nextWrf.get_taskProcessTime(self.nextTask) / self.con_queues_cpu[con_ind]
    #     #     task_est_finishTime = task_exe_time + task_est_startTime
    #     #     temp = round(self.CONRemainingTime[self.con_queues_id[con_ind]] - task_est_finishTime, 5)
    #     #     if temp > 0:  # the con can process the task within its current rental hour
    #     #         extra_CON_hour = 0
    #     #         con_remainTime = temp
    #     #     else: # need extra CON rental hour
    #     #         extra_CON_hour = math.ceil(- temp / self.set.CONpayInterval)
    #     #         con_remainTime = round(self.set.CONpayInterval * extra_CON_hour + self.CONRemainingTime[self.con_queues_id[con_ind]] - task_est_finishTime, 5)
    #     #     extraCost = (self.set.dataset.datacenter[self.con_queues[con_ind].get_relativeCONloc()][-1]) / 2 * self.con_queues_cpu[con_ind] * extra_CON_hour
    #     #     if task_est_finishTime + self.nextTimeStep < self.appSubDeadline[self.nextWrf][self.nextTask]:  # con can satisfy task sub-deadline
    #     #         meetDeadline = 1  # 1: indicate the con can meet the task sub-deadline
    #     #     else:
    #     #         meetDeadline = 0
    #     #         extraCost += 1 + self.set.dataset.wsetBeta[self.nextWrf.get_appID()] * (task_exe_time + self.nextTimeStep - self.appSubDeadline[self.nextWrf][self.nextTask])  # add SLA penalty
    #     #     ob.append([])
    #     #     ob[-1] = task_ob + [meetDeadline, extraCost, con_remainTime]

    #     for dcind in range(self.dcNum):  # for new CON that can be rented
    #         for cpuNum in self.set.dataset.vm_vcpu:
    #             dc = self.set.dataset.datacenter[dcind]
    #             task_exe_time = self.nextWrf.get_taskProcessTime(self.nextTask) / cpuNum
    #             extra_CON_hour = math.ceil(task_exe_time / self.set.CONpayInterval)
    #             extraCost = dc[-1] / 2 * cpuNum * extra_CON_hour
    #             if task_exe_time + self.nextTimeStep < self.appSubDeadline[self.nextWrf][self.nextTask]:  # con can satisfy task sub-deadline
    #                 meetDeadline = 1  # 1: indicate the con can meet the task sub-deadline
    #             else:
    #                 meetDeadline = 0
    #                 extraCost += 1 + self.set.dataset.wsetBeta[self.nextWrf.get_appID()] * (task_exe_time + self.nextTimeStep - self.appSubDeadline[self.nextWrf][self.nextTask])  # add SLA penalty
    #             con_remainTime = round(self.set.CONpayInterval * extra_CON_hour - task_exe_time, 5)
    #             ob.append([])
    #             ob[-1] = task_ob + [meetDeadline, extraCost, con_remainTime]

    #     # if a CON is the best fit, i.e., min(extraCost)
    #     temp = np.array(ob)
    #     row_ind = np.where(temp[:,-2] == np.amin(temp[:,-2]))[0]  # relative_row_ind indicates the relative index of CON in con_satisfyDL_row_ind
    #     bestFit = np.zeros((len(ob), 1))
    #     bestFit[row_ind,:] = 1
    #     ob = np.hstack((temp,bestFit))

    #     return ob
    