import numpy as np
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
        self.TaskRule = None

        self.predicted_workload = [0, 0, 0, 0, 0]
        self.workload_next = 0

        if self.set.is_wf_trace_record:
            self.df = {}
            __location__ = os.getcwd() + r'\Saved_Results'
            self.pkt_trace_file = os.path.join(__location__, r'allocation_trace_%s_seed%s_arr%s_gamma%s.csv' % (args["algo"],  args["seed"], args["arrival rate"], args["gamma"]))
            write_csv_header(self.pkt_trace_file, ['Workflow ID', 'Workflow Pattern', 'Workflow Arrival Time', 'Workflow Finish Time', 'Workflow Deadline', 'Workflow Deadline Penalty',
                                                   'Task Index', 'Task Size', 'Task Execution Time', 'Task Ready Time', 'Task Start Time', 'Task Finish Time',
                                                   'CON ID', 'CON speed', 'Price', 'CON Rent Start Time', 'CON Rent End Time', 'CON Pending Index' ])


        self.observation_space = _BoxShim(low=0, high=10000, shape=(6 + self.set.history_len,))
        self.action_space = _DiscreteShim(n=100)

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
        self.finished_req = 0
        self.response_time = np.array([])
        self.mean_step_resptime = np.array([])
        self.step_cost = np.array([])

        self.average_resptime = 0
        self.re_cost = 0
        self.total_cost = 0
        self.pre_total_cost = 0
        self.vm_queues = np.array([])
        self.vm_map_id_vcpu = {}
        self.vm_queues_vcpu = []
        self.vm_queues_mem = []
        self.pm_queues = np.array([])
        self.pm_map_id_vcpu = {}
        self.pm_queues_vcpu = []
        self.pm_queues_mem = []

        self.appSubDeadline = {}
        self.usr_queues = []
        self.con_queues = {}
        self.con_queues_id = []
        self.con_queues_vcpu = []
        self.con_queues_rentEndTime = []


        self.map_con_type_id = {}
        self.usrNum = self.set.usrNum
        self.dcNum = self.set.dcNum
        self.wrfNum = self.set.wrfNum
        self.totWrfNum = self.set.totWrfNum
        self.CONtypeNum = len(self.set.dataset.vm_vcpu)
        self.numTimestep = 0
        self.completedWF = 0
        self.CONRemainingTime = {}
        self.CONRemainAvaiTime = {}
        self.CONrentInfos = {}
        self.notNormalized_arr_hist = np.zeros((self.usrNum, self.wrfNum, self.set.history_len)) 
        self.CONcost = 0
        self.SLApenalty = 0
        self.wrfIndex = 0
        self.usrcurrentTime = np.zeros(self.usrNum)
        self.remainWrfNum = 0
        self.missDeadlineNum = 0
        self.CONrentHours = 0  
        self.CONexecHours = 0  

        self.firstconWrfLeaveTime = {}
        self.firstusrWrfGenTime = np.zeros(self.usrNum)

        self.uselessAllocation = 0
        self.CONtobeRemove = None

        self.usr_respTime = np.zeros((self.usrNum, self.wrfNum)) 
        self.usr_received_wrfNum = np.zeros((self.usrNum, self.wrfNum)) 


        self.num_req = 0
        for i in range(self.usrNum):
            self.usr_queues.append(SimQueue())
            for request in self.set.workload:
                self.num_req += 1
                self.workflow_generator(i, request)
            self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstPktEnqueueTime()
        self.nextUsr, self.nextTimeStep = self.get_nextWrfFromUsr()
        self.PrenextTimeStep = self.nextTimeStep     
        self.previous_time = self.nextTimeStep      
        self.nextisUsr = True
        self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt()
        temp = self.nextWrf.get_allnextTask(self.finishTask)
        self.dispatchParallelTaskNum = 0
        self.nextTask = temp[self.dispatchParallelTaskNum]

        self.step_response = np.array([])
        self.req_rate = self.set.Workload[0]
        self.step_missdealine = 0
        self.step_finished_req = 0

        if len(temp) > 1:
            self.isDequeue = False
            self.isNextTaskParallel = True
        else:
            self.isDequeue = True
            self.isNextTaskParallel = False

        self.stat = Stats(self.set)

        self.init_deployment()

    def workflow_generator(self, usr, next_arrivaltime):
        app_id = 0
        wrf = self.set.dataset.wset[0]
    
        self.remainWrfNum += 1
        pkt = Workflow(self.usrcurrentTime[usr], wrf, app_id, usr, self.set.dataset.wsetSlowestT[app_id], self.set.dueTimeCoef[usr, app_id], self.wrfIndex)
        self.usr_queues[usr].enqueue(pkt, self.usrcurrentTime[usr], None, usr, app_id)
        self.usrcurrentTime[usr] = next_arrivaltime
        self.totWrfNum -= 1
        self.wrfIndex +=1


    def reset(self, seed, test=False):
        random.seed(seed)
        np.random.seed(seed)
        self._init(test=test)

    def input_task_rule(self, rule):
        self.TaskRule = rule



    def get_nextWrfFromUsr(self):
        usrInd = np.argmin(self.firstusrWrfGenTime)
        firstPktTime = self.firstusrWrfGenTime[usrInd]
        return usrInd, firstPktTime

    def get_nextWrfFromCON(self):
        if len(self.firstconWrfLeaveTime) > 0:
            conInd = min(self.firstconWrfLeaveTime, key=self.firstconWrfLeaveTime.get)
            firstPktTime = self.firstconWrfLeaveTime[conInd]
            return conInd, firstPktTime
        else:
            return None, math.inf

    def get_nextTimeStep(self):
        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep
        tempnextloc, tempnextTimeStep = self.get_nextWrfFromUsr()  
        tempnextloc1, tempnextTimeStep1 = self.get_nextWrfFromCON()
        if tempnextTimeStep > tempnextTimeStep1:
            self.nextUsr, self.nextTimeStep = tempnextloc1, tempnextTimeStep1  
            self.nextisUsr = False
            self.nextWrf, self.finishTask = self.con_queues[self.nextUsr].get_firstDequeueTask()
        else:
            if tempnextTimeStep == math.inf:
                self.nextTimeStep = None
                self.nextUsr = None
                self.nextWrf = None
                self.nextisUsr = True
            else:
                self.nextUsr, self.nextTimeStep = tempnextloc, tempnextTimeStep
                self.nextisUsr = True
                self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt()



    def record_a_completed_workflow(self, ddl_penalty):

        if self.set.is_wf_trace_record:        
            Workflow_Infos = [self.nextWrf.appArivalIndex, self.nextWrf.appID,
                            self.nextWrf.generateTime, self.nextTimeStep, self.nextWrf.deadlineTime, ddl_penalty]

            for task in range(len(self.nextWrf.executeTime)):

                Task_Infos = [task, self.nextWrf.app.nodes[task]['process_time'], self.nextWrf.executeTime[task], 
                            self.nextWrf.readyTime[task], self.nextWrf.enqueueTime[task], self.nextWrf.dequeueTime[task]]

                CON_Infos = self.CONrentInfos[self.nextWrf.processDC[task]] + [self.nextWrf.pendingIndexOnDC[task]]

                write_csv_data(self.pkt_trace_file, Workflow_Infos + Task_Infos + CON_Infos)


    def step(self, pre_timestape, action=None):
        """
        get the response time of all request that generated when initialization
        """
        timestamp = False
        done = False
        reward = 0
        self.step_response = np.array([])
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
                selected_conid = random.choices(cand_con, weights=ratio, k=1)[0]
            else:
                selected_conid = cand_con[0]
            
            self.nextWrf.update_mapping(self.PrenextTask, self.con_queues[selected_conid].pm.get_pmid())
            comm_time = self.nextWrf.get_comm_time(self.PrenextTask, self.con_queues)

            reward = 0

            self.nextTimeStep += comm_time
            self.PrenextTimeStep = self.nextTimeStep
            parentTasks = self.nextWrf.get_allpreviousTask(self.PrenextTask)
            if len(parentTasks) == len(self.nextWrf.completeTaskSet(parentTasks)):
                process_time =  self.con_queues[selected_conid].task_enqueue(self.PrenextTask, self.PrenextTimeStep, self.nextWrf)
                self.CONexecHours += (process_time / 3600)    
                self.firstconWrfLeaveTime[selected_conid] = self.con_queues[selected_conid].get_firstTaskDequeueTime()

            if self.isDequeue:
                if self.nextisUsr:
                    self.nextWrf.update_dequeueTime(self.PrenextTimeStep, self.finishTask)
                    _, _ = self.usr_queues[self.PrenextUsr].dequeue()
                    self.firstusrWrfGenTime[self.PrenextUsr] = self.usr_queues[self.PrenextUsr].getFirstPktEnqueueTime() 

                    self.stat.add_app_arrival_rate(self.PrenextUsr, self.nextWrf.get_appID(), self.nextWrf.get_generateTime())
                else:
                    _, _ = self.con_queues[self.PrenextUsr].task_dequeue()
                    self.firstconWrfLeaveTime[self.PrenextUsr] = self.con_queues[self.PrenextUsr].get_firstTaskDequeueTime()


            temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

            if len(temp_Children_finishTask) > 0:
                self.dispatchParallelTaskNum += 1
            
            
            while True: 
                
                while len(temp_Children_finishTask) == 0:
                    if self.nextisUsr:
                        print('self.nextisUsr maybe wrong')
                    _, app = self.con_queues[self.nextUsr].task_dequeue()  
                    self.firstconWrfLeaveTime[self.nextUsr] = self.con_queues[self.nextUsr].get_firstTaskDequeueTime() 
                    if self.nextWrf.is_completeTaskSet(self.nextWrf.get_allTask()):
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

                    else:
                        _, _ = self.con_queues[self.nextUsr].task_dequeue()
                        self.firstconWrfLeaveTime[self.nextUsr] = self.con_queues[self.nextUsr].get_firstTaskDequeueTime()
                        self.get_nextTimeStep() 
                     
                        self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask) 
                        self.dispatchParallelTaskNum = 0                     
                        if self.nextTimeStep is not None:
                            temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)                                

                else:
                    if not self.isDequeue:
                        print('self.isDequeue maybe wrong')  
                    self.get_nextTimeStep()
                
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                    self.dispatchParallelTaskNum = 0
                    if self.nextTimeStep is not None:
                        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

            self.numTimestep = self.numTimestep + 1
            self.notNormalized_arr_hist = self.stat.update_arrival_rate_history()
            
            if self.remainWrfNum == 0:
                if len(self.firstconWrfLeaveTime) == 0:
                    done = True
                elif next(iter(self.firstconWrfLeaveTime.values())) == math.inf and list(self.firstconWrfLeaveTime.values()).count(next(iter(self.firstconWrfLeaveTime.values()))) == len(self.firstconWrfLeaveTime):
                    done = True
   
            if (self.PrenextTimeStep - pre_timestape) >= 180000:  
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
                cout_vm = 0
                cout_pm = 0
                self.replica_number.append(len(self.con_queues) - self.num_app)

                replica_num = {}
                for key, con_list in self.map_con_type_id.items():
                    replica_num[key] = len(con_list) - 1
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
                

                self.episode_info = {"CON_execHour": self.CONexecHours, "average_resptime": np.mean(self.response_time), "99_percentile": np.percentile(self.response_time, 99),
                        "VM_cost": self.total_cost, "SLA_penalty": self.SLApenalty, "missDeadlineNum": self.missDeadlineNum,
                        "response_list": self.response_time, "mean_step_resptime": self.mean_step_resptime,
                        "step_cost": self.step_cost, "replica_number": replica_num}
        

        return reward, done, self.response_time, self.total_cost

    def update_CONcost(self, dc, cpu, add=True):
        if add:
            temp = 1
        else:
            temp = 0
        self.CONcost += temp * self.set.dataset.conPrice[cpu]
        self.CONrentHours += temp


    def calculate_penalty(self, app, respTime):
        threshold = self.deadline
        if respTime < threshold or round(respTime - threshold,5) == 0:
            return 0
        else:
            self.missDeadlineNum += 1
            self.step_missdealine += 1
            return (respTime / self.deadline)

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
        scaling_num = scaling
        action_list.append(scaling_num)
        if scaling_num > 0:
            max_vcpu_add = con.max_scal_vcpu
            if max_vcpu_add >= scaling_num:
                con.v_scaling(scaling_num, self.con_queues, self.map_con_type_id, self.vm_map_id_vcpu, self.PrenextTimeStep, self.app_instances)
            else:
                con.v_scaling(max_vcpu_add, self.con_queues, self.map_con_type_id, self.vm_map_id_vcpu, self.PrenextTimeStep, self.app_instances)
                replica_con = con.h_scaling(0, scaling_num - max_vcpu_add, self.con_queues, self.con_queues_id, self.map_con_type_id, self.PrenextTimeStep, self.firstconWrfLeaveTime, self.app_instances)
                self.deploy_con_vm(replica_con)

        elif scaling_num < 0:
            util = con.get_utilization(self.nextWrf)
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

        priority = vm_remaining[available_index] / container.get_vcpu()
        selected_vm_index = available_index[0][np.argmin(priority)]
        if selected_vm_index > len(self.vm_queues) - 1:
            selected_vm_id = selected_vm_index - len(self.vm_queues)
            new_vm_id = max(self.vm_map_id_vcpu.keys()) + 1
            new_vm_vcpu = self.vm_vcpu_types[selected_vm_id]
            new_vm_price = self.vm_prices[new_vm_vcpu]
            new_vm = VM(new_vm_id, new_vm_vcpu, self.PrenextTimeStep, self.PrenextTimeStep, new_vm_price, [])
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
                self.firstconWrfLeaveTime[i] = new_container.get_firstTaskDequeueTime()
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
                vm_vcpu = 16
                vm_price = self.vm_prices[16]
                new_vm = VM(i, vm_vcpu, self.PrenextTimeStep, self.PrenextTimeStep, vm_price, [])
                
                self.vm_queues = np.append(self.vm_queues, new_vm)
                self.vm_map_id_vcpu[i] = vm_vcpu
                self.vm_queues_vcpu.append(vm_vcpu)

            type = 0
            con_vcpu = 1
            for i in range(app_type):
                type = i
                new_container = Container(i, type, con_vcpu, self.nextTimeStep, self.TaskRule)
                if type in self.map_con_type_id:
                    self.map_con_type_id[type].append(i)
                else:
                    self.map_con_type_id[type] = [i]

                self.con_queues[i] = new_container
                self.firstconWrfLeaveTime[i] = new_container.get_firstTaskDequeueTime()
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

        return ob
    
    def layer_graph_construct(self):
        return graph_construct(self.pm_queues, self.vm_queues, self.con_queues, self.app_instances), self.set.Workload

    def normalize_z_score(self, arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        return (arr - mean) / std_dev
        






    