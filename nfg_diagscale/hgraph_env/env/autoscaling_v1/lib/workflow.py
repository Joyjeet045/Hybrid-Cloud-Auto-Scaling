
class Workflow:
    def __init__(self, generateTime1, app, appID, originDC, time, ratio, index):
        self.generateTime = generateTime1
        self.origin = originDC # comes from which user
        self.app = app  # e.g., w1 = [n1, n2, n3, n4, n5, n6],  n1 = [[], [2], 's1'], n2 = [[1], [3], 's14']
        self.appID = appID ## Workflow is workflow；AppID is used to differentiate different workflows. A workflow includes a number of different tasks 
        self.appArivalIndex = index
        self.readyTime = {} # ready time
        self.enqueueTime = {}  # {nodeID: enqueueTime for the service}
        self.dequeueTime = {}  # {nodeID: dequeueTime for each service} to ensure all parallel services are completed before the aggregated service starts
        self.pendingIndexOnDC ={}
        self.processDC = {}  # {nodeID: dc_ind} ensure parallel services are aggregated to the same destination service
        self.executeTime = {}
        self.queuingTask = []  # tasks that are currently in queue. Mostly for parallel services
        self.maxProcessTime = time  # the max processing time for the app if parallel processing is enabled and no propagation latency
        self.dueTimeCoef = ratio
        self.deadlineTime = self.maxProcessTime * self.dueTimeCoef + self.generateTime

        self.map_task_pm = {}    # the mapping of container that process each task

    def __lt__(self, other):  
        return self.generateTime < other.generateTime

    def get_completeTaskNum(self):
        return len(self.dequeueTime)

    def get_maxProcessTime(self):
        return self.maxProcessTime
    
    def get_comm_time(self, task_id, con_queues):
        parent_list = list(self.app.predecessors(task_id))
        comm_time = 0
        tar_pm_id = self.map_task_pm[task_id]
        # tar_pm = con_queues[tar_con_id].get_pm()
        for p_id in parent_list:
            # print(task_id, con_queues)
            par_pm_id = self.map_task_pm[p_id]
            # par_pm = con_queues[par_con].get_pm()
            if tar_pm_id != par_pm_id:
                if float(self.app.get_edge_data(p_id, task_id).get('weight', None)) > comm_time:
                    comm_time = float(self.app.get_edge_data(p_id, task_id).get('weight', None))

        return comm_time

    # return True if tasks in s_list are completed, otherwise False
    def is_completeTaskSet(self, s_list):
        # print(self.dequeueTime)
        if set(s_list).issubset(set(self.dequeueTime.keys())):
            return True
        else:
            return False

    # return the set of tasks in s_list that are completed
    def completeTaskSet(self, s_list):
        return set(s_list).intersection(set(self.dequeueTime.keys()))
    

    def update_pendingIndexCON(self, task, pendingTndex):
        self.pendingIndexOnDC[task] = pendingTndex

    def update_mapping(self, task_id, pm_id):
        self.map_task_pm[task_id] = pm_id

    # time: enqueue time for "service" which is allocated in "dc"
    def update_enqueueTime(self, time, task, vmID):
        if task not in self.enqueueTime:
            self.readyTime[task] = time
            self.enqueueTime[task] = time
            self.processDC[task] = vmID  # guarantee results from all parallel services aggregating at the same destination

        elif time > self.enqueueTime[task]:
            self.enqueueTime[task] = time
        # print(f"{self.appArivalIndex} enqueueTime = {self.enqueueTime}")

    def update_dequeueTime(self, time, task):
        self.dequeueTime[task] = time

    def update_executeTime(self, time, task):
        self.executeTime[task] = time

    def get_executeTime(self, task):
        return self.executeTime[task]

    def get_appArivalIndex(self):
        return self.appArivalIndex

    def add_queuingTask(self, task):
        self.queuingTask.append(task)

    def remove_queuingTask(self, task):
        self.queuingTask.remove(task)

    def get_taskDC(self, task):
        if task in self.processDC:
            return self.processDC[task]
        else:
            return None

    def get_generateTime(self):
        return self.generateTime

    def get_enqueueTime(self, task):
        return self.enqueueTime[task]

    def get_readyTime(self, task):
        return self.readyTime[task]

    def get_originDC(self):
        return self.origin

    def get_appID(self):
        return self.appID

    def get_taskProcessTime(self, task):
        return self.app.nodes[task]['processTime']  # networkx
        # return self.app.vs[task]['processTime']  # igraph

    # return the next task list given the current task ID
    def get_allnextTask(self, task):
        if task is None:
            root = [n for n, d in self.app.in_degree() if d == 0]  # networkx get root
            return root
        else:
            return list(self.app.successors(task))

    def get_NumofSuccessors(self, task):          
        node_succ = self.get_allnextTask(task)
        return len(node_succ)

    def get_Deadline(self):              
        return self.deadlineTime 
 
    # return the parent tasks given the current task
    def get_allpreviousTask(self, task):
        if task is None:
            return []
        else:
            return list(self.app.predecessors(task))

    def get_totNumofTask(self):
        return self.app.number_of_nodes()  # networkx
        # return self.app.vcount()    # igraph

    def get_allTask(self):
        return list(self.app.nodes)  # networkx
        # return self.app.vs.indices   # igraph
