from env.autoscaling_v1.lib.cloud_env_maxPktNum import cloud_simulator
import numpy as np
import env.autoscaling_v1.lib.dataset as dataset


class ASEnv(cloud_simulator):
    def __init__(self, name, args):

        # Application type


        # Setup
        config = {"seed": args.seed, "envid": 0,
                  "app_size": args.app_size, "app_num": args.app_num, 
                  "app_types": args.app_size, "workload_pattern": args.workload_pattern,
                  "budget": args.budget}

        super(ASEnv, self).__init__(config)
        super(ASEnv, self)._init()
        self.name = name
        

    def reset(self, seed=None, test=False):
        super(ASEnv, self).reset(seed, test)

        s, workload = self.layer_graph_construct()

        # predicted_workload = torch.tensor(0)

        # s = s + (predicted_workload, )

        return s
    
    # def step(self, action_step, action=None):
    def step(self, action=None):
        reward, done, ar, c,  = super(ASEnv, self).step(self.nextTimeStep, action)

        s, workload = self.layer_graph_construct()
        
        # # future predict
        # if action_step > 1:
        #     predicted_workload = (workload[action_step] + workload[action_step - 1]) / 2
        # else:
        #     predicted_workload = 0
        
        # predicted_workload = torch.tensor(predicted_workload)

        # s = s + (predicted_workload, )

        return s, reward, done, ar
    

    def get_agent_ids(self):
        return ["0"]

    def close(self):
        super(ASEnv, self).close()
