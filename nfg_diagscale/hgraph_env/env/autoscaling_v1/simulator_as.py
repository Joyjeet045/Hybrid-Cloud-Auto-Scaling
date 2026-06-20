from env.autoscaling_v1.lib.cloud_env_maxPktNum import cloud_simulator
import numpy as np
import env.autoscaling_v1.lib.dataset as dataset


class ASEnv(cloud_simulator):
    def __init__(self, name, args):



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



        return s
    
    def step(self, action=None):
        reward, done, ar, c,  = super(ASEnv, self).step(self.nextTimeStep, action)

        s, workload = self.layer_graph_construct()
        
        


        return s, reward, done, ar
    

    def get_agent_ids(self):
        return ["0"]

    def close(self):
        super(ASEnv, self).close()
