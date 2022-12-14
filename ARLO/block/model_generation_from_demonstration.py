"""
This module contains the implementation of the Class ModelGenerationFromDemonstration.
The Class ModelGenerationMushroomFromDemonstration inherits from the Class ModelGeneration.

The Class ModelGenerationFromDemonstration is a Class used to group all Classes that do ModelGenerationFromDemonstration.
"""

from abc import abstractmethod

import copy
import numpy as np
import matplotlib.pyplot as plt

from mushroom_rl.core.core import Core
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy, OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC, DDPG
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.parameters import LinearParameter
from ARLO.block.block_output import BlockOutput
from ARLO.block.model_generation import ModelGeneration
from ARLO.block.pretraining import Pretraining, PretrainingValue
from ARLO.hyperparameter.hyperparameter import Real, Integer, Categorical

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ModelGenerationFromDemonstration(ModelGeneration):
    
    def __repr__(self):
        return str(self.__class__.__name__)+'('+'eval_metric='+str(self.eval_metric)+', obj_name='+str(self.obj_name)\
        +', seeder='+ str(self.seeder)+', local_prng='+ str(self.local_prng)+', model='+str(self.model)\
        +', algo_params='+str(self.algo_params)+', log_mode='+str(self.log_mode)\
        +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
        +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
        +', works_on_online_rl='+str(self.works_on_online_rl)+', works_on_offline_rl='+str(self.works_on_offline_rl)\
        +', works_on_box_action_space='+str(self.works_on_box_action_space)\
        +', works_on_discrete_action_space='+str(self.works_on_discrete_action_space)\
        +', works_on_box_observation_space='+str(self.works_on_box_observation_space)\
        +', works_on_discrete_observation_space='+str(self.works_on_discrete_observation_space)\
        +', pipeline_type='+str(self.pipeline_type)+', is_learn_successful='+str(self.is_learn_successful)\
        +', is_parametrised='+str(self.is_parametrised)+', block_eval='+str(self.block_eval)\
        +', algo_params_upon_instantiation='+str(self.algo_params_upon_instantiation)\
        +', logger='+str(self.logger)+', fully_instantiated='+str(self.fully_instantiated)\
        +', info_MDP='+str(self.info_MDP)+', regressor_type='+str(self.regressor_type)+')'  

    def learn(self, train_data=None, env=None): 
        """
        Parameters
        ----------
        train_data: This can be a dataset that will be used for training. It must be an object of a Class inheriting from Class
                    BaseDataSet.
                    
                    The default is None.
                                              
        env: This must be a simulator/environment. It must be an object of a Class inheriting from Class BaseEnvironment.
        
             The default is None.
        
        Returns
        -------
        This method calls the method learn() implemented in the Class Block: if such call was successful the method fit() of the 
        algo_object is called onto the dataset contained in the starting_train_data, else an empty object of Class BlockOutput is 
        returned.
        res: This is an object of Class BlockOutput containing the learnt policy. If something went wrong in the execution of the
             method the object of Class BlockOutput is empty.
        """
        
        #resets is_learn_successful to False, checks pipeline_type, checks the types of train_data and env, and makes sure that 
        #they are not both None and selects the right inputs:
        starting_train_data_and_env = super().learn(train_data=train_data, env=env)
        
        #if super().learn() returned something that is of Class BlockOutput it means that up in the chain there was an error and
        #i need to return here the empty object of Class BlockOutput
        if(isinstance(starting_train_data_and_env, BlockOutput)):
            return BlockOutput(obj_name=self.obj_name)
        
        #since this is a block from demonstration we have both expert_train_data, which is the first element of the list 
        #starting_train_data_and_env, and an environment, which is the second element from the list
        expert_train_data = starting_train_data_and_env[0]
        starting_env = starting_train_data_and_env[1]
        
        """"""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""
        self.logger.info(msg='Now learning the following block: '+self.pretrain_block.obj_name)   
        can_proceed = self.pretrain_block.pre_learn_check(train_data=expert_train_data, env=env)
                 
        if(not can_proceed):
            self.is_learn_successful = False
            err_msg = 'The current block failed the check on consistency with the pipeline and the environment spaces!'
            self.logger.error(msg=err_msg)
            return BlockOutput(obj_name=self.obj_name)
        
        #fully instantiate model generation tmp block with the info of the MDP of the environment produced by the 
        #previous block. The other blocks are already fully instantiated. I simply pass env since this is kept updated
        #with the results of all blocks:
        if(isinstance(self.pretrain_block, Pretraining) and (not self.pretrain_block.fully_instantiated)):
            block_fully_instantiated = self.pretrain_block.full_block_instantiation(info_MDP = self.info_MDP)
        
        #expert_train_data is an object of Class TabularDataSet: so we pick the member dataset to feed to the Behavioral Cloning initialization
        #method, that initializes the policy and/or the Q function approximator from expert demonstrations         
        #      
        tmp_res = self.pretrain_block.learn(train_data=expert_train_data)

        #If the block was not learned successfully the pipeline learning process needs to finish here
        if(not self.pretrain_block.is_learn_successful):
            self.is_learn_successful = False
            self.logger.error(msg='There was an error learning the current block!')
            return BlockOutput(obj_name=self.obj_name)

        # #tmp_res must be an object of Class BlockOutput
        # if(not isinstance(tmp_res, BlockOutput)):
        #     self.is_learn_successful = False
        #     self.logger.error(msg='\'tmp_res\' must be an object of Class \'BlockOutput\'!')
        #     return BlockOutput(obj_name=self.obj_name)
            
        # #in online RL every block must return only one output:
        # if(tmp_res.n_outputs != 1):
        #     self.is_learn_successful = False
        #     self.logger.error(msg='In onlineRL each block must return exactly one output! This block did not!')   
        #     return BlockOutput(obj_name=self.obj_name)


        """"""""""""""""""""""""
        self.initialize_agent()
        """"""""""""""""""""""""
            

        res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode,
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity,                        
                              policy=self.construct_policy(policy=self.algo_object.policy, regressor_type=self.regressor_type,
                              approximator=self.algo_object.approximator))
        if(self.deterministic_output_policy):
            #If this method is called then in the metric DiscountedReward you can use batch_eval
            res.make_policy_deterministic()
        #if i reach this point the block was learnt successfully. Each block in onlineRL can either return a
        #BaseEnvironment or a BasePolicy:  
        if(isinstance(self.pretrain_block, Pretraining)):
            self.logger.info(msg='Starting evaluation (BC from expert demonstrations): ')    
            #ModelGeneration blocks return a BasePolicy:                                        
            # policy = self.pretrain_block.policy         
            #computes policy evaluation:
            self.logger.info(msg='Now evaluating learnt policy...')
            res_eval = self.eval_metric.evaluate(block_res=res, block=self.pretrain_block, env=env)                    
            policy_eval = {'eval_mean': res_eval, 'eval_var': self.eval_metric.eval_var}
            self.logger.info(msg='Mean of policy evaluation: '+str(policy_eval['eval_mean']))
            self.logger.info(msg='Standard deviation of policy evaluation: '+str(np.sqrt(policy_eval['eval_var'])))
        
        # starting_eval = self.eval_metric.evaluate(block_res=res, env=starting_env)
        
        self.dict_of_evals = {}
        #update dict_of_evals:        
        self.update_dict_of_evals(current_epoch=0, single_episodes_eval=self.eval_metric.single_episode_evaluations, 
                                  env=starting_env)
        
        
        if('only_BC' in self.algo_params.keys()):
            self.is_learn_successful = True
            return res

        #if i have a method called _default_network() it means I am using a PyTorch network. This is ok: MushroomRL does not allow
        #the use of other deep learning frameworks so there are not going to be issues:
        if(hasattr(self, '_default_network')):
            #sets torch number of threads
            torch.set_num_threads(self.n_jobs)
        
        #create core object with starting_env:
        self._create_core(env=starting_env)

        #if the algorithm has a replay buffer i fill it randomly:
        if('initial_replay_size' in list(self.algo_params.keys())):
            #fill replay memory with random dataset
            self.core.learn(n_steps=self.algo_params['initial_replay_size'].current_actual_value, 
                            n_steps_per_fit=self.algo_params['initial_replay_size'].current_actual_value, 
                            quiet=True)     
           

        for n_epoch in range(self.algo_params['n_epochs'].current_actual_value):
            self.logger.info(msg='Epoch: '+str(n_epoch))
            
            #learning step:
            self.core.learn(n_steps=self.algo_params['n_steps'].current_actual_value, 
                            n_steps_per_fit=self.algo_params['n_steps_per_fit'].current_actual_value, 
                            n_episodes=self.algo_params['n_episodes'].current_actual_value, 
                            n_episodes_per_fit=self.algo_params['n_episodes_per_fit'].current_actual_value, 
                            quiet=True)
                             
            #evaluation step:
            res = BlockOutput(obj_name=str(self.obj_name)+'_result', log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity,
                              policy=self.construct_policy(policy=self.algo_object.policy, regressor_type=self.regressor_type))
            
            if(self.deterministic_output_policy):
                #If this method is called then in the metric DiscountedReward you can use batch_eval
                res.make_policy_deterministic()
            
            tmp_eval = self.eval_metric.evaluate(block_res=res, env=starting_env)
            
            self.logger.info(msg='Current evaluation: '+str(tmp_eval))
                     
            #update dict_of_evals 
            self.update_dict_of_evals(current_epoch=n_epoch+1, single_episodes_eval=self.eval_metric.single_episode_evaluations, 
                                      env=starting_env)    
                                      
        self.is_learn_successful = True 
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object learnt successfully!')
        return res
        
    @abstractmethod
    def initialize_agent(self): 
        """
        This method uses the expert demonstration to initialize the policy and/or the Q function approximator.
        The approximator is then set into the algo_object, the object that contains the algorithm from MushroomRL.
        The method has a specific implementation for every "from demonstration" algorithm, since they all have a customized loss
        """
        raise NotImplementedError
        
    def plot_dict_of_evals(self):       
        """
        This method plots and saves the dict_of_evals of the block.
        """
        
        x = np.array(list(self.dict_of_evals.keys()))
        if(len(x) == 0):
            exc_msg = 'The \'dict_of_evals\' is empty!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        evals_values = list(self.dict_of_evals.values())
        y = np.array([np.mean(evals_values[i]) for i in range(len(evals_values))])
        
        std_dev = np.array([np.std(evals_values[i]) for i in range(len(evals_values))])
        
        plt.figure()
        plt.xlabel('Environment Steps')
        plt.ylabel('Average Discounted Reward')
        plt.title('Average Discounted Reward and Standard Deviation for '+str(self.obj_name))
        plt.grid(True)
        plt.plot(x, y, color='#FF9860')
        if(len(evals_values[0]) > 1):
            plt.fill_between(x, y-std_dev, y+std_dev, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9860')
        plt.show()    
    
    def update_dict_of_evals(self, current_epoch, single_episodes_eval, env):
        """
        Parameters
        ----------
        current_epoch: This is a non-negative integer and it represents the current epoch.
        
        single_episodes_eval: This is a list of floats containing the evaluation of the agent over the single episodes, for as 
                              many episodes as specified by the eval_metric. 
        
        env: This is the environment in which we are acting. It must be an object of a Class inheriting from the Class 
             BaseEnvironmnet.
            
        This method updates the dict_of_evals.
        """
   
        number_of_steps = self.algo_params['n_steps'].current_actual_value
        if(number_of_steps is None):
            number_of_steps = env.horizon*self.algo_params['n_episodes'].current_actual_value       
        
        new_dict = {current_epoch*number_of_steps: single_episodes_eval}  
                        
        if(len(list(self.dict_of_evals.keys())) == 0):
            self.dict_of_evals = new_dict
        else:
            self.dict_of_evals = {**self.dict_of_evals, **new_dict}
       
    def _create_core(self, env):
        """
        Parameters
        ---------
        env: This is the environment in which we are acting. It must be an object of a Class inheriting from the Class 
             BaseEnvironmnet.
             
        This method updates the value of the core member by creating an object of Class mushroom_rl.core.Core.
        """
             
        self.core = Core(agent=self.algo_object, mdp=env) 
        
    def analyse(self):
        """
        This method is not yet implemented.
        """
        
        raise NotImplementedError
        
    def save(self):
        """
        This method saves to a pickle file the object. Before saving it the core and the algo_object are cleared since these two
        can weigh quite a bit.
        """
        
        #clean up the core and algo_object: these two, in algorithms that have ReplayMemory, are going to make the output file, 
        #created when calling the method save, be very heavy. 
        
        #I need to clean these in a deep copy: otherwise erasing algo_object I cannot call twice in a row the learn method 
        #because the algo_object is set in the method set_params
        
        copy_to_save = copy.deepcopy(self)
        
        copy_to_save.core = None
        copy_to_save.algo_object = None
        
        #calls method save() implemented in base Class ModelGeneration of the instance copy_to_save
        super(ModelGenerationFromDemonstration, copy_to_save).save()
   

class ModelGenerationMushroomOnlineValueBased(ModelGenerationFromDemonstration):


    def initialize_agent(self):
        pass
        
class ModelGenerationMushroomOnlinePolicyBased(ModelGenerationFromDemonstration):

    def __init__():
        pass
    
    def initialize_agent(self):
        pass
   
class ModelGenerationMushroomOnlineDQNfD(ModelGenerationMushroomOnlineValueBased):
    """
    This Class implements a specific online model generation algorithm: DQN. This Class wraps the DQN method implemented in 
    MushroomRL.
    
    cf. https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/algorithms/value/dqn/dqn.py
    
    This Class inherits from the Class ModelGenerationMushroomOnline.
    """
    
    def __init__(self, eval_metric, obj_name, regressor_type='q_regressor', seeder=2, algo_params=None, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process', deterministic_output_policy=True):
        """        
        Parameters
        ----------
        algo_params: This is either None or a dictionary containing all the needed parameters.
                            
                     The default is None.        
                                                 
                     If None then the following parameters will be used:
                     'epsilon': LinearParameter(value=1, threshold_value=0.01, n=1000000)
                     'policy': EpsGreedy(epsilon=LinearParameter(value=1, threshold_value=0.01, n=1000000)),
                     'approximator': TorchApproximator,
                     'network': one hidden layer, 16 neurons, 
                     'input_shape': self.info_MDP.observation_space.shape,
                     'n_actions': self.info_MDP.action_space.n, 
                     'output_shape': (self.info_MDP.action_space.n,),
                     'optimizer': Adam,
                     'lr': 0.0001,
                     'critic_loss': smooth_l1_loss,
                     'batch_size': 32,
                     'target_update_frequency': 250,
                     'replay_memory': ReplayMemory, 
                     'initial_replay_size': 50000,
                     'max_replay_size': 1000000,
                     'clip_reward': False,
                     'n_epochs': 10,
                     'n_steps': None,
                     'n_steps_per_fit': None,
                     'n_episodes': 500,
                     'n_episodes_per_fit': 50,
        
        regressor_type: This is a string and it can either be: 'action_regressor', 'q_regressor' or 'generic_regressor'. This is
                        used to pick one of the 3 possible kind of regressor made available by MushroomRL.
                        
                        Note that if you want to use a 'q_regressor' then the picked regressor must be able to perform 
                        multi-target regression, as a single regressor is used for all actions. 
                        
                        The default is 'q_regressor'.
                        
        deterministic_output_policy: If this is True then the output policy will be rendered deterministic else if False nothing
                                     will be done. Note that the policy is made deterministic only at the end of the learn()
                                     method.
                        
        Non-Parameters Members
        ----------------------
        fully_instantiated: This is True if the block is fully instantiated, False otherwise. It is mainly used to make sure that 
                            when we call the learn method the model generation blocks have been fully instantiated as they 
                            undergo two stage initialisation being info_MDP unknown at the beginning of the pipeline.
                            
        info_MDP: This is a dictionary compliant with the parameters needed in input to all MushroomRL model generation 
                  algorithms. It containts the observation space, the action space, the MDP horizon and the MDP gamma.
        
        algo_object: This is the object containing the actual model generation algorithm.
                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of
                                        algo_params that the object got upon creation. This is needed for re-loading
                                        objects.
                                        
        model: This is used in set_params in the generic Class ModelGenerationMushroomOnline. With this member we avoid 
               re-writing for each Class inheriting from the Class ModelGenerationMushroomOnline the set_params method. 
               In this Class this member equals to DQN, which is the Class of MushroomRL implementing DQN.
        
        core: This is used to contain the Core object of MushroomRL needed to run online RL algorithms.
        
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = True
        self.works_on_offline_rl = False
        self.works_on_box_action_space = False
        self.works_on_discrete_action_space = True
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        self.regressor_type = regressor_type
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
        
        self.deterministic_output_policy = deterministic_output_policy

        self.fully_instantiated = False               
        self.info_MDP = None 
        self.algo_object = None
        """"""
        self.pretrain_block = None
        """"""
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)

        self.model = DQN
        
        self.core = None
               
        #seeds torch
        torch.manual_seed(self.seeder)
        torch.cuda.manual_seed(self.seeder)

        #this seeding is needed for the policy of MushroomRL. Indeed the evaluation at the start of the learn method is done 
        #using the policy and in the method draw_action, np.random is called! 
        np.random.seed(self.seeder)
        
    def _default_network(self):
        """
        This method creates a default Network with 1 hidden layer and ReLU activation functions.
        
        Returns
        -------
        Network: the Class wrapper representing the default network.
        """
        
        class Network(nn.Module):
            def __init__(self, input_shape, output_shape, **kwargs):
                super().__init__()
               
                n_input = input_shape[-1]
                n_output = output_shape[0]

                self.hl0 = nn.Linear(n_input, 16)
                self.hl1 = nn.Linear(16, 16)
                self.hl2 = nn.Linear(16, n_output)
                
                nn.init.xavier_uniform_(self.hl0.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
    
            def forward(self, state, action=None):
                h = F.relu(self.hl0(state.float()))
                h = F.relu(self.hl1(h))
                q = self.hl2(h)

                if action is None:
                    return q
                else:
                    q_acted = torch.squeeze(q.gather(1, action.long()))            
                    return q_acted
                    
        return Network
    
    def full_block_instantiation(self, info_MDP):
        """
        Parameters
        ----------
        info_MDP: This is an object of Class mushroom_rl.environment.MDPInfo. It contains the action and observation spaces, 
                  gamma and the horizon of the MDP.
        
        Returns
        -------
        This method returns True if the algo_params were set successfully, and False otherwise.
        """
        
        self.info_MDP = info_MDP
        
        if(self.algo_params is None):   
            approximator = Categorical(hp_name='approximator', obj_name='approximator_'+str(self.model.__name__), 
                                       current_actual_value=TorchApproximator)
            
            network = Categorical(hp_name='network', obj_name='network_'+str(self.model.__name__), 
                                  current_actual_value=self._default_network())
            
            optimizer_class = Categorical(hp_name='class', obj_name='optimizer_class_'+str(self.model.__name__), 
                                          current_actual_value=optim.Adam) 
            
            lr = Real(hp_name='lr', obj_name='optimizer_lr_'+str(self.model.__name__), 
                      current_actual_value=0.0001, range_of_values=[1e-5, 1e-3], to_mutate=True, seeder=self.seeder, 
                      log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
                        
            critic_loss = Categorical(hp_name='critic_loss', obj_name='critic_loss_'+str(self.model.__name__), 
                                      current_actual_value=F.smooth_l1_loss)
            
            batch_size = Integer(hp_name='batch_size', obj_name='batch_size_'+str(self.model.__name__), 
                                 current_actual_value=32, range_of_values=[16, 128], to_mutate=True, seeder=self.seeder, 
                                 log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            target_update_frequency = Integer(hp_name='target_update_frequency', current_actual_value=250, 
                                              range_of_values=[100,1000], to_mutate=True, 
                                              obj_name='target_update_frequency_'+str(self.model.__name__), seeder=self.seeder, 
                                              log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, 
                                              verbosity=self.verbosity)
            
            initial_replay_size = Integer(hp_name='initial_replay_size', current_actual_value=50000, 
                                          range_of_values=[10000, 100000], 
                                          obj_name='initial_replay_size_'+str(self.model.__name__),
                                          to_mutate=True, seeder=self.seeder, log_mode=self.log_mode, 
                                          checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            max_replay_size = Integer(hp_name='max_replay_size', current_actual_value=1000000, range_of_values=[10000, 1000000],
                                      obj_name='max_replay_size_'+str(self.model.__name__), to_mutate=True, seeder=self.seeder, 
                                      log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, 
                                      verbosity=self.verbosity)
            
            replay_memory = Categorical(hp_name='replay_memory', obj_name='replay_memory_'+str(self.model.__name__), 
                                        current_actual_value=ReplayMemory(initial_size=initial_replay_size.current_actual_value, 
                                                                          max_size=max_replay_size.current_actual_value))
                
            clip_reward = Categorical(hp_name='clip_reward', obj_name='clip_reward_'+str(self.model.__name__), 
                                      current_actual_value=False, possible_values=[True, False], to_mutate=True, 
                                      seeder=self.seeder, log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, 
                                      verbosity=self.verbosity)
            
            n_epochs = Integer(hp_name='n_epochs', current_actual_value=10, range_of_values=[1,50], to_mutate=True, 
                               obj_name='n_epochs_'+str(self.model.__name__), seeder=self.seeder, log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            n_steps = Integer(hp_name='n_steps', current_actual_value=None, to_mutate=False, 
                              obj_name='n_steps_'+str(self.model.__name__), seeder=self.seeder, log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=None, 
                                      to_mutate=False, obj_name='n_steps_per_fit_'+str(self.model.__name__), seeder=self.seeder,
                                      log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, 
                                      verbosity=self.verbosity)
            
            n_episodes = Integer(hp_name='n_episodes', current_actual_value=500, range_of_values=[10,1000], to_mutate=True, 
                                 obj_name='n_episodes_'+str(self.model.__name__), seeder=self.seeder, log_mode=self.log_mode, 
                                 checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=1, range_of_values=[1,100], 
                                         to_mutate=True, obj_name='n_episodes_per_fit_'+str(self.model.__name__), 
                                         seeder=self.seeder, log_mode=self.log_mode, 
                                         checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
        
            epsilon = Categorical(hp_name='epsilon', obj_name='epsilon_'+str(self.model.__name__), 
                                  current_actual_value=LinearParameter(value=1, threshold_value=0.01, n=1000000))

            n_epochs_pretraining = Integer(hp_name='n_epochs_pretraining', current_actual_value=20000, range_of_values=[5000,1000000], to_mutate=True, 
                               obj_name='n_epochs_'+str(self.model.__name__), seeder=self.seeder, log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            lambda1 = Real(hp_name='lambda1', obj_name='lambda1_'+str(self.model.__name__), 
                      current_actual_value=1, range_of_values=[0.1, 1.5], to_mutate=True, seeder=self.seeder, 
                      log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)

            lambda2 = Real(hp_name='lambda2', obj_name='lambda2_'+str(self.model.__name__), 
                      current_actual_value=1, range_of_values=[0.1, 1.5], to_mutate=True, seeder=self.seeder, 
                      log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)

            lambda3 = Real(hp_name='lambda3', obj_name='lambda3_'+str(self.model.__name__), 
                      current_actual_value=1, range_of_values=[0.1, 1.5], to_mutate=True, seeder=self.seeder, 
                      log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)

            use_n_step = Categorical(hp_name='use_n_step', obj_name='critic_loss_'+str(self.model.__name__), 
                                      current_actual_value=True, possible_values=[True, False])

            n_step_lookahead = Integer(hp_name='n_epochs_pretraining', current_actual_value=10, range_of_values=[5,15], to_mutate=True, 
                               obj_name='n_epochs_'+str(self.model.__name__), seeder=self.seeder, log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            dict_of_params = {'approximator': approximator, 
                              'network': network, 
                              'class': optimizer_class, 
                              'lr': lr, 
                              'loss': critic_loss, 
                              'batch_size': batch_size, 
                              'target_update_frequency': target_update_frequency,
                              'replay_memory': replay_memory, 
                              'initial_replay_size': initial_replay_size,
                              'max_replay_size': max_replay_size,
                              'clip_reward': clip_reward,
                              'n_epochs': n_epochs,
                              'n_steps': n_steps,
                              'n_steps_per_fit': n_steps_per_fit,
                              'n_episodes': n_episodes,
                              'n_episodes_per_fit': n_episodes_per_fit,
                              'epsilon': epsilon,
                              'n_epochs_pretraining' : n_epochs_pretraining,
                              'lambda1': lambda1,
                              'lambda2': lambda2,
                              'lambda3': lambda3,
                              'use_n_step': use_n_step,
                              'n_step_lookahead': n_step_lookahead
                              }
            
            self.algo_params = dict_of_params
        
        is_set_param_success = self.set_params(new_params=self.algo_params)
        
        if(not is_set_param_success):
             err_msg = 'There was an error setting the parameters of a'+'\''+str(self.__class__.__name__)+'\' object!'
             self.logger.error(msg=err_msg)
             self.fully_instantiated = False
             self.is_learn_successful = False
             return False
           
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object fully instantiated!')
        self.fully_instantiated = True
        return True 
           
    def set_params(self, new_params):
        """
        Parameters
        ----------
        new_params: The new parameters to be used in the specific model generation algorithm. It must be a dictionary that does 
                    not contain any dictionaries(i.e: all parameters must be at the same level).
                                        
                    We need to create the dictionary in the right form for MushroomRL. Then it needs to update self.algo_params. 
                    Then it needs to update the object self.algo_object: to this we need to pass the actual values and not 
                    the Hyperparameter objects. 
                    
                    We call _select_current_actual_value_from_hp_classes: to this method we need to pass the dictionary already 
                    in its final form. 
        Returns
        -------
        bool: This method returns True if new_params is set correctly, and False otherwise.
        """
                   
        if(new_params is not None):                
            mdp_info = Categorical(hp_name='mdp_info', obj_name='mdp_info_'+str(self.model.__name__), 
                                   current_actual_value=self.info_MDP)
            
            input_shape = Categorical(hp_name='input_shape', obj_name='input_shape_'+str(self.model.__name__), 
                                      current_actual_value=self.info_MDP.observation_space.shape)
            
            if(self.regressor_type == 'action_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=(1,))
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=self.info_MDP.action_space.n)
            elif(self.regressor_type == 'q_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=(self.info_MDP.action_space.n,))
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=self.info_MDP.action_space.n)
            elif(self.regressor_type == 'generic_regressor'):
                output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                           current_actual_value=self.info_MDP.action_space.shape)
                #to have a generic regressor I must not specify n_actions
                n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                         current_actual_value=None)
            
            policy = Categorical(hp_name='policy', obj_name='policy_'+str(self.model.__name__), 
                                 current_actual_value=EpsGreedy(new_params['epsilon'].current_actual_value))
            
            tmp_structured_algo_params = {'mdp_info': mdp_info,
                                          'policy': policy,
                                          'approximator_params': {'input_shape': input_shape,
                                                                  'n_actions': n_actions,
                                                                  'output_shape': output_shape,
                                                                  'optimizer': {'class': None, 'params': {'lr': None}}, 
                                                                 }
                                         }
            tmp_pretrain_params = { 'mdp_info': mdp_info,
                                    'input_shape': input_shape,
                                    'n_actions': n_actions,
                                    'output_shape': output_shape,
                                    'class': None,
                                    'lr': None
                                    }
            
            
            for tmp_key in list(new_params.keys()):
                #i do not want to change mdp_info or policy                              
                if(tmp_key in ['batch_size', 'target_update_frequency', 'replay_memory', 
                               'initial_replay_size', 'max_replay_size', 'clip_reward']):
                    tmp_structured_algo_params.update({tmp_key: new_params[tmp_key]})
                
                if(tmp_key in ['approximator']):
                    tmp_structured_algo_params.update({tmp_key: new_params[tmp_key]})
                    tmp_pretrain_params.update({tmp_key:new_params[tmp_key]})

                if(tmp_key in ['network', 'loss']):
                    tmp_structured_algo_params['approximator_params'].update({tmp_key: new_params[tmp_key]})
                    tmp_pretrain_params.update({tmp_key:new_params[tmp_key]})


                if(tmp_key in ['class']):
                    tmp_structured_algo_params['approximator_params']['optimizer'].update({tmp_key: new_params[tmp_key]})  
                    tmp_pretrain_params.update({tmp_key:new_params[tmp_key]})

                if(tmp_key in ['lr']):
                    tmp_structured_algo_params['approximator_params']['optimizer']['params'].update({tmp_key: new_params[tmp_key]})                                                                   
                    tmp_pretrain_params.update({tmp_key:new_params[tmp_key]})
                
                if(tmp_key in ['n_epochs_pretraining', 'lambda1', 'lambda2', 'lambda3', 'use_n_step', 'n_step_lookahead']):
                    tmp_pretrain_params.update({tmp_key:new_params[tmp_key]})
            
            structured_dict_of_values = self._select_current_actual_value_from_hp_classes(params_structured_dict=
                                                                                          tmp_structured_algo_params)
            
            
            #i need to un-pack structured_dict_of_values for DQN  
            self.algo_object = DQN(**structured_dict_of_values)
            
            # pretrain_params = self._select_current_actual_value_from_hp_classes(params_structured_dict=tmp_structured_pretrain_params)
            self.pretrain_block = PretrainingValue(eval_metric = self.eval_metric, obj_name = self.obj_name+'_pretrain', algo_params=tmp_pretrain_params)
            
            final_dict_of_params = tmp_structured_algo_params
            #add n_epochs, n_steps, n_steps_per_fit, n_episodes, n_episodes_per_fit:
            dict_to_add = {'n_epochs': new_params['n_epochs'],
                           'n_steps': new_params['n_steps'],
                           'n_steps_per_fit': new_params['n_steps_per_fit'],
                           'n_episodes': new_params['n_episodes'],
                           'n_episodes_per_fit': new_params['n_episodes_per_fit'],
                           'epsilon': new_params['epsilon'],
                           'n_epochs_pretraining' : new_params['n_epochs_pretraining'],
                           'lambda1' : new_params['lambda1'],
                           'lambda2' : new_params['lambda2'],
                           'lambda3' : new_params['lambda3'],
                           'use_n_step' : new_params['use_n_step'],
                           'n_step_lookahead' : new_params['n_step_lookahead']
                          }
            
            final_dict_of_params = {**final_dict_of_params, **dict_to_add}
            
            self.algo_params = final_dict_of_params
            self.pretrain_params = tmp_pretrain_params
            tmp_new_params = self.get_params()
            
            if(tmp_new_params is not None):
                self.algo_params_upon_instantiation = copy.deepcopy(tmp_new_params)
            else:
                self.logger.error(msg='There was an error getting the parameters!')
                return False

            return True
        else:
            self.logger.error(msg='Cannot set parameters: \'new_params\' is \'None\'!')
            return False          
    
    def initialize_agent(self):
        self.algo_object.approximator.set_weights(self.pretrain_block.q_approximator.get_weights())
        self.algo_object._update_target() 
        
        
class ModelGenerationDDPGfD(ModelGenerationFromDemonstration):
    """
    This class implements a model generation algorithm: DDPGfD. The class initializes the policy
    of the algorithm through behavioral cloning.
    
    This class inherits from the class ModelGenerationFromDemonstration.
    """
    
    def __init__(self, eval_metric, obj_name, regressor_type='generic_regressor', seeder=2, algo_params=None, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process', deterministic_output_policy=True):
        """        
        Parameters
        ----------
        algo_params: This is either None or a dictionary containing all the needed parameters.
                            
                     The default is None.        
                                                 
                     If None then the following parameters will be used:
                     'input_shape': self.info_MDP.observation_space.shape,
                     'n_actions': None, 
                     'output_shape': self.info_MDP.action_space.shape,
                     'policy': OrnsteinUhlenbeckPolicy(sigma=0.2*np.ones(1), theta=0.15, dt=1e-2)
                     'actor_network': one hidden layer, 16 neurons, 
                     'actor_class': Adam,
                     'actor_lr': 1e-3,
                     'critic_network': one hidden layer, 16 neurons, 
                     'critic_class': Adam,
                     'critic_lr': 1e-3,
                     'loss': F.mse_loss,
                     'batch_size': 100,
                     'initial_replay_size': 50000,
                     'max_replay_size': 1000000,
                     'tau': 0.005,     
                     'policy_delay': 1,
                     'n_epochs': 10,
                     'n_steps': None,
                     'n_steps_per_fit': None,
                     'n_episodes': 500,
                     'n_episodes_per_fit': 50
                     
        regressor_type: This is a string and it can either be: 'action_regressor',  'q_regressor' or 'generic_regressor'. This is
                        used to pick one of the 3 possible kind of regressor made available by MushroomRL.
                        
                        Note that if you want to use a 'q_regressor' then the picked regressor must be able to perform 
                        multi-target regression, as a single regressor is used for all actions. 
                    
                        The default is 'generic_regressor'.
        
        deterministic_output_policy: If this is True then the output policy will be rendered deterministic else if False nothing
                                     will be done. Note that the policy is made deterministic only at the end of the learn()
                                     method.
                        
        Non-Parameters Members
        ----------------------
        fully_instantiated: This is True if the block is fully instantiated, False otherwise. It is mainly used to make sure that 
                            when we call the learn method the model generation blocks have been fully instantiated as they 
                            undergo two stage initialisation being info_MDP unknown at the beginning of the pipeline.
                            
        info_MDP: This is a dictionary compliant with the parameters needed in input to all MushroomRL model generation 
                  algorithms. It containts the observation space, the action space, the MDP horizon and the MDP gamma.
        
        
        algo_object: This is the object containing the actual model generation algorithm.
                     
        algo_params_upon_instantiation: This a copy of the original value of algo_params, namely the value of
                                        algo_params that the object got upon creation. This is needed for re-loading
                                        objects.
                                        
        model: This is used in set_params in the generic Class ModelGenerationMushroomOnline. With this member we avoid 
               re-writing for each Class inheriting from the Class ModelGenerationFromDemonstration the set_params method. 
               In this Class this member equals to DDPG, which is the Class of MushroomRL implementing DDPG.
        
        core: This is used to contain the Core object of MushroomRL needed to run online RL algorithms.
    
        The other parameters and non-parameters members are described in the Class Block.
        """
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
        self.works_on_online_rl = True
        self.works_on_offline_rl = False
        self.works_on_box_action_space = True
        self.works_on_discrete_action_space = False
        self.works_on_box_observation_space = True
        self.works_on_discrete_observation_space = True
        
        self.regressor_type = regressor_type
        
        #this block has parameters and I may want to tune them:
        self.is_parametrised = True
        
        self.algo_params = algo_params
        
        self.deterministic_output_policy = deterministic_output_policy
    
        self.fully_instantiated = False               
        self.info_MDP = None 
        self.algo_object = None
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)
    
        self.model = DDPG
        
        self.core = None
        
        #seeds torch
        torch.manual_seed(self.seeder)
        torch.cuda.manual_seed(self.seeder)
        
        #this seeding is needed for the policy of MushroomRL. Indeed the evaluation at the start of the learn method is done 
        #using the policy and in the method draw_action, np.random is called! 
        np.random.seed(self.seeder)
    
    def _default_network(self):  
        """
        This method creates a default CriticNetwork with 1 hidden layer and ReLU activation functions and a default ActorNetwork 
        with 1 hidden layer and ReLU activation functions.
        
        Returns
        -------
        CriticNetwork, ActorNetwork: the Class wrappers representing the default CriticNetwork and ActorNetwork.
        """
        
        class CriticNetwork(nn.Module):
            def __init__(self, input_shape, output_shape, **kwargs):
                super().__init__()
        
                n_input = input_shape[-1]
                n_output = output_shape[0]
                
                self.hl0 = nn.Linear(n_input, 16)
                self.hl1 = nn.Linear(16, 16)
                self.hl2 = nn.Linear(16, n_output)
                
                nn.init.xavier_uniform_(self.hl0.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
        
            def forward(self, state, action, **kwargs):
                state_action = torch.cat((state.float(), action.float()), dim=1)
                h = F.relu(self.hl0(state_action))
                h = F.relu(self.hl1(h))
                q = self.hl2(h)
        
                return torch.squeeze(q)
        
        class ActorNetwork(nn.Module):
            def __init__(self, input_shape, output_shape,  **kwargs):
                super(ActorNetwork, self).__init__()
        
                n_input = input_shape[-1]
                n_output = output_shape[0]
        
                self.hl0 = nn.Linear(n_input, 16)
                self.hl1 = nn.Linear(16, 16)
                self.hl2 = nn.Linear(16, n_output)
                
                nn.init.xavier_uniform_(self.hl0.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
             
            def forward(self, state, **kwargs):
                h = F.relu(self.hl0(torch.squeeze(state, 1).float()))
                h = F.relu(self.hl1(h))
            
                return self.hl2(h)
            
        return CriticNetwork, ActorNetwork
                
    def full_block_instantiation(self, info_MDP):   
        """
        Parameters
        ----------
        info_MDP: This is an object of Class mushroom_rl.environment.MDPInfo. It contains the action and observation spaces, 
                  gamma and the horizon of the MDP.
        
        Returns
        -------
        This method returns True if the algo_params were set successfully, and False otherwise.
        """
        
        self.info_MDP = info_MDP
                
        if(self.algo_params is None):
            policy_class = Categorical(hp_name='policy_class', obj_name='policy_class_'+str(self.model.__name__), 
                                       current_actual_value=OrnsteinUhlenbeckPolicy)
            
            sigma = Real(hp_name='sigma', current_actual_value=0.2, obj_name='sigma_'+str(self.model.__name__))
            
            theta = Real(hp_name='theta', current_actual_value=0.15, obj_name='theta_'+str(self.model.__name__))
            
            dt =  Real(hp_name='dt', current_actual_value=1e-2, obj_name='dt_'+str(self.model.__name__))
        
            critic, actor = self._default_network()
            
            #actor:
            actor_network = Categorical(hp_name='actor_network', obj_name='actor_network_'+str(self.model.__name__), 
                                        current_actual_value=actor)
             
            actor_class = Categorical(hp_name='actor_class', obj_name='actor_class_'+str(self.model.__name__), 
                                      current_actual_value=optim.Adam) 
    
            actor_lr = Real(hp_name='actor_lr', obj_name='actor_lr_'+str(self.model.__name__), 
                            current_actual_value=1e-3, range_of_values=[1e-5, 1e-3], to_mutate=True, seeder=self.seeder, 
                            log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            #critic:
            critic_network = Categorical(hp_name='critic_network', obj_name='critic_network_'+str(self.model.__name__), 
                                         current_actual_value=critic)
            
            critic_class = Categorical(hp_name='critic_class', obj_name='critic_class_'+str(self.model.__name__), 
                                       current_actual_value=optim.Adam) 
            
            critic_lr = Real(hp_name='critic_lr', obj_name='critic_lr_'+str(self.model.__name__), 
                             current_actual_value=1e-3, range_of_values=[1e-5, 1e-3], to_mutate=True, seeder=self.seeder, 
                             log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            critic_loss = Categorical(hp_name='loss', obj_name='loss_'+str(self.model.__name__), 
                                      current_actual_value=F.mse_loss)
                        
            batch_size = Integer(hp_name='batch_size', obj_name='batch_size_'+str(self.model.__name__), 
                                 current_actual_value=100, range_of_values=[8, 128], to_mutate=True, seeder=self.seeder, 
                                 log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            initial_replay_size = Integer(hp_name='initial_replay_size', current_actual_value=50000,
                                          range_of_values=[1000, 10000], to_mutate=True, seeder=self.seeder, 
                                          log_mode=self.log_mode, obj_name='initial_replay_size_'+str(self.model.__name__))
            
            max_replay_size = Integer(hp_name='max_replay_size', current_actual_value=1000000, range_of_values=[10000, 1000000],
                                      to_mutate=True, seeder=self.seeder, log_mode=self.log_mode,
                                      obj_name='max_replay_size_'+str(self.model.__name__))
            
            tau = Real(hp_name='tau', current_actual_value=0.005, obj_name='tau_'+str(self.model.__name__))
            
            policy_delay = Integer(hp_name='policy_delay', current_actual_value=1, 
                                   obj_name='policy_delay_'+str(self.model.__name__))
            
            n_epochs = Integer(hp_name='n_epochs', current_actual_value=10, range_of_values=[1,50], to_mutate=True, 
                               obj_name='n_epochs_'+str(self.model.__name__), seeder=self.seeder, log_mode=self.log_mode, 
                               checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            
            n_steps = Integer(hp_name='n_steps', current_actual_value=None, obj_name='n_steps_'+str(self.model.__name__), 
                              seeder=self.seeder, log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, 
                              verbosity=self.verbosity)
            
            n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=None, to_mutate=False, 
                                      obj_name='n_steps_per_fit_'+str(self.model.__name__), seeder=self.seeder, 
                                      log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, 
                                      verbosity=self.verbosity)
            
            n_episodes = Integer(hp_name='n_episodes', current_actual_value=500, range_of_values=[10,1000], to_mutate=True, 
                                 obj_name='n_episodes_'+str(self.model.__name__), seeder=self.seeder, 
                                 log_mode=self.log_mode, checkpoint_log_path=self.checkpoint_log_path, 
                                 verbosity=self.verbosity)
            
            n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=50, range_of_values=[1,1000], 
                                         to_mutate=True, obj_name='n_episodes_per_fit_'+str(self.model.__name__), 
                                         seeder=self.seeder, log_mode=self.log_mode, 
                                         checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
                
            dict_of_params = {'policy_class': policy_class,
                              'sigma': sigma,
                              'theta': theta,
                              'dt': dt,
                              'actor_network': actor_network, 
                              'actor_class': actor_class, 
                              'actor_lr': actor_lr,
                              'critic_network': critic_network, 
                              'critic_class': critic_class, 
                              'critic_lr': critic_lr,           
                              'loss': critic_loss,
                              'batch_size': batch_size,
                              'initial_replay_size': initial_replay_size,
                              'max_replay_size': max_replay_size,
                              'tau': tau,
                              'policy_delay': policy_delay,
                              'n_epochs': n_epochs,
                              'n_steps': n_steps,
                              'n_steps_per_fit': n_steps_per_fit,
                              'n_episodes': n_episodes,
                              'n_episodes_per_fit': n_episodes_per_fit
                             }
            
            self.algo_params = dict_of_params
        
        is_set_param_success = self.set_params(new_params=self.algo_params)
        
        if(not is_set_param_success):
             err_msg = 'There was an error setting the parameters of a'+'\''+str(self.__class__.__name__)+'\' object!'
             self.logger.error(msg=err_msg)
             self.fully_instantiated = False
             self.is_learn_successful = False
             return False
        
        self.logger.info(msg='\''+str(self.__class__.__name__)+'\' object fully instantiated!')
        self.fully_instantiated = True
        return True 
    
    def set_params(self, new_params):
       """
       Parameters
       ----------
       new_params: The new parameters to be used in the specific model generation algorithm. It must be a dictionary that does 
                   not contain any dictionaries(i.e: all parameters must be at the same level).
                                       
                   We need to create the dictionary in the right form for MushroomRL. Then it needs to update self.algo_params. 
                   Then it needs to update the object self.algo_object: to this we need to pass the actual values and not 
                   the Hyperparameter objects. 
                   
                   We call _select_current_actual_value_from_hp_classes: to this method we need to pass the dictionary already in 
                   its final form. 
       Returns
       -------
       bool: This method returns True if new_params is set correctly, and False otherwise.
       """
          
       if(new_params is not None):                
           mdp_info = Categorical(hp_name='mdp_info', obj_name='mdp_info_'+str(self.model.__name__), 
                                  current_actual_value=self.info_MDP)
           
           input_shape = Categorical(hp_name='input_shape', obj_name='input_shape_'+str(self.model.__name__), 
                                     current_actual_value=self.info_MDP.observation_space.shape)
           
           if(self.regressor_type == 'action_regressor'):
               output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                          current_actual_value=(1,))
               n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                        current_actual_value=self.info_MDP.action_space.n)
           elif(self.regressor_type == 'q_regressor'):
               output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                          current_actual_value=(self.info_MDP.action_space.n,))
               n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                        current_actual_value=self.info_MDP.action_space.n)
           elif(self.regressor_type == 'generic_regressor'):
               output_shape = Categorical(hp_name='output_shape', obj_name='output_shape_'+str(self.model.__name__), 
                                          current_actual_value=self.info_MDP.action_space.shape)
               #to have a generic regressor I must not specify n_actions
               n_actions =  Categorical(hp_name='n_actions', obj_name='n_actions_'+str(self.model.__name__), 
                                        current_actual_value=None)
               
           """tmp_structured_algo_params, dict_to_add = self.model_specific_set_params(new_params=new_params, mdp_info=mdp_info, 
                                                                                    input_shape=input_shape, 
                                                                                    output_shape=output_shape,
                                                                                    n_actions=n_actions"""
           
           critic_input_shape = Categorical(hp_name='critic_input_shape', obj_name='critic_input_shape_'+str(self.model.__name__), 
                                         current_actual_value=(input_shape.current_actual_value[0]+
                                                               self.info_MDP.action_space.shape[0],))
                   
           critic_output_shape = Categorical(hp_name='critic_output_shape', current_actual_value=(1,),
                                          obj_name='critic_output_shape_'+str(self.model.__name__))
        
           tmp_structured_algo_params = {'mdp_info': mdp_info,
                                      'actor_params': {'input_shape': input_shape,
                                                       'n_actions': n_actions,
                                                       'output_shape': output_shape
                                                      },
                                      'actor_optimizer': {'class': None, 'params': {'lr': None}}, 
                                      'critic_params': {'input_shape': critic_input_shape,
                                                        'output_shape': critic_output_shape,
                                                        'optimizer': {'class': None, 'params': {'lr': None}} 
                                                       }
                                     }
        
           #either np.ones(1) or np.ones(self.info_MDP.action_space.shape[0])        
           new_sigma = np.ones(1)*new_params['sigma'].current_actual_value
        
           policy_params_dict = dict(sigma=new_sigma, theta=new_params['theta'].current_actual_value, 
                                  dt=new_params['dt'].current_actual_value)
        
           policy_params = Categorical(hp_name='policy_params', current_actual_value=policy_params_dict, 
                                    obj_name='policy_params_'+str(self.model.__name__))
        
           new_params.update({'policy_params': policy_params}) 
        
           for tmp_key in list(new_params.keys()):
            #i do not want to change mdp_info                             
               if(tmp_key in ['policy_class', 'policy_params', 'batch_size', 'initial_replay_size', 'max_replay_size', 'tau', 
                           'policy_delay']):
                   tmp_structured_algo_params.update({tmp_key: new_params[tmp_key]})
                                                  
               if(tmp_key == 'loss'):
                   tmp_structured_algo_params['critic_params'].update({tmp_key: new_params[tmp_key]})
             
               if(tmp_key == 'critic_network'):
                   tmp_structured_algo_params['critic_params'].update({'network': new_params[tmp_key]})
        
               if(tmp_key == 'critic_class'):
                   tmp_structured_algo_params['critic_params']['optimizer'].update({'class': new_params[tmp_key]})  
                
               if(tmp_key == 'critic_lr'):
                   tmp_structured_algo_params['critic_params']['optimizer']['params'].update({'lr': new_params[tmp_key]})                                                                                       
                
               if(tmp_key == 'actor_network'):
                   tmp_structured_algo_params['actor_params'].update({'network': new_params[tmp_key]}) 
            
               if(tmp_key == 'actor_class'):
                   tmp_structured_algo_params['actor_optimizer'].update({'class': new_params[tmp_key]})  
                
               if(tmp_key == 'actor_lr'):
                   tmp_structured_algo_params['actor_optimizer']['params'].update({'lr': new_params[tmp_key]})                                                                   
                   
           structured_dict_of_values = self._select_current_actual_value_from_hp_classes(params_structured_dict=
                                                                                      tmp_structured_algo_params)
                    
           #i need to un-pack structured_dict_of_values for DDPG  
           self.algo_object = DDPG(**structured_dict_of_values)
        
           #now that i have created the DDPG object i can resolve the conflict between the 'actor_class', 'actor_lr', 
           #'actor_network', 'critic_class', 'critic_lr' and 'critic_network'. To resolve it, i need to change their keys from 
           #generic 'class', 'lr' and 'network', that are needed for MushroomRL, to 'actor_class', 'actor_lr', 'actor_network', 
           #'critic_class', critic_lr' and 'critic_network':
           tmp_structured_algo_params['critic_params']['critic_network'] = tmp_structured_algo_params['critic_params']['network'] 
           del tmp_structured_algo_params['critic_params']['network']
        
           new_val = tmp_structured_algo_params['critic_params']['optimizer']['class'] 
           tmp_structured_algo_params['critic_params']['optimizer']['critic_class'] = new_val
           del tmp_structured_algo_params['critic_params']['optimizer']['class'] 
            
           new_val = tmp_structured_algo_params['critic_params']['optimizer']['params']['lr'] 
           tmp_structured_algo_params['critic_params']['optimizer']['params']['critic_lr'] = new_val
           del tmp_structured_algo_params['critic_params']['optimizer']['params']['lr'] 
           
           new_val = tmp_structured_algo_params['actor_params']['network'] 
           tmp_structured_algo_params['actor_params']['actor_network'] = new_val
           del tmp_structured_algo_params['actor_params']['network']
                   
           tmp_structured_algo_params['actor_optimizer']['actor_class'] = tmp_structured_algo_params['actor_optimizer']['class'] 
           del tmp_structured_algo_params['actor_optimizer']['class'] 
            
           new_val = tmp_structured_algo_params['actor_optimizer']['params']['lr'] 
           tmp_structured_algo_params['actor_optimizer']['params']['actor_lr'] = new_val
           del tmp_structured_algo_params['actor_optimizer']['params']['lr'] 
            
           #delete policy_params: this is constructed new each time here:
           del tmp_structured_algo_params['policy_params']
                        
           #add n_epochs, n_steps, n_steps_per_fit, n_episodes, n_episodes_per_fit, sigma, theta, dt:
           dict_to_add = {'n_epochs': new_params['n_epochs'],
                           'n_steps': new_params['n_steps'],
                           'n_steps_per_fit': new_params['n_steps_per_fit'],
                           'n_episodes': new_params['n_episodes'],
                           'n_episodes_per_fit': new_params['n_episodes_per_fit'],
                           'sigma': new_params['sigma'], 
                           'theta': new_params['theta'], 
                           'dt': new_params['dt']
                         }
           
           final_dict_of_params = {**tmp_structured_algo_params, **dict_to_add}
           
           self.algo_params = final_dict_of_params

           tmp_new_params = self.get_params()
           
           if(tmp_new_params is not None):
               self.algo_params_upon_instantiation = copy.deepcopy(tmp_new_params)
           else:
               self.logger.error(msg='There was an error getting the parameters!')
               return False

           return True
       else:
           self.logger.error(msg='Cannot set parameters: \'new_params\' is \'None\'!')
           return False           
    
    def _default_network(self):  
        """
        This method creates a default CriticNetwork with 1 hidden layer and ReLU activation functions and a default ActorNetwork 
        with 1 hidden layer and ReLU activation functions.
        
        Returns
        -------
        CriticNetwork, ActorNetwork: the Class wrappers representing the default CriticNetwork and ActorNetwork.
        """
        
        class CriticNetwork(nn.Module):
            def __init__(self, input_shape, output_shape, **kwargs):
                super().__init__()
        
                n_input = input_shape[-1]
                n_output = output_shape[0]
                
                self.hl0 = nn.Linear(n_input, 16)
                self.hl1 = nn.Linear(16, 16)
                self.hl2 = nn.Linear(16, n_output)
                
                nn.init.xavier_uniform_(self.hl0.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
        
            def forward(self, state, action, **kwargs):
                state_action = torch.cat((state.float(), action.float()), dim=1)
                h = F.relu(self.hl0(state_action))
                h = F.relu(self.hl1(h))
                q = self.hl2(h)
        
                return torch.squeeze(q)
        
        class ActorNetwork(nn.Module):
            def __init__(self, input_shape, output_shape,  **kwargs):
                super(ActorNetwork, self).__init__()
        
                n_input = input_shape[-1]
                n_output = output_shape[0]
        
                self.hl0 = nn.Linear(n_input, 16)
                self.hl1 = nn.Linear(16, 16)
                self.hl2 = nn.Linear(16, n_output)
                
                nn.init.xavier_uniform_(self.hl0.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.hl2.weight, gain=nn.init.calculate_gain('relu'))
             
            def forward(self, state, **kwargs):
                h = F.relu(self.hl0(torch.squeeze(state, 1).float()))
                h = F.relu(self.hl1(h))
            
                return self.hl2(h)
            
        return CriticNetwork, ActorNetwork


    def initialize_agent(self, algo_object, demos, n_steps):
        states, actions, _, _, _, _ = parse_dataset(demos.dataset)
        for n in n_steps:
            algo_object._actor_approximator.fit(states, actions)
        algo_object._init_target(algo_object._target_actor_approximator, algo_object._actor_approximator)       