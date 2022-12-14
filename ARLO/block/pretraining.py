"""
This module contains the implementation of the Class Pretraining, PretrainingValue, PretrainingPolicy and PretrainingAC.

The Class Pretraining inherits from the class Block.

The Classes PretrainingValue, PretrainingPolicy and PretrainingAC inherit from the Class Pretraining.

"""


from abc import abstractmethod

import copy
import numpy as np

from mushroom_rl.policy import EpsGreedy
from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from ARLO.block.block import Block, BlockOutput
from ARLO.dataset.n_step_transition_buffer import NStepTransitionBuffer
from ARLO.hyperparameter.hyperparameter import Real, Integer, Categorical
from ARLO.dataset.n_step_transition_buffer import get_n_step_info_from_demo
import torch
from torch.optim import Adam


class Pretraining(Block):
    
        def learn(self, train_data=None, env=None):
    
            res = super().learn(train_data=train_data, env=env)
            if(isinstance(res, BlockOutput)):
                return BlockOutput(obj_name=self.obj_name)
            self.demo_buffer = list()
            n_epochs_pretraining = self.algo_params['n_epochs_pretraining']
            if(self.use_n_step):
                demos, demo_n = get_n_step_info_from_demo(gamma=self.info_MDP.gamma, n_step=self.n_step_lookahead,demo=train_data.dataset)
                self.n_step_buffer = NStepTransitionBuffer(gamma=self.info_MDP.gamma, demo=demo_n)
                self.demo_buffer.extend(demos)
            else:
                self.demo_buffer.extend(train_data.dataset)
            self.pretrain(n_epochs_pretraining.current_actual_value)
            self.is_learn_successful = True 
        
        @abstractmethod
        def update_model(self, dataset):
            raise NotImplementedError

        def pretrain(self, n_epochs):
        
            print("[INFO] Pre-Train %d steps." % n_epochs)
            # indexes = np.random.random_integers(0, self.n_step_buffer.buffer_size, self.batch_size)
            dataset = self.demo_buffer
            for i in range(n_epochs):
                loss = self.update_model(dataset)
                if(i%500 == 0):
                    print('Pretrain epoch: '+str(i))
                    print('Current loss: ', str(loss))
        
        def analyse(self):
            """
            This method is not yet implemented.
            """    
            raise NotImplementedError

        def _walk_dict_to_select_current_actual_value(self, dict_of_hyperparams):
            """
            Parameters
            ----------
            dict_of_hyperparams: This is a dictionary containing objects of a Class inheriting from the Class HyperParameter. This 
                                can be a structured dictionary.
            
            Returns
            -------
            dict_of_hyperparams: This is a dictionary containing the corresponding current_actual_value of the objects of a Class 
                                inheriting from the Class HyperParameter, that were in the original dictionary.
            """
            
            for key in dict_of_hyperparams:
                if isinstance(dict_of_hyperparams[key], dict):
                    self._walk_dict_to_select_current_actual_value(dict_of_hyperparams=dict_of_hyperparams[key])
                else:
                    dict_of_hyperparams.update({key: dict_of_hyperparams[key].current_actual_value})
            
            return dict_of_hyperparams
                    
        def _select_current_actual_value_from_hp_classes(self, params_structured_dict):
            """
            Parameters
            ----------
            params_structured_dict: This is a dictionary containing objects of a Class inheriting from the Class HyperParameter. 
                                    This can be a structured dictionary.
                
            Returns
            -------
            algo_params_values: This is a dictionary containing the corresponding current_actual_value of the objects of a Class 
                                inheriting from the Class HyperParameter, that were in the original dictionary.
            """
            
            #this method is called before creating the actual agent object. This method makes sure to pass to the actual agent object 
            #numerical values and not objects of the Class Hyperparameters
            
            #deep copy the parameters: I need the dictionary with the objects of Class Hyperparameters for the Tuner algorithms.  
            copy_params_structured_dict = copy.deepcopy(params_structured_dict)
            algo_params_values = self._walk_dict_to_select_current_actual_value(dict_of_hyperparams=copy_params_structured_dict)
            
            return algo_params_values

        def _walk_dict_to_flatten_it(self, structured_dict, dict_to_fill):   
            """
            Parameters
            ----------
            structured_dict: This is the input structured dictonary: it can contain several nested sub-dictionaries.
            
            dict_to_fill: This is the dictionary passed to every recursive call to this method.
            
            Returns
            -------
            dict_to_fill: This is the final flat dictionary: it does not contain sub-dictionaries.
            """
            
            for key in structured_dict:
                if isinstance(structured_dict[key], dict):
                    self._walk_dict_to_flatten_it(structured_dict=structured_dict[key], dict_to_fill=dict_to_fill)
                else:
                    dict_to_fill.update({key: structured_dict[key]})
            
            return dict_to_fill    
    
#class PretrainingPolicy(Pretraining):
    
class PretrainingValue(Pretraining):
    
    """
        ALGO_PARAMS:
            approximator: class of the approximator to train (either given from user or default MGfD)
            approximator_params: input/output size, n_actions, optimizer{ class, params{ lr } }, network, loss
        HP:
            n_epochs_pretraining: number of supervised training epochs
            lambda1: weight of the n_step loss
            lambda2: weight of the supervised margin loss
            lambda3: weight of the L2 regularization loss, passed directly to optimizer (weight_decay=lambda3)
            margin: supervised margin, non expert actions are forced to have a value smaller than expert by at least "margin"
            n_step
            Q_approximator
            optimizer
        OBJECTS:
            model: Regressor.Approximator.model (nn.module)
            n_step_transition_buffer
            gamma: contained in mdp_info
        """
    
    def __init__(self, eval_metric, obj_name, regressor_type='q_regressor', seeder=2, algo_params=None, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process', deterministic_output_policy=True):        
        
        super().__init__(eval_metric=eval_metric, obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                         checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
        
                
        self.q_approximator = None
        self.model = None
        self.optimizer = None
        self.n_step_buffer = None
        self.demo_buffer = None
        self.use_n_step = False
        self.pipeline_type='online'
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
        self.algo_params_upon_instantiation = copy.deepcopy(self.algo_params)
        
        self.core = None
               
        #seeds torch
        torch.manual_seed(self.seeder)
        torch.cuda.manual_seed(self.seeder)

        #this seeding is needed for the policy of MushroomRL. Indeed the evaluation at the start of the learn method is done 
        #using the policy and in the method draw_action, np.random is called! 
        np.random.seed(self.seeder)

    def full_block_instantiation(self, info_MDP):
        
        self.info_MDP = info_MDP
        params = self.algo_params
        dict_of_params = dict()
        if('n_epochs_pretraining' not in params.keys()):
            n_epochs_pretraining = Integer(hp_name='n_epochs_pretraining', current_actual_value=1000, to_mutate=False, 
                              obj_name='n_epochs_pretraining'+str(self.obj_name), seeder=self.seeder, log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            dict_of_params.update({ 'n_epochs_pretraining': n_epochs_pretraining})
        if('lambda1' not in params.keys()):
            lambda1 = Real(hp_name='lambda1', current_actual_value=0.2, to_mutate=False, 
                              obj_name='lambda1'+str(self.obj_name), seeder=self.seeder, log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            dict_of_params.update({ 'lambda1': lambda1})
        if('lambda2' not in params.keys()):
            lambda2 = Real(hp_name='lambda2', current_actual_value=0.2, to_mutate=False, 
                              obj_name='lambda2'+str(self.obj_name), seeder=self.seeder, log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            dict_of_params.update({ 'lambda2': lambda2})
        if('lambda3' not in params.keys()):
            lambda3 = Real(hp_name='lambda3', current_actual_value=0.2, to_mutate=False, 
                              obj_name='lambda3'+str(self.obj_name), seeder=self.seeder, log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            dict_of_params.update({ 'lambda3': lambda3})
        if('margin' not in params.keys()):    
            margin = Real(hp_name='margin', current_actual_value=0.2, to_mutate=False, 
                              obj_name='margin'+str(self.obj_name), seeder=self.seeder, log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            dict_of_params.update({ 'margin': margin})
        if('n_step_lookahead' not in params.keys()):
            n_step_lookahead = Integer(hp_name='n_step_lookahead', current_actual_value=10, to_mutate=False, 
                              obj_name='n_step_lookahead'+str(self.obj_name), seeder=self.seeder, log_mode=self.log_mode, 
                              checkpoint_log_path=self.checkpoint_log_path, verbosity=self.verbosity)
            dict_of_params.update({ 'n_step_lookahead': n_step_lookahead})

        self.algo_params = {**params, **dict_of_params}
        
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

            tmp_structured_algo_params = {  'mdp_info': self.info_MDP,
                                            'approximator_params': { 'optimizer': {'class': None, 'params':{'lr': None}} }
                                        }
            regressor_dict = dict()
            for tmp_key in list(new_params.keys()):
                #i do not want to change mdp_info or policy
                if(tmp_key in ['approximator', 'n_epochs_pretraining', 'lambda1', 'lambda2', 'lambda3', 'use_n_step', 'n_step_lookahead', 'margin']):
                    tmp_structured_algo_params.update({tmp_key: new_params[tmp_key]})
                    if(tmp_key=='approximator'):
                        regressor_dict.update({tmp_key: new_params[tmp_key]})
                if(tmp_key in ['network', 'loss', 'input_shape', 'output_shape', 'n_actions']):
                    tmp_structured_algo_params['approximator_params'].update({tmp_key: new_params[tmp_key]})
                    regressor_dict.update({tmp_key: new_params[tmp_key]})
                if(tmp_key in ['class']):
                    tmp_structured_algo_params['approximator_params']['optimizer'].update({tmp_key: new_params[tmp_key]})  
                    regressor_dict.update({tmp_key: new_params[tmp_key]})
                if(tmp_key in ['lr']):
                    tmp_structured_algo_params['approximator_params']['optimizer']['params'].update({tmp_key: new_params[tmp_key]})                                                                   
                    regressor_dict.update({tmp_key: new_params[tmp_key]})

            regressor_params = self._walk_dict_to_select_current_actual_value(regressor_dict)

            print(regressor_params)
            #i need to un-pack structured_dict_of_values for the regressor  
            self.q_approximator = Regressor(**regressor_params)
            self.target_approximator = Regressor(**regressor_params)
            self.model=self.q_approximator.model

            #setting all the pretrain parameters (class and lr to create regressor, lambdas as losses' weights)
            optimizer_class = tmp_structured_algo_params['approximator_params']['optimizer']['class'].current_actual_value
            lr = tmp_structured_algo_params['approximator_params']['optimizer']['params']['lr'].current_actual_value
            self.lambda1 = tmp_structured_algo_params['lambda1'].current_actual_value
            self.lambda2 = tmp_structured_algo_params['lambda2'].current_actual_value
            self.lambda3 = tmp_structured_algo_params['lambda3'].current_actual_value
            self.margin = tmp_structured_algo_params['margin'].current_actual_value
            
            if(tmp_structured_algo_params['use_n_step'].current_actual_value):
                self.use_n_step = True
                self.n_step_lookahead = tmp_structured_algo_params['n_step_lookahead'].current_actual_value
            
            self.optimizer = optimizer_class(self.q_approximator.model.network.parameters(), lr=lr, weight_decay=self.lambda3)
            
            self.algo_params = tmp_structured_algo_params
            
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
            
    def get_params(self):
        """
        Returns
        -------
        flat_dict: This is a deep copy of the parameters in the dictionary self.algo_params but they are inserted in a dictionary 
                   that is of only one level (unlike the self.algo_params which is a dictionary nested into a dictionary).
                   This is needed for the Tuner Classes.
        """
        
        #I need a copy of the dict since i need to mutate it afterwards and create a new aget with these parameters.
        #I need to do this because since I want to save the best agent then without deep copying i might mutate the parameters of 
        #the best agent. 
                
        #i need to deep copy since self.algo_params contains objects of class HyperParameter and i need a new copy of these 
        #objects:        
        flat_dict = self._walk_dict_to_flatten_it(structured_dict=copy.deepcopy(self.algo_params), dict_to_fill={})
                
        return flat_dict  


    def update_model(self, dataset):
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones, absorbing = parse_dataset(dataset=dataset)
        
        # Q function loss
        masks = 1 - dones
        gamma = self.info_MDP.gamma
        actions = actions.astype(int)
        q_pred = self.q_approximator.predict(states, actions, output_tensor=True)
        next_q = self.q_next(next_states, absorbing)
        q_target = rewards + self.info_MDP.gamma * next_q * masks
        q_loss = torch.mean((q_pred - torch.FloatTensor(q_target).detach()).pow(2))

        """Calculate n_step --> add transition in the form of (state, action, next_state_n_steps, discounted_n_step_reward, done)"""

        if self.use_n_step:
            experiences_n = self.n_step_buffer.parse_data()
            _, _, rewards, next_states, ab, dones = experiences_n
            gamma = gamma ** self.n_step_lookahead
            masks = 1 - dones

            next_q = self.q_next(next_states, ab.reshape(-1))
            q_target = rewards + gamma * next_q * masks
            q_loss_n = torch.mean((q_pred - torch.FloatTensor(q_target).detach()).pow(2))
            
            # to update loss and priorities
            q_loss = q_loss + q_loss_n * self.lambda1
        q_target = self._margin_q(states, actions, self.margin)
        supervised_margin_loss = torch.mean(abs(q_target - q_pred))
        q_loss = q_loss + self.lambda2*supervised_margin_loss
        # train Q function
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        return q_loss.data    
    
    
    def _margin_q(self, states, actions, margin):
        q = self.q_approximator.predict(states, output_tensor=True)
        l = torch.ones_like(q) * margin
        l.scatter_(1, torch.LongTensor(actions.reshape(-1, 1)), torch.zeros_like(q))
        return torch.max(q+l, dim=1).values
        
    def q_next(self, next_state, absorbing):
        q = self.q_approximator.predict(next_state)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)
    
#class PretrainingAC(Pretraining):