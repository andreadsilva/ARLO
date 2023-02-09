from ARLO.block import ModelGenerationMushroomOnlineDQN
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.metric import DiscountedReward
from ARLO.environment import BaseCarOnHill
from ARLO.hyperparameter import Integer, Categorical, Real
from ARLO.dataset import TabularDataSet
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import LinearParameter
from mushroom_rl.utils.replay_memory import ReplayMemory
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _default_network():
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


if __name__ == '__main__':
    #uses default hyper-paramters of SAC
    # data_gen = DataGenerationRandomUniformPolicy(eval_metric=SomeSpecificMetric('data_gen'), obj_name='data_gen', 
    #                                             algo_params={'n_samples': Integer(obj_name='n_samples_data_gen', 
    #                                                                             hp_name='n_samples', 
    #                                                                             current_actual_value=10800)}) 
    
    my_log_mode = 'console'
    dir_chkpath = './Logg.txt'
    n_agents = 3

    my_env = BaseCarOnHill(obj_name='my_car', gamma=1, horizon=1000)
    my_env.seed(2)

    if True:

        approximator = Categorical(hp_name='approximator', obj_name='approximator_'+'DQN', 
                                        current_actual_value=TorchApproximator)
                
        network = Categorical(hp_name='network', obj_name='network_'+'DQN', 
                                current_actual_value=_default_network())
        
        optimizer_class = Categorical(hp_name='class', obj_name='optimizer_class_'+'DQN', 
                                        current_actual_value=torch.optim.Adam) 
        
        lr = Real(hp_name='lr', obj_name='optimizer_lr_'+'DQN', 
                    current_actual_value=1e-2, seeder=2, 
                    log_mode='console', verbosity=3)
                    
        critic_loss = Categorical(hp_name='critic_loss', obj_name='critic_loss_'+'DQN', 
                                    current_actual_value=F.smooth_l1_loss)
        
        batch_size = Integer(hp_name='batch_size', obj_name='batch_size_'+'DQN', 
                                current_actual_value=32, seeder=2, 
                                log_mode='console', verbosity=3)

        target_update_frequency = Integer(hp_name='target_update_frequency', current_actual_value=250, 
                                            obj_name='target_update_frequency_'+'DQN', seeder=2, 
                                            log_mode='console', 
                                            verbosity=3)
        
        initial_replay_size = Integer(hp_name='initial_replay_size', current_actual_value=2000, 
                                        obj_name='initial_replay_size_'+'DQN',
                                        seeder=2, log_mode='console', 
                                        verbosity=3)
        
        max_replay_size = Integer(hp_name='max_replay_size', current_actual_value=50000,
                                    obj_name='max_replay_size_'+'DQN', seeder=2, 
                                    log_mode='console', 
                                    verbosity=3)
        
        replay_memory = Categorical(hp_name='replay_memory', obj_name='replay_memory_'+'DQN', 
                                    current_actual_value=ReplayMemory(initial_size=initial_replay_size.current_actual_value, 
                                                                        max_size=max_replay_size.current_actual_value))
            
        clip_reward = Categorical(hp_name='clip_reward', obj_name='clip_reward_'+'DQN', 
                                    current_actual_value=False,
                                    seeder=2, log_mode='console', 
                                    verbosity=3)
        
        n_epochs = Integer(hp_name='n_epochs', current_actual_value=20,
                            obj_name='n_epochs_'+'DQN', seeder=2, log_mode='console', 
                            verbosity=3)
        
        n_steps = Integer(hp_name='n_steps', current_actual_value=None,
                            obj_name='n_steps_'+'DQN', seeder=2, log_mode='console', 
                            verbosity=3)
        
        n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=None, 
                                     obj_name='n_steps_per_fit_'+'DQN', seeder=2,
                                    log_mode='console', 
                                    verbosity=3)
        
        n_episodes = Integer(hp_name='n_episodes', current_actual_value=500,
                                obj_name='n_episodes_'+'DQN', seeder=2, log_mode='console', 
                                verbosity=3)
        
        n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=5, obj_name='n_episodes_per_fit_'+'DQN', 
                                        seeder=2, log_mode='console', 
                                        verbosity=3)

        epsilon = Categorical(hp_name='epsilon', obj_name='epsilon_'+'DQN', 
                                current_actual_value=LinearParameter(value=1, threshold_value=0.01, n=1000000))


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
                            'epsilon': epsilon
                            }

    
    
    my_dqn = ModelGenerationMushroomOnlineDQN(eval_metric=DiscountedReward(obj_name='dqfd_metric', n_episodes=10, batch=True,
                                            log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                            obj_name='my_dqn', deterministic_output_policy=True, log_mode=my_log_mode, algo_params=dict_of_params,
                                            checkpoint_log_path=dir_chkpath, n_jobs=4, seeder=2)

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_dqn],
                                   eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=10, batch=True), 
                                   obj_name='OnlinePipeline') 

    out = my_pipeline.learn(env=my_env)
    
    #learnt policy:
    my_policy = out.policy.policy

    evals = my_dqn.dict_of_evals
    
    """
    This method plots and saves the dict_of_evals of the block.
    """
    
    x = np.array(list(evals.keys()))
    if(len(x) == 0):
        exc_msg = 'The \'dict_of_evals\' is empty!'
        print(exc_msg)
        raise ValueError(exc_msg)
        
    evals_values = list(evals.values())
    y = np.array([np.mean(evals_values[i]) for i in range(len(evals_values))])
    
    std_dev = np.array([np.std(evals_values[i]) for i in range(len(evals_values))])
    
    plt.figure()


    plt.xlabel('Environment Steps')
    plt.ylabel('Average Discounted Reward')
    plt.title('Average Discounted Reward and Standard Deviation for SAC from demonstration')
    plt.grid(True)
    plt.plot(x, y, color='#FF9860')
    if(len(evals_values[0]) > 1):
        plt.fill_between(x, y-std_dev, y+std_dev, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9860')
    plt.savefig('./DQNresult/learning.png') 