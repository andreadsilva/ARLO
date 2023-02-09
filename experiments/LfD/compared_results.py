import copy

import numpy as np

np.random.seed(2)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tikzplotlib as tikz 

from ARLO.environment import LQG
from ARLO.block import ModelGenerationFromDemonstrationSAC, ModelGenerationMushroomOnlineSAC
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.tuner import TunerGenetic
from ARLO.block import AutoModelGeneration
from ARLO.input_loader import LoadSameTrainDataAndEnv
from ARLO.metric import DiscountedReward
from ARLO.hyperparameter import Categorical, Real, Integer
from ARLO.dataset import TabularDataSet


def parse_file(f):
    
    states=list()
    actions=list()
    next_states=list()
    rewards=list()
    absorbing=list()
    dones = list()
    lines = f.read().split(';')
    lines[-1] = None
    for line in lines:
      if not line:
        break
      tupla = line.replace('\n', '').split(',')
      s = np.fromstring(tupla[0].replace('[', '').replace(']', ''), sep=' ', dtype=float)
      a = np.fromstring(tupla[1].replace('[', '').replace(']', ''), sep=' ', dtype=float)  
      
      #a=float(tupla[1].replace('[', '').replace(']', ''))
      r=float(tupla[2])
      ns=np.fromstring(tupla[3].replace('[', '').replace(']', ''), sep=' ')
      ab= 1 if (tupla[4].replace(' ', '')=='True') else 0
      done= 1 if (tupla[5].replace(';', '').replace(' ', '')=='True') else 0
      
      states.append(s)
      actions.append(a)
      rewards.append(r)
      next_states.append(ns)
      absorbing.append(ab)
      dones.append(done)
      line = f.readline()

    f.close()
    # return states, actions, rewards, next_states, absorbing, dones
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(absorbing), np.array(dones)



if __name__ == '__main__':
    dir_chkpath = './' 
    
    A = np.array([[1,0],[0,1]])
    B = np.array([[1,0,0],[0,0,1]])
    Q = 0.7*np.array([[1,0],[0,1]])
    R = 0.3*np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    my_lqg = LQG(obj_name='lqg', A=A, B=B, Q=Q, R=R, max_pos=3.5, max_action=3.5, env_noise=0.1*np.eye(2), 
                 controller_noise=0*1e-4*np.eye(3), seeder=2, horizon=15, gamma=0.9)
    my_log_mode = 'console'
    current_seed = 2

    
    # Extracting expert demonstrations
    f = open('/home/andrea/ARLO/experiments/LfD/demo_LQG.txt', 'r')
    s, a, r, ns, ab, d = parse_file(f)
    dataset_expert=list(zip(s, a, r, ns, ab, d))
    train_data = TabularDataSet(dataset=dataset_expert[:200], observation_space=my_lqg.observation_space, action_space=my_lqg.action_space, discrete_actions=True, discrete_observations=False,
                                        gamma=my_lqg.gamma, horizon=my_lqg.horizon, obj_name='expert demonstrations')

    
    class CriticNetwork(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
    
            n_input = input_shape[0]
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
        
            n_input = input_shape[0]
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

    if(True):
        #actor:
        actor_network_mu = Categorical(hp_name='actor_network_mu', obj_name='actor_network_mu_sac', 
                                    current_actual_value=ActorNetwork)
        
        actor_network_sigma = Categorical(hp_name='actor_network_sigma', obj_name='actor_network_sigma_sac', 
                                        current_actual_value=copy.deepcopy(ActorNetwork))
        
        actor_class = Categorical(hp_name='actor_class', obj_name='actor_class_sac', 
                                current_actual_value=optim.Adam) 
        
        actor_lr = Categorical(hp_name='actor_lr', obj_name='actor_lr_sac', current_actual_value=1e-2)
        
        #critic:
        critic_network = Categorical(hp_name='critic_network', obj_name='critic_network_sac', current_actual_value=CriticNetwork)
        
        critic_class = Categorical(hp_name='critic_class', obj_name='critic_class_sac', current_actual_value=optim.Adam) 
        
        critic_lr = Categorical(hp_name='critic_lr', obj_name='critic_lr_sac', current_actual_value=1e-2)
        
        critic_loss = Categorical(hp_name='loss', obj_name='loss_sac', current_actual_value=F.mse_loss)
                    
        batch_size = Categorical(hp_name='batch_size', obj_name='batch_size_sac', 
                                current_actual_value=64)

        initial_replay_size = Categorical(hp_name='initial_replay_size', current_actual_value=100, obj_name='initial_replay_size_sac')
        
        max_replay_size = Categorical(hp_name='max_replay_size', current_actual_value=10000, obj_name='max_replay_size_sac')
        
        warmup_transitions = Categorical(hp_name='warmup_transitions', current_actual_value=100, obj_name='warmup_transitions_sac')
        
        tau = Categorical(hp_name='tau', current_actual_value=0.005, obj_name='tau_sac')
        
        lr_alpha = Categorical(hp_name='lr_alpha', current_actual_value=1e-3, obj_name='lr_alpha_sac')
        
        log_std_min = Real(hp_name='log_std_min', current_actual_value=-20, obj_name='log_std_min_sac')
        
        log_std_max = Real(hp_name='log_std_max', current_actual_value=3, obj_name='log_std_max_sac')
        
        target_entropy = Real(hp_name='target_entropy', current_actual_value=None, obj_name='target_entropy_sac')
        
        n_epochs = Integer(hp_name='n_epochs', current_actual_value=15, obj_name='n_epochs')

        n_steps = Integer(hp_name='n_steps', current_actual_value=None,  obj_name='n_steps')
        
        n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=None, obj_name='n_steps_per_fit')
        
        n_episodes = Integer(hp_name='n_episodes', current_actual_value=500, obj_name='n_episodes')

        n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=10, obj_name='n_episodes_per_fit')

        dict_of_params_sac = {'actor_network_mu': actor_network_mu, 
                            'actor_network_sigma': actor_network_sigma,
                            'actor_class': actor_class, 
                            'actor_lr': actor_lr,
                            'critic_network': critic_network, 
                            'critic_class': critic_class, 
                            'critic_lr': critic_lr,           
                            'loss': critic_loss,
                            'batch_size': batch_size,
                            'initial_replay_size': initial_replay_size,
                            'max_replay_size': max_replay_size,
                            'warmup_transitions': warmup_transitions,
                            'tau': tau,
                            'lr_alpha': lr_alpha,
                            'log_std_min': log_std_min,
                            'log_std_max': log_std_max,
                            'target_entropy': target_entropy,
                            'n_epochs': n_epochs,
                            'n_steps': n_steps,
                            'n_steps_per_fit': n_steps_per_fit,
                            'n_episodes': n_episodes,
                            'n_episodes_per_fit': n_episodes_per_fit
                            } 
        
    my_sac = ModelGenerationMushroomOnlineSAC(eval_metric=DiscountedReward(obj_name='sac_metric', n_episodes=100, batch=False,
                                                                           log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                              obj_name='my_sac', regressor_type='generic_regressor', log_mode=my_log_mode, 
                                              checkpoint_log_path=dir_chkpath, n_jobs=16, seeder=current_seed, 
                                              algo_params=dict_of_params_sac,
                                              deterministic_output_policy=False)
                                                                                         
    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_sac], 
                                   eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=100, batch=False,
                                                                log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                   obj_name='OnlinePipeline', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath) 
    
    out = my_pipeline.learn(train_data=train_data, env=my_lqg)

    evals_sac = my_sac.dict_of_evals


    demo_batch_size = Categorical(hp_name='demo_batch_size', obj_name='demo_batch_size_sac', current_actual_value=16)
        
    n_epochs_pretraining = Integer(hp_name='n_epochs_pretraining', current_actual_value=500, obj_name='n_epochs')
    
    lambda1 = Real(hp_name='lambda1', obj_name='lambda1', current_actual_value=8)

    lambda1_pt = Real(hp_name='lambda1_pt', obj_name='lambda1_pt', current_actual_value=8)

    pretrain_critic = Categorical(hp_name='pretrain_critic', obj_name='pretrain_critic', current_actual_value=False) 

    l2_actor = Real(hp_name='l2_actor', current_actual_value=1e-4, obj_name='l2_actor')
    
    l2_critic = Real(hp_name='l2_critic', current_actual_value=1e-4, obj_name='l2_critic')

    dict_of_params_sac.update({'demo_batch_size': demo_batch_size, 'n_epochs_pretraining': n_epochs_pretraining,
                     'lambda1': lambda1, 'lambda1_pt':lambda1_pt,'pretrain_critic': pretrain_critic,'l2_actor': l2_actor, 'l2_critic': l2_critic})

    my_sac = ModelGenerationFromDemonstrationSAC(eval_metric=DiscountedReward(obj_name='sac_metric', n_episodes=100, batch=False,
                                                                            log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                                obj_name='my_sac', regressor_type='generic_regressor', log_mode=my_log_mode, 
                                                checkpoint_log_path=dir_chkpath, n_jobs=16, seeder=current_seed, 
                                                algo_params=dict_of_params_sac,
                                                deterministic_output_policy=False)

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_sac], 
                                    eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=100, batch=False,
                                                                log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                    obj_name='OnlinePipeline', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath) 

    out = my_pipeline.learn(env=my_lqg, train_data=train_data)

    #learnt policy:
    # my_policy = out.policy.policy

    evals_sacfd_pretrainBC = my_sac.dict_of_evals


    pretrain_critic = Categorical(hp_name='pretrain_critic', obj_name='pretrain_critic', current_actual_value=True) 
    dict_of_params_sac.update({'pretrain_critic': pretrain_critic})

    my_sac = ModelGenerationFromDemonstrationSAC(eval_metric=DiscountedReward(obj_name='sac_metric', n_episodes=100, batch=False,
                                                                            log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                                obj_name='my_sac', regressor_type='generic_regressor', log_mode=my_log_mode, 
                                                checkpoint_log_path=dir_chkpath, n_jobs=16, seeder=current_seed, 
                                                algo_params=dict_of_params_sac,
                                                deterministic_output_policy=False)

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_sac], 
                                    eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=100, batch=False,
                                                                log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                    obj_name='OnlinePipeline', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath) 

    out = my_pipeline.learn(env=my_lqg, train_data=train_data)

    #learnt policy:
    # my_policy = out.policy.policy

    evals_sacfd_pretrainAC = my_sac.dict_of_evals


    plt.figure()
    plt.xlabel('Environment Steps')
    plt.ylabel('Average Discounted Reward')
    plt.title('Compared Average Discounted Reward and Standard Deviation')
    plt.grid(True)
    x = np.array(list(evals_sac.keys()))
    evals_values = list(evals_sac.values())
    y = np.array([np.mean(evals_values[i]) for i in range(len(evals_values))])
    std_dev = np.array([np.std(evals_values[i]) for i in range(len(evals_values))])/10
    plt.plot(x, y, color='#FF9860', label='SAC')
    plt.fill_between(x, y-std_dev, y+std_dev, alpha=0.5, facecolor='#FF9860')
    x = np.array(list(evals_sacfd_pretrainAC.keys()))
    evals_values = list(evals_sacfd_pretrainAC.values())
    y = np.array([np.mean(evals_values[i]) for i in range(len(evals_values))])
    std_dev = np.array([np.std(evals_values[i]) for i in range(len(evals_values))])/10
    plt.plot(x, y, color='#9860FF', label='SAC pretraining AC')
    plt.fill_between(x, y-std_dev, y+std_dev, alpha=0.5, facecolor='#9860FF')
    x = np.array(list(evals_sacfd_pretrainBC.keys()))
    evals_values = list(evals_sacfd_pretrainBC.values())
    y = np.array([np.mean(evals_values[i]) for i in range(len(evals_values))])
    std_dev = np.array([np.std(evals_values[i]) for i in range(len(evals_values))])/10
    plt.plot(x, y, color='#98FF60', label='SAC pretraining BC')
    plt.fill_between(x, y-std_dev, y+std_dev, alpha=0.5, facecolor='#98FF60')
    # if(len(evals_values[0]) > 1):
    #     plt.fill_between(x, y-std_dev, y+std_dev, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9860')
    plt.legend()
    tikz.save('/home/andrea/real_comparison.tex')  
