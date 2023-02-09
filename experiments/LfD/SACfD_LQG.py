
import matplotlib.pyplot as plt
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.metric import DiscountedReward
from ARLO.environment import BaseCarOnHill, LQG
from ARLO.hyperparameter import Integer, Categorical, Real
from ARLO.dataset import TabularDataSet
from ARLO.input_loader import LoadSameTrainDataAndEnv
from ARLO.tuner import TunerGenetic
from ARLO.block import ModelGenerationFromDemonstrationSAC
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import LinearParameter
from mushroom_rl.utils.replay_memory import ReplayMemory
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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
      a=np.fromstring(tupla[1].replace('[', '').replace(']', ''), sep=' ', dtype=float)
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

def gen_transitions(my_env,my_policy, n_demo):
    dataset = list()
    done = False
    s = my_env.reset()
    steps=0
    while steps<n_demo:
        while not done:
            a = my_policy.draw_action(s)
            ns , rew,  done, info = my_env.step(a)
            steps +=1
            transition = (s, a, rew, ns, done, done)
            dataset.append(transition)
            s=ns
        done=False
        ns= my_env.reset()
    return dataset


#NETS
n_features = 16

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

if __name__ == '__main__':
    my_log_mode = 'file'
    current_seed = 2
    dir_chkpath = './'
    n_agents = 3

    A = np.array([[1,0],[0,1]])
    B = np.array([[1,0,0],[0,0,1]])
    Q = 0.7*np.array([[1,0],[0,1]])
    R = 0.3*np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    my_lqg = LQG(obj_name='lqg', A=A, B=B, Q=Q, R=R, max_pos=3.5, max_action=3.5, env_noise=0.1*np.eye(2), 
                 controller_noise=0*1e-4*np.eye(3), seeder=2, horizon=15, gamma=0.9)
    my_log_mode = 'console'
    current_seed = 2

    my_lqg.seed(2)
    # Extracting expert demonstrations
    f = open('/home/andrea/ARLO/experiments/LfD/demo_LQG.txt', 'r')
    s, a, r, ns, ab, d = parse_file(f)
    dataset_expert=list(zip(s, a, r, ns, ab, d))
    train_data = TabularDataSet(dataset=dataset_expert, observation_space=my_lqg.observation_space, action_space=my_lqg.action_space, discrete_actions=True, discrete_observations=False,
                                        gamma=my_lqg.gamma, horizon=my_lqg.horizon, obj_name='expert demonstrations')


    if True:    
        #actor:
        actor_network_mu = Categorical(hp_name='actor_network_mu', obj_name='actor_network_mu_sac', 
                                    current_actual_value=ActorNetwork)

        optimizer_class = Categorical(hp_name='class', obj_name='optimizer_class_', 
                                            current_actual_value=torch.optim.Adam) 
        
        actor_network_sigma = Categorical(hp_name='actor_network_sigma', obj_name='actor_network_sigma_sac', 
                                        current_actual_value=copy.deepcopy(ActorNetwork))
        
        actor_class = Categorical(hp_name='actor_class', obj_name='actor_class_sac', 
                                current_actual_value=torch.optim.Adam) 
        
        actor_lr = Categorical(hp_name='actor_lr', obj_name='actor_lr_sac', current_actual_value=1e-2)
        
        #critic:
        critic_network = Categorical(hp_name='critic_network', obj_name='critic_network_sac', current_actual_value=CriticNetwork)
        
        critic_class = Categorical(hp_name='critic_class', obj_name='critic_class_sac', current_actual_value=torch.optim.Adam) 
        
        critic_lr = Categorical(hp_name='critic_lr', obj_name='critic_lr_sac', current_actual_value=1e-2)
        
        critic_loss = Categorical(hp_name='loss', obj_name='loss_sac', current_actual_value=F.mse_loss)
                    
        batch_size = Categorical(hp_name='batch_size', obj_name='batch_size_sac', 
                                current_actual_value=32)
        
        demo_batch_size = Categorical(hp_name='demo_batch_size', obj_name='demo_batch_size_sac', 
                                current_actual_value=4)

        initial_replay_size = Categorical(hp_name='initial_replay_size', current_actual_value=400, obj_name='initial_replay_size_sac')
        
        max_replay_size = Categorical(hp_name='max_replay_size', current_actual_value=10000, obj_name='max_replay_size_sac')
        
        warmup_transitions = Categorical(hp_name='warmup_transitions', current_actual_value=100, obj_name='warmup_transitions_sac')
        
        tau = Categorical(hp_name='tau', current_actual_value=0.005, obj_name='tau_sac')
        
        lr_alpha = Categorical(hp_name='lr_alpha', current_actual_value=1e-3, obj_name='lr_alpha_sac')
        
        log_std_min = Real(hp_name='log_std_min', current_actual_value=-20, obj_name='log_std_min_sac')
        
        log_std_max = Real(hp_name='log_std_max', current_actual_value=2, obj_name='log_std_max_sac')
        
        target_entropy = Real(hp_name='target_entropy', current_actual_value=None, obj_name='target_entropy_sac')
        
        n_epochs = Integer(hp_name='n_epochs', current_actual_value=0, obj_name='n_epochs')
        
        n_steps = Integer(hp_name='n_steps', current_actual_value=None,  obj_name='n_steps')
        
        n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=None, obj_name='n_steps_per_fit')
        
        n_episodes = Integer(hp_name='n_episodes', current_actual_value=500, obj_name='n_episodes')
        
        n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=10, obj_name='n_episodes_per_fit')

        n_epochs_pretraining = Integer(hp_name='n_epochs_pretraining', current_actual_value=500, obj_name='n_epochs_pretraining')

        lambda1 = Real(hp_name='lambda1', obj_name='lambda1', current_actual_value=10)
        
        lambda1_pt = Real(hp_name='lambda1_pt', obj_name='lambda1_pt', current_actual_value=10)

        pretrain_critic = Categorical(hp_name='pretrain_critic', current_actual_value=True,
                                        obj_name='pretrain_critic_')   
        l2_actor = Real(hp_name='l2_actor', current_actual_value=1e-3, to_mutate=False, obj_name='l2_actor_ddpg')

        l2_critic = Real(hp_name='l2_critic', current_actual_value=1e-3, obj_name='l2_actor_ddpg')

        dict_of_params_sac = {'optimizer_class':optimizer_class,
                            'actor_network_mu': actor_network_mu, 
                            'actor_network_sigma': actor_network_sigma,
                            'actor_class': actor_class, 
                            'actor_lr': actor_lr,
                            'critic_network': critic_network, 
                            'critic_class': critic_class, 
                            'critic_lr': critic_lr,           
                            'loss': critic_loss,
                            'batch_size': batch_size,
                            'demo_batch_size': demo_batch_size,
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
                            'n_episodes_per_fit': n_episodes_per_fit,
                            'n_epochs_pretraining': n_epochs_pretraining,
                            'lambda1': lambda1,
                            'lambda1_pt': lambda1_pt,
                            'pretrain_critic': pretrain_critic,
                            'l2_actor': l2_actor,
                            'l2_critic': l2_critic
                            }

    my_sac = ModelGenerationFromDemonstrationSAC(eval_metric=DiscountedReward(obj_name='sac_metric', n_episodes=100, batch=False,
                                                                            log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                                obj_name='my_sac', regressor_type='generic_regressor', log_mode=my_log_mode, 
                                                checkpoint_log_path=dir_chkpath, n_jobs=4, seeder=current_seed, 
                                                algo_params=dict_of_params_sac,
                                                deterministic_output_policy=False)

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_sac], 
                                    eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=100, batch=False,
                                                                log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                    obj_name='OnlinePipeline', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath) 

    out = my_pipeline.learn(env=my_lqg, train_data=train_data)

    #learnt policy:
    my_policy = out.policy.policy

    my_policy = out.policy.policy

    dataset = gen_transitions(my_env=my_lqg, my_policy=my_policy, n_demo=3000)

    f = open('demo_LQG_1.txt', 'w')
    for i in dataset:
        f.write(''+('%s, %s, %s, %s, %s, %s' % (i[0], i[1], i[2],i[3], i[4],i[5]))+';\n')
    f.close()


    evals = my_sac.dict_of_evals


    evals = my_sac.dict_of_evals

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
    plt.title('Average Discounted Reward and Standard Deviation for SACfD')
    plt.grid(True)
    plt.plot(x, y, color='#FF9860')
    if(len(evals_values[0]) > 1):
        plt.fill_between(x, y-std_dev, y+std_dev, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9860')
    # plt.savefig('final/200_pt_l3_warmup.png') 