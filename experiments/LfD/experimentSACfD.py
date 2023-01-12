from ARLO.block import ModelGenerationMushroomOnlineDQN, ModelGenerationMushroomOnlineDQNfD, ModelGenerationFromDemonstrationSAC
from ARLO.rl_pipeline import OnlineRLPipeline
from ARLO.metric import DiscountedReward, SomeSpecificMetric
from ARLO.environment import BaseCarOnHill, BaseCartPole, BaseInvertedPendulum
from ARLO.hyperparameter import Integer, Categorical, Real
from ARLO.dataset import TabularDataSet
from ARLO.input_loader import LoadSameTrainDataAndEnv, LoadUniformSubSampleWithReplacementAndEnv, LoadSameEnv
from ARLO.tuner import TunerGenetic
from ARLO.block import AutoModelGeneration
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

      
      a=float(tupla[1].replace('[', '').replace(']', ''))
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

#NETS
n_features = 32

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 32)
        self._h2 = nn.Linear(32, n_features)
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
    my_log_mode = 'console'
    current_seed = 2
    dir_chkpath = './Logg.txt'
    n_agents = 3

    my_env = BaseInvertedPendulum(obj_name='my_env')#DMControl('walker', 'stand', horizon, gamma)

    my_env.seed(2)
    # Extracting expert demonstrations
    f = open('/home/andrea/ARLO/demo_ddpg.txt', 'r')
    s, a, r, ns, ab, d = parse_file(f)
    dataset_expert=list(zip(s, a, r, ns, ab, d))
    train_data = TabularDataSet(dataset=dataset_expert, observation_space=my_env._mdp_info.observation_space, action_space=my_env._mdp_info.action_space, discrete_actions=True, discrete_observations=False,
                                        gamma=my_env._mdp_info.gamma, horizon=my_env._mdp_info.horizon, obj_name='expert demonstrations')

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
        
        actor_lr = Categorical(hp_name='actor_lr', obj_name='actor_lr_sac', current_actual_value=1e-2, 
                            possible_values=[1e-5, 1e-4, 1e-3, 1e-2], to_mutate=True)
        
        #critic:
        critic_network = Categorical(hp_name='critic_network', obj_name='critic_network_sac', current_actual_value=CriticNetwork)
        
        critic_class = Categorical(hp_name='critic_class', obj_name='critic_class_sac', current_actual_value=torch.optim.Adam) 
        
        critic_lr = Categorical(hp_name='critic_lr', obj_name='critic_lr_sac', current_actual_value=1e-2, 
                                possible_values=[1e-5, 1e-4, 1e-3, 1e-2], to_mutate=True)
        
        critic_loss = Categorical(hp_name='loss', obj_name='loss_sac', current_actual_value=F.mse_loss)
                    
        batch_size = Categorical(hp_name='batch_size', obj_name='batch_size_sac', 
                                current_actual_value=64, possible_values=[8, 16, 32, 64, 128], to_mutate=True)
        
        demo_batch_size = Categorical(hp_name='demo_batch_size', obj_name='demo_batch_size_sac', 
                                current_actual_value=64, possible_values=[8, 16, 32, 64, 128], to_mutate=True)

        initial_replay_size = Categorical(hp_name='initial_replay_size', current_actual_value=100, 
                                        possible_values=[10, 100, 300, 500, 1000, 5000],
                                        to_mutate=True, obj_name='initial_replay_size_sac')
        
        max_replay_size = Categorical(hp_name='max_replay_size', current_actual_value=10000, possible_values=[3000, 10000, 30000, 100000],
                                    to_mutate=True, obj_name='max_replay_size_sac')
        
        warmup_transitions = Categorical(hp_name='warmup_transitions', current_actual_value=100, possible_values=[50, 100, 500], 
                                        to_mutate=True, obj_name='warmup_transitions_sac')
        
        tau = Categorical(hp_name='tau', current_actual_value=0.005, obj_name='tau_sac')
        
        lr_alpha = Categorical(hp_name='lr_alpha', current_actual_value=1e-3, obj_name='lr_alpha_sac', possible_values=[1e-5, 1e-4, 1e-3], 
                            to_mutate=True)
        
        log_std_min = Real(hp_name='log_std_min', current_actual_value=-20, obj_name='log_std_min_sac')
        
        log_std_max = Real(hp_name='log_std_max', current_actual_value=3, obj_name='log_std_max_sac')
        
        target_entropy = Real(hp_name='target_entropy', current_actual_value=None, obj_name='target_entropy_sac')
        
        n_epochs = Integer(hp_name='n_epochs', current_actual_value=2, range_of_values=[1,30], to_mutate=True, obj_name='n_epochs')
        
        n_steps = Integer(hp_name='n_steps', current_actual_value=50,  obj_name='n_steps')
        
        n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=5, obj_name='n_steps_per_fit')
        
        n_episodes = Integer(hp_name='n_episodes', current_actual_value=None, range_of_values=[1, 1600], to_mutate=True, 
                            obj_name='n_episodes')
        
        n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=None, range_of_values=[1, 500], to_mutate=True, 
                                    obj_name='n_episodes_per_fit')
        n_epochs_pretraining = Integer(hp_name='n_epochs_pretraining', current_actual_value=2000, range_of_values=[500,10000], to_mutate=True, 
                        obj_name='n_epochs')

        lambda1 = Real(hp_name='lambda1', obj_name='lambda1', 
                        current_actual_value=1, range_of_values=[0.1, 1.5], to_mutate=True)    
                
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
                            'lambda1': lambda1
                            }

    my_sac = ModelGenerationFromDemonstrationSAC(eval_metric=DiscountedReward(obj_name='sac_metric', n_episodes=1, batch=False,
                                                                            log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                                obj_name='my_sac', regressor_type='generic_regressor', log_mode=my_log_mode, 
                                                checkpoint_log_path=dir_chkpath, n_jobs=16, seeder=current_seed, 
                                                algo_params=dict_of_params_sac,
                                                deterministic_output_policy=False)
                                                                                            
    tuner_dict = dict(block_to_opt=my_sac, n_agents=20, n_generations=10, n_jobs=16, job_type='thread', seeder=current_seed,
                        eval_metric=DiscountedReward(obj_name='discounted_rew_genetic_algo', n_episodes=1, batch=False, 
                                                    n_jobs=1, job_type='process', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath),
                        input_loader=LoadSameTrainDataAndEnv(obj_name='input_loader_env'), obj_name='genetic_algo', prob_point_mutation=0.5, 
                        output_save_periodicity=1, log_mode=my_log_mode, checkpoint_log_path=dir_chkpath, tuning_mode='no_elitism')

    tuner = TunerGenetic(**tuner_dict)

    auto_model_gen = AutoModelGeneration(eval_metric=DiscountedReward(obj_name='discounted_rew_auto_model_gen', n_episodes=1,
                                                                        batch=False, n_jobs=1, job_type='process', 
                                                                        log_mode=my_log_mode, checkpoint_log_path=dir_chkpath),
                                            obj_name='auto_model_gen', tuner_blocks_dict={'genetic_tuner': tuner}, 
                                            log_mode=my_log_mode, checkpoint_log_path=dir_chkpath)

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[auto_model_gen], 
                                    eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=1, batch=False,
                                                                log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                    obj_name='OnlinePipeline', log_mode=my_log_mode, checkpoint_log_path=dir_chkpath) 

    out = my_pipeline.learn(env=my_env, train_data=train_data)

    #learnt policy:
    my_policy = out.policy.policy

    done = False
    ns = my_env.reset()
    steps=0
    discount = 1
    gamma = my_env._mdp_info.gamma
    my_env.render('human')
    episode = ['The Phantom Menace', 'Attack of the Clones', 'The Revenge of the Sith', 'A New Hope', 'The Empire Strikes Back']
    for i in range(5):
        print('\n', episode[i])
        while not done:
            ns , rew,  done, info = my_env.step(np.argmax(my_policy.draw_action(ns)))
            steps +=1
            discount *=gamma
        done=False
        ns= my_env.reset()
        print('\nTotal steps: ', steps)
        steps=0