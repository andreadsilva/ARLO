from ARLO.block import ModelGenerationMushroomOnlineDQNfD
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


def parse_file(f):
    
    states=list()
    actions=list()
    next_states=list()
    rewards=list()
    absorbing=list()
    dones = list()
    line = f.readline()
    while line:
      
      tupla = line.replace('\n', '').split(',')
      s = np.fromstring(tupla[0].replace('[', '').replace(']', ''), sep=' ', dtype=float)

      
      a=int(tupla[1].replace('[', '').replace(']', ''))
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


def get_n_step_info_from_demo(demo, n_step, gamma):
    """Return 1 step and n step demos."""
    assert demo
    assert n_step > 1

    demos_1_step = list()
    demos_n_step = list()
    n_step_buffer = deque(maxlen=n_step)

    for transition in demo:
        n_step_buffer.append(transition)

        if len(n_step_buffer) == n_step:
            # add a single step transition
            demos_1_step.append(n_step_buffer[0])

            # add a multi step transition
            curr_state, action = n_step_buffer[0][:2]
            reward, next_state, done = get_n_step_info(n_step_buffer, gamma)
            transition = (curr_state, action, reward, next_state, done)
            demos_n_step.append(transition)

    return demos_1_step, demos_n_step

def get_n_step_info(n_step_buffer, gamma):
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done


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

    my_env = BaseCarOnHill(obj_name='my_car', gamma=.99, horizon=1000)
    my_env.seed(2)
    # Extracting expert demonstrations
    f = open('/home/andrea/ARLO/experiences.txt', 'r')
    s, a, r, ns, ab, d = parse_file(f)
    dataset_expert=list(zip(s, a, r, ns, ab, d))
    if True:
        train_data = TabularDataSet(dataset=dataset_expert, observation_space=my_env._mdp_info.observation_space, action_space=my_env._mdp_info.action_space, discrete_actions=True, discrete_observations=False,
                                    gamma=my_env._mdp_info.gamma, horizon=my_env._mdp_info.horizon, obj_name='expert demonstrations')
        
        approximator = Categorical(hp_name='approximator', obj_name='approximator_'+'DQfD', 
                                        current_actual_value=TorchApproximator)
                
        network = Categorical(hp_name='network', obj_name='network_'+'DQfD', 
                                current_actual_value=_default_network())
        
        optimizer_class = Categorical(hp_name='class', obj_name='optimizer_class_'+'DQfD', 
                                        current_actual_value=torch.optim.Adam) 
        
        lr = Real(hp_name='lr', obj_name='optimizer_lr_'+'DQfD', 
                    current_actual_value=1e-3, seeder=2, 
                    log_mode='console', verbosity=3)
                    
        critic_loss = Categorical(hp_name='critic_loss', obj_name='critic_loss_'+'DQfD', 
                                    current_actual_value=F.smooth_l1_loss)
        
        batch_size = Integer(hp_name='batch_size', obj_name='batch_size_'+'DQfD', 
                                current_actual_value=32, seeder=2, 
                                log_mode='console', verbosity=3)
        
        demo_batch_size = Integer(hp_name='batch_size', obj_name='demo_batch_size_'+'DQfD', 
                                current_actual_value=8, seeder=2, 
                                log_mode='console', verbosity=3)

        target_update_frequency = Integer(hp_name='target_update_frequency', current_actual_value=250, 
                                            range_of_values=[100,1000], to_mutate=True, 
                                            obj_name='target_update_frequency_'+'DQfD', seeder=2, 
                                            log_mode='console', 
                                            verbosity=3)
        
        initial_replay_size = Integer(hp_name='initial_replay_size', current_actual_value=2000, 
                                        range_of_values=[1000, 10000], 
                                        obj_name='initial_replay_size_'+'DQfD',
                                        seeder=2, log_mode='console', 
                                        verbosity=3)
        
        max_replay_size = Integer(hp_name='max_replay_size', current_actual_value=10000,
                                    obj_name='max_replay_size_'+'DQfD', seeder=2, 
                                    log_mode='console', 
                                    verbosity=3)
        
        replay_memory = Categorical(hp_name='replay_memory', obj_name='replay_memory_'+'DQfD', 
                                    current_actual_value=ReplayMemory(initial_size=initial_replay_size.current_actual_value, 
                                                                        max_size=max_replay_size.current_actual_value))
            
        clip_reward = Categorical(hp_name='clip_reward', obj_name='clip_reward_'+'DQfD', 
                                    current_actual_value=False,
                                    seeder=2, log_mode='console', 
                                    verbosity=3)
        
        n_epochs = Integer(hp_name='n_epochs', current_actual_value=0, 
                            obj_name='n_epochs_'+'DQfD', seeder=2, log_mode='console', 
                            verbosity=3)
        
        n_steps = Integer(hp_name='n_steps', current_actual_value=None,
                            obj_name='n_steps_'+'DQfD', seeder=2, log_mode='console', 
                            verbosity=3)
        
        n_steps_per_fit = Integer(hp_name='n_steps_per_fit', current_actual_value=None, 
                                     obj_name='n_steps_per_fit_'+'DQfD', seeder=2,
                                    log_mode='console', 
                                    verbosity=3)
        
        n_episodes = Integer(hp_name='n_episodes', current_actual_value=500,
                                obj_name='n_episodes_'+'DQfD', seeder=2, log_mode='console', 
                                verbosity=3)
        
        n_episodes_per_fit = Integer(hp_name='n_episodes_per_fit', current_actual_value=10, obj_name='n_episodes_per_fit_'+'DQfD', 
                                        seeder=2, log_mode='console', 
                                        verbosity=3)

        epsilon = Categorical(hp_name='epsilon', obj_name='epsilon_'+'DQfD', 
                                current_actual_value=LinearParameter(value=1, threshold_value=0.01, n=1000000))

        n_epochs_pretraining = Integer(hp_name='n_epochs_pretraining', current_actual_value=400, 
                            obj_name='n_epochs_'+'DQfD', seeder=2, log_mode='console', 
                            verbosity=3)
        
        lambda1 = Real(hp_name='lambda1', obj_name='lambda1_'+'DQfD', 
                    current_actual_value=1, seeder=2, 
                    log_mode='console',  verbosity=3)

        lambda2 = Real(hp_name='lambda2', obj_name='lambda2_'+'DQfD', 
                    current_actual_value=1, seeder=2, 
                    log_mode='console', verbosity=3)

        lambda3 = Real(hp_name='lambda3', obj_name='lambda3_'+'DQfD', 
                    current_actual_value=1e-4,  seeder=2, 
                    log_mode='console', verbosity=3)

        use_n_step = Categorical(hp_name='use_n_step', obj_name='critic_loss_'+'DQfD', 
                                    current_actual_value=False)

        n_step_lookahead = Integer(hp_name='n_steps', current_actual_value=10, 
                            obj_name='n_epochs_'+'DQfD', seeder=2, log_mode='console', 
                             verbosity=3)

        margin = Real(hp_name='margin', current_actual_value=0.6,
                              obj_name='margin'+'DQfD', seeder=2, log_mode='console', 
                              checkpoint_log_path='/logg.txt', verbosity=3)
        
        dict_of_params = {'approximator': approximator, 
                            'network': network, 
                            'class': optimizer_class, 
                            'lr': lr, 
                            'loss': critic_loss, 
                            'batch_size': batch_size,
                            'demo_batch_size': demo_batch_size, 
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
                            'n_step_lookahead': n_step_lookahead,
                            'margin': margin
                            }

    
    
    my_dqfd = ModelGenerationMushroomOnlineDQNfD(eval_metric=DiscountedReward(obj_name='dqfd_metric', n_episodes=10, batch=True,
                                            log_mode=my_log_mode, checkpoint_log_path=dir_chkpath), 
                                            obj_name='my_dqfd', deterministic_output_policy=True, log_mode=my_log_mode, algo_params=dict_of_params,
                                            checkpoint_log_path=dir_chkpath, n_jobs=4, seeder=2)

    my_pipeline = OnlineRLPipeline(list_of_block_objects=[my_dqfd],
                                   eval_metric=DiscountedReward(obj_name='pipeline_metric', n_episodes=10, batch=True), 
                                   obj_name='OnlinePipeline') 

    out = my_pipeline.learn(env=my_env, train_data=train_data)
    
    #learnt policy:
    my_policy = out.policy.policy

    done = False
    ns = my_env.reset()
    while not done:
        action = my_policy.draw_action(ns)
        ns, r, done = my_env.step(action=action)
        my_env.render()
    evals = my_dqfd.dict_of_evals

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
    plt.savefig('./DQNfD/res.png') 