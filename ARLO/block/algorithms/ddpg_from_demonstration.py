from copy import deepcopy

import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ddpg import DDPG
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.utils.parameters import to_parameter
import torch.nn.functional as F

class DDPGFromDemonstration(DDPG):

    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params, batch_size,
                 demo_batch_size, initial_replay_size, max_replay_size, tau, policy_delay=1,
                 lambda1=1, critic_fit_params=None, actor_predict_params=None, critic_predict_params=None):
        """
        Constructor.

        Args:
            demo_batch_size ([int, Parameter]): the number of samples in a batch of demonstrations;
            demo_replay_memory([ReplayMemory]): buffer of expert demonstrations;
            lambda1([int, Parameter]): weight of the behavioral cloning loss

        """
        self._demo_batch_size = demo_batch_size
        self._demo_replay_memory = None
        self._lambda1 = lambda1

        super().__init__(mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, policy_delay,
                 critic_fit_params, actor_predict_params, critic_predict_params)
        
    def fit(self, dataset):
        
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size())
            d_state, d_action, d_reward, d_next_state, d_absorbing, _ = \
                self._demo_replay_memory.get(self._demo_batch_size)
            state = np.vstack([state,d_state])
            action = np.vstack([action,d_action])
            reward = np.concatenate([reward,d_reward])
            next_state =np.vstack([next_state,d_next_state])
            absorbing = np.concatenate([absorbing,d_absorbing])
            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            if self._fit_count % self._policy_delay() == 0:
                loss = self._loss(state)
                self._optimize_actor_parameters(loss)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)
            self._update_target(self._actor_approximator,
                                self._target_actor_approximator)

            self._fit_count += 1

            # self._actor_approximator.fit(state, action)
            loss = self._bc_loss(d_state, d_action)
            self._optimize_actor_parameters(self._lambda1*loss)

    def _bc_loss(self, state, action):
        pred_action = self._actor_approximator(state, output_tensor=True, **self._actor_predict_params)
        # pred_action = pred_action.reshape(-1, 1)
        return torch.mean(pred_action.reshape(-1, self.mdp_info.action_space.shape[0])- torch.FloatTensor(action.reshape(-1, self.mdp_info.action_space.shape[0])).pow(2))

    def set_demo_replay_memory(self, demo_replay_memory):
        self._demo_replay_memory = demo_replay_memory