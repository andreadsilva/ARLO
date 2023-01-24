from copy import deepcopy

import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic.sac import SAC
from mushroom_rl.approximators.parametric.torch_approximator import *
import torch.nn.functional as F
class SACFromDemonstration(SAC):

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params,
                 actor_optimizer, critic_params, batch_size, demo_batch_size,
                 lambda1, initial_replay_size, max_replay_size,
                 warmup_transitions, tau, lr_alpha, log_std_min=-20, log_std_max=2,
                 target_entropy=None, critic_fit_params=None):
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

        super().__init__(mdp_info, actor_mu_params, actor_sigma_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, log_std_min, log_std_max, target_entropy,
                 critic_fit_params)
        
    def fit(self, dataset):
        
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())
            d_state, d_action, d_reward, d_next_state, d_absorbing, _ = \
                self._replay_memory.get(self._demo_batch_size)
            state = np.vstack([state,d_state])
            action = np.vstack([action,d_action])
            reward = np.concatenate([reward,d_reward])
            next_state =np.vstack([next_state,d_next_state])
            absorbing = np.concatenate([absorbing,d_absorbing])

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

            # self.policy._mu_approximator.fit(state, action)
            loss = self._bc_loss(d_state, d_action)
            self._optimize_actor_parameters(self._lambda1*loss)

    def _bc_loss(self, state, action):
        pred_action, _ = self.policy.compute_action_and_log_prob_t(state)
        return torch.mean((pred_action.reshape(-1, self.mdp_info.action_space.shape[0])- torch.FloatTensor(action.reshape(-1,self.mdp_info.action_space.shape[0]))).pow(2))

    def set_demo_replay_memory(self, demo_replay_memory):
        self._demo_replay_memory = demo_replay_memory