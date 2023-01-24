from copy import deepcopy

import numpy as np

from mushroom_rl.algorithms.value.dqn.dqn import DQN
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory, ReplayMemory
from mushroom_rl.utils.parameters import to_parameter

class DQNFromDemonstration(DQN):

    def __init__(self, mdp_info, policy, approximator, approximator_params,
                 batch_size, demo_batch_size, target_update_frequency,
                 margin=0.5, lambda2=1, replay_memory=None, initial_replay_size=500,
                 max_replay_size=5000, fit_params=None, predict_params=None, clip_reward=False):
        """
        Constructor.

        Args:
            demo_batch_size ([int, Parameter]): the number of samples in a batch of demonstrations;
            demo_replay_memory([ReplayMemory]): buffer of expert demonstrations;
            margin ([float, Parameter]): supervised margin to make the value of the demonstrated action higher than the others
        """
        self._demo_batch_size = to_parameter(demo_batch_size)
        self._demo_replay_memory = None
        self._margin = margin
        self._lambda2 = lambda2

        super().__init__(mdp_info, policy, approximator, approximator_params,
                 batch_size, target_update_frequency,
                 replay_memory, initial_replay_size,
                 max_replay_size, fit_params, predict_params, clip_reward)
        self._fit = self._fit_demo

    def _fit_demo(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())
            d_state, d_action, d_reward, d_next_state, d_absorbing, _ = \
                self._replay_memory.get(self._demo_batch_size())
            if self._clip_reward:
                reward = np.clip(reward, -1, 1)
                d_reward = np.clip(reward, -1, 1)
            state = np.vstack([state,d_state])
            action = np.vstack([action,d_action])
            reward = np.concatenate([reward,d_reward])
            next_state =np.vstack([next_state,d_next_state])
            absorbing = np.concatenate([absorbing,d_absorbing])
            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self.approximator.fit(state, action, q, **self._fit_params)
            #Supervised fit step from demonstrations
            q_pred = self.approximator.predict(d_state, d_action, output_tensor=True)
            q_target = self._margin_q(d_state, d_action, self._margin)
            supervised_margin_loss = torch.mean(abs(q_target - q_pred))
            q_loss = self._lambda2*supervised_margin_loss
            # train Q function
            self.approximator.model._optimizer.zero_grad()
            q_loss.backward()
            self.approximator.model._optimizer.step()
    
    def _margin_q(self, states, actions, margin):
        q = self.approximator.predict(states, output_tensor=True)
        l = torch.ones_like(q) * margin
        l.scatter_(1, torch.LongTensor(actions.reshape(-1, 1)), torch.zeros_like(q))
        return torch.max(q+l, dim=1).values

    def set_demo_replay_memory(self, demo_replay_memory):
        self._demo_replay_memory = demo_replay_memory