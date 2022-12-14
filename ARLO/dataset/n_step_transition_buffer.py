from collections import deque
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            transition = (curr_state, action, reward, next_state, done, done)
            demos_n_step.append(transition)

    return demos_1_step, demos_n_step


def get_n_step_info(n_step_buffer, gamma):
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, _, done = n_step_buffer[-1][-4:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, _, d = transition[-4:]

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done

class NStepTransitionBuffer(object):
    """Fixed-size buffer to store experience tuples.
    Attributes:
        buffer (list): list of replay buffer
        buffer_size (int): buffer size not storing demos
        demo_size (int): size of a demo to permanently store in the buffer
        cursor (int): position to store next transition coming in
    """

    def __init__(self, buffer_size=5000, n_step=10, gamma=0.99, demo=None):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): size of replay buffer for experience
            demo (list): demonstration transitions
        """
        assert buffer_size > 0

        self.n_step_buffer = deque(maxlen=n_step)
        self.buffer_size = buffer_size
        self.buffer = list()
        self.n_step = n_step
        self.gamma = gamma
        self.demo_size = 0
        self.cursor = 0

        # if demo exists
        if demo:
            self.demo_size = len(demo)
            self.buffer.extend(demo)

        # self.buffer.extend([None] * self.buffer_size)

    def add(self, transition):
        """Add a new transition to memory."""
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]
        new_transition = (curr_state, action, reward, next_state, done)

        # insert the new transition to buffer
        idx = self.demo_size + self.cursor
        self.buffer[idx] = new_transition
        self.cursor = (self.cursor + 1) % self.buffer_size

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        states, actions, rewards, next_states, absorbing, dones = [], [], [], [], [], []

        for tuple in self.buffer:
            if(tuple is not None):
                s, a, r, n_s, ab, d = (tuple)
                states.append(np.array(s, copy=False))
                actions.append(np.array(a, copy=False))
                rewards.append(np.array(r, copy=False))
                next_states.append(np.array(n_s, copy=False))
                absorbing.append(np.array(ab, copy=False))
                dones.append(np.array(float(d), copy=False))

        states_ = torch.FloatTensor(np.array(states)).to(device)
        actions_ = torch.FloatTensor(np.array(actions)).to(device)
        rewards_ = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states_ = torch.FloatTensor(np.array(next_states)).to(device)
        absorbing_ = torch.FloatTensor(np.array(absorbing).reshape(-1, 1)).to(device)
        dones_ = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        if torch.cuda.is_available():
            states_ = states_.cuda(non_blocking=True)
            actions_ = actions_.cuda(non_blocking=True)
            rewards_ = rewards_.cuda(non_blocking=True)
            next_states_ = next_states_.cuda(non_blocking=True)
            absorbing_ = absorbing_.cuda(non_blocking=True)
            dones_ = dones_.cuda(non_blocking=True)

        return states_, actions_, rewards_, next_states_, absorbing_, dones_


    def parse_data(self, features=None):
        assert len(self.buffer) > 0

        shape = self.buffer[0][0].shape if features is None else (features.size,)

        state = np.ones((len(self.buffer),) + shape)
        action = np.ones((len(self.buffer),) + self.buffer[0][1].shape)
        reward = np.ones(len(self.buffer))
        next_state = np.ones((len(self.buffer),) + shape)
        absorbing = np.ones(len(self.buffer))
        last = np.ones(len(self.buffer))

        if features is not None:
            for i in range(len(self.buffer)):
                if(self.buffer[i] is not None):
                    state[i, ...] = features(self.buffer[i][0])
                    action[i, ...] = self.buffer[i][1]
                    reward[i] = self.buffer[i][2]
                    next_state[i, ...] = features(self.buffer[i][3])
                    absorbing[i] = self.buffer[i][4]
                    last[i] = self.buffer[i][5]
        else:
            for i in range(len(self.buffer)):
                if(self.buffer[i] is not None):
                    state[i, ...] = self.buffer[i][0]
                    action[i, ...] = self.buffer[i][1]
                    reward[i] = self.buffer[i][2]
                    next_state[i, ...] = self.buffer[i][3]
                    absorbing[i] = self.buffer[i][4]
                    last[i] = self.buffer[i][5]

        return np.array(state), np.array(action), np.array(reward), np.array(
            next_state), np.array(absorbing), np.array(last)