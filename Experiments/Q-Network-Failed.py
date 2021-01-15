# Failed Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import random
import numpy as np
import gym

from collections import namedtuple, deque


class Q(nn.Module):
    def __init__(self, env_space, action_space):
        super(Q, self).__init__()
        self.env_space = env_space
        self.action_space = action_space

        self.fc = nn.Sequential(
            nn.Linear(self.env_space, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, self.action_space),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

# Memory


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# Agent


class Agent:
    def __init__(self, env):
        self.gamma = 0.99
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.eps = self.eps_start
        self.episodes = 0
        self.batchSize = 32
        self.tau = 1e-4
        self.lr = 0.001

        # TODO Add a bunch of data collections like reward average, loss, etc
        self.losses = []
        self.averageLoss = []

        self.rewards = []
        self.averageReward = []

        self.env = env
        self.actionSpace = self.env.action_space.n
        self.observationSpace = self.env.observation_space.shape[0]
        self.memory = ReplayBuffer(self.actionSpace, 100000, self.batchSize, 0)

        # Agent
        self.model = Q(self.observationSpace, self.actionSpace)
        self.localModel = Q(self.observationSpace, self.actionSpace)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

    def updateData(self, recentLoss, recentReward):
        self.losses.append(recentLoss)
        self.rewards.append(recentReward)

        self.averageLoss.append(sum(self.losses)/len(self.losses))
        self.averageReward.append(sum(self.rewards)/len(self.rewards))

    @torch.no_grad
    def initState(self, state):
        state = state.astype(np.float32)
        state = torch.from_numpy(state)
        return state

    def Run(self, episodes, render=True):
        for episode in range(0, episodes):
            self.episodes = episode

            if episode % 100 == 0:
                print(f"Episode: {episode}")
                print(f"Epilson: {self.eps}")
                print("Average Loss: ",
                      self.averageLoss[len(self.averageLoss)-1])
                print("Average Reward: ",
                      self.averageReward[len(self.averageReward)-1])

            state = self.initState(env.reset())
            done = False

            while not done:
                action = self.Action(state)
                next_state, reward, done, _ = self.env.step(action)

                # TODO learning, memory, ...
                self.memory.add(state, action, next_state, reward, done)

                if len(self.memory) > self.batchSize:
                    experiences = self.memory.sample()
                    loss = self.learn(experiences)
                    self.updateData(loss, reward)

                state = self.initState(next_state)

                env.render() if render else None

        plt.plot(self.averageRewards)
        plt.plot(self.averageLoss)
        plt.show()

    @torch.no_grad
    def learn(self, experiences):
        # *('state', 'action', 'next_state', 'reward', 'done'))

        state, action, next_state, reward, done, = experiences
        nextTargets = self.model.forward(next_state).detach().max(1)[
            0].unsqueeze(1)
        targets = reward+(self.gamma*nextTargets*(1-done))

        expected = self.localModel(state).gather(1, action)

        loss = F.mse_loss(expected, targets)

        for param in self.model.parameters():
            param.grad = None

        loss.backward()
        self.optim.step()

        for target_param, local_param in zip(self.model.parameters(), self.localModel.parameters()):
            target_param.data.copy_(
                self.tau*local_param.data + (1.0-self.tau)*target_param.data)

        if self.eps > self.eps_end and self.episodes % 10 == 0:
            self.eps *= self.eps_decay

        return loss

    def Action(self, state):
        if np.random.rand() < self.eps:
            return random.randrange(self.actionSpace)

        qValues = self.model.forward(state)
        qValues = qValues.detach().numpy()
        return np.argmax(qValues[0])


env = gym.make("CartPole-v1")
agent = Agent(env)
agent.Run(5000)

env.close()
