import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli, Normal
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output logits for each of the 3 possible actions in each of the 6 dimensions
        self.action_head = nn.Linear(hidden_size, action_size * 3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        # Reshape logits to (batch_size, action_size, 3)
        return action_logits.view(-1, 6, 3)

    def evaluate(self, state, epsilon=1e-6):
        action_logits = self.forward(state)
        dists = [Categorical(logits=logits) for logits in action_logits.transpose(0, 1)]
        actions = torch.stack([dist.sample() for dist in dists], dim=1)
        log_probs = torch.stack(
            [dist.log_prob(action) for dist, action in zip(dists, actions.T)], dim=1
        )
        return actions, log_probs.sum(dim=1)  # Sum log probabilities across actions

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_logits = self.forward(state)
        # Softmax to convert logits to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        # Sampling from the distribution
        actions = torch.multinomial(action_probs, num_samples=1, replacement=True)
        # Subtract 1 to map [0, 1, 2] to [-1, 0, 1]
        actions = actions.squeeze(-1) - 1
        return actions

    def get_det_action(self, state):
        with torch.no_grad():
            action_logits = self.forward(state)
            # print("Action logits:", action_logits)
            action_probs = F.softmax(action_logits, dim=-1)  # Convert logits to probabilities
            # print("Action probabilities:", action_probs)
            action_indices = torch.argmax(action_probs, dim=-1)  # Highest probability index
            # print("Action indices before mapping:", action_indices)
            actions = action_indices - 1  # Map indices: [0, 1, 2] -> [-1, 0, 1]
            # print("Actions after mapping:", actions)
        return actions.cpu().numpy()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(
            state_size + action_size, hidden_size
        )  # action_size added to state_size
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_out = nn.Linear(
            hidden_size, 1
        )  # Output one Q-value for the given state-action pair

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Ensure action is in the correct shape [batch_size, action_size]
        if action.dim() == 1:
            action = action.squeeze(-1)  # Adds a dimension at the end if it's 1D

        # Now let's concatenate along dimension 1 (features dimension)
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value
