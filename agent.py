import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
import numpy as np


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
import numpy as np


class SAC(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, device):
        """Initialize an Agent object."""
        super(SAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 1e-3
        self.clip_grad_param = 1

        self.alpha = 0.2  # Example static initialization
        self.target_entropy = (
            -action_size
        )  # Typical heuristic is negative action dimension
        self.log_alpha = torch.tensor(np.log(self.alpha)).to(device)
        self.log_alpha.requires_grad = True  # Allows optimization
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=learning_rate
        )  # Separate optimizer for alpha

        # Actor Network
        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=learning_rate
        )

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

    def get_action(self, state, add_noise=True, noise_scale=0.1):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action_logits = self.actor_local.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)
            if add_noise:
                noise = torch.randn_like(action_probs) * noise_scale
                action_probs = action_probs + noise
                action_probs = F.softmax(action_probs, dim=-1)  # Re-normalize
            if action_probs.dim() == 3:
                action_probs = action_probs.view(
                    -1, 3
                )  # Flatten to (batch_size*action_size, 3)
            actions = torch.multinomial(action_probs, num_samples=1, replacement=True)
            actions = (
                actions.view(-1, 6) - 1
            )  # Reshape to (batch_size, action_size) and adjust to [-1, 0, 1]
        return actions[0].cpu().numpy()

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_Q = torch.min(q1, q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q)).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        states, next_states, actions, rewards, dones = (
            states.to(self.device),
            next_states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
        )

        actions = actions.long()

        # ---------------------------- Update Critic ---------------------------- #
        with torch.no_grad():
            next_action_logits = self.actor_local(next_states)
            next_action_probs = F.softmax(next_action_logits, dim=-1)
            next_actions = torch.argmax(next_action_probs, dim=-1) - 1
            next_actions = next_actions.squeeze(-1)  # Ensure proper shape

            Q_targets_next1 = self.critic1_target(next_states, next_actions.float())
            Q_targets_next2 = self.critic2_target(next_states, next_actions.float())
            Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)

            Q_targets = rewards + (gamma * (1 - dones) * Q_targets_next)

        Q_expected1 = self.critic1(states, actions.float())
        Q_expected2 = self.critic2(states, actions.float())

        critic1_loss = F.mse_loss(Q_expected1, Q_targets)
        critic2_loss = F.mse_loss(Q_expected2, Q_targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ---------------------------- Update Actor ---------------------------- #
        action_logits = self.actor_local(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(
            actions.squeeze(-1) + 1
        )  # Adjust index to match action probability index
        entropy = dist.entropy()  # Calculate entropy

        current_alpha = self.log_alpha.exp()
        actor_loss = -(
            self.critic1(states, actions.float()).mean()
            - current_alpha * entropy.mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- Update Alpha ----------------------- #
        alpha_loss = -(
            self.log_alpha.exp() * (log_probs + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ----------------------- Update Target Networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
