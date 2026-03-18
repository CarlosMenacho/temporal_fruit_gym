import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List

from src.utils import image_to_tensor, flatten_state


class PPONetwork(nn.Module):

    def __init__(self,
                 propioceptive_dim: int = 18,
                 action_dim: int = 7,
                 hidden_dim: int = 256,
                 use_image: bool = True,
                 image_channels: int = 3,
                 image_size: int = 128):
        """
        Process: tcp_pose (7) + tcp_vel (6) + gripper_pos (1) + gripper_vec (4)
        """
        super().__init__()

        self.action_dim = action_dim
        self.use_image = use_image

        self.propioceptive_net = nn.Sequential(
            nn.Linear(propioceptive_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_image:
            # first using a single image
            # TODO: add temporal dependencies
            self.cnn = nn.Sequential(
                nn.Conv2d(image_channels,
                          32,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                dummy = torch.zeros(1, image_channels, image_size, image_size)
                cnn_output_size = self.cnn(dummy).shape[1]

            self.cnn_fc = nn.Sequential(
                nn.Linear(cnn_output_size, hidden_dim),
                nn.ReLU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )
            feature_dim = hidden_dim

        else:
            feature_dim = hidden_dim

        # actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # critic value function
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor,
                img: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            obs: torch.Tensor 
                contains all state compressed in a tensor of shape (1,18)
            img: torch.Tensor
                contains image tensor from robot
        
        Returns:
            action_mean: (batch_size, 7)
            value: (batch_size, 1)
        """
        prop_features = self.propioceptive_net(obs)

        if self.use_image:
            image_features = self.cnn(img)
            image_features = self.cnn_fc(image_features)
            features = torch.cat([prop_features, image_features], dim=1)
            features = self.fusion(features)
        else:
            features = prop_features

        action_mean = self.actor_mean(features)
        value = self.critic(features)

        return action_mean, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        image: torch.Tensor = None,
        action: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action.

        Returns:
            action: (batch, action_dim)
            log_prob: (batch,)
            entropy: (batch,)
            value: (batch, 1)
        """
        action_mean, value = self.forward(state, image)

        std = self.log_std.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)

        if action is None:
            action = action_mean if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value


class PPOAgent:

    def __init__(
        self,
        propioceptive_dim: int = 18,
        hidden_dim: int = 256,
        image_channels: int = 3,
        image_size: int = 128,
        action_dim: int = 7,
        use_image: bool = True,
        device: str = "cuda",
        learning_rate: float = 3e-4,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.image_size = image_size

        # network
        self.network = PPONetwork(
            propioceptive_dim=propioceptive_dim,
            action_dim=action_dim,
            use_image=use_image,
            hidden_dim=hidden_dim,
            image_channels=image_channels,
            image_size=image_size,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=learning_rate)

        self.timesteps = 0

    def calculate_returns(self, rewards: List[float],
                          discount_factor: float) -> torch.Tensor:
        returns = []
        cumulative_rewars = 0
        for r in reversed(rewards):
            cumulative_rewars = r + cumulative_rewars * discount_factor
            returns.insert(0, cumulative_rewars)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / returns.std()
        return returns

    def calculate_advantages(self, returns: torch.Tensor,
                             values: torch.Tensor) -> torch.Tensor:
        advatages = returns - values
        advatages = (advatages - advatages.mean()) / advatages.std()
        return advatages

    def calculate_surrogate_loss(self, actions_log_prob_old: torch.Tensor,
                                 actions_log_prob_new: torch.Tensor,
                                 epsilon: float, advantages: torch.Tensor):
        advantages = advantages.detach()
        policy_ratio = (actions_log_prob_new - actions_log_prob_old).exp()

        surrogate_loss1 = policy_ratio * advantages
        surrogate_loss2 = torch.clamp(
            policy_ratio, min=1.0 - epsilon, max=1.0 + epsilon) * advantages

        surrogate_loss = torch.min(surrogate_loss1, surrogate_loss2)
        return surrogate_loss

    def predict(self, obs: Dict, deterministic: bool = False) -> np.ndarray:
        """Get action from observation.
        
        Args:
            obs: Dictionary with proprioceptive data (and image if available)
            deterministic: If True, use mean action without noise
        
        Returns:
            Action as numpy array in [-1, 1]
        """
        obs_tensor = flatten_state(state=obs["state"], device=self.device)

        image_tensor = None
        if "images" in obs:
            image_tensor = image_to_tensor(
                image=obs["images"]["wrist2"],
                device=self.device,
                size=self.image_size,
            )

        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(
                state=obs_tensor,
                image=image_tensor,
                deterministic=deterministic)

        return action.cpu().numpy().squeeze()

    def compute_loss(
        self,
        obs: Dict,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_prob: torch.Tensor,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> Dict[str, float]:
        """Compute PPO loss.
        
        Returns:
            Dictionary with loss components
        """

        obs_tensor = flatten_state(obs["state"], device=self.device)
        img_tensor = image_to_tensor(obs["images"]["wrist2"],
                                     device=self.device)
        action_prob, log_probs, entropy, values = self.network.get_action_and_value(
            state=obs_tensor,
            image=img_tensor,
            action=actions,
        )

        ratio = torch.exp(log_probs - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * ((returns - values.squeeze())**2).mean()

        entropy_loss = -entropy.mean()

        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        return {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/entropy": entropy_loss.item(),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "timesteps": self.timesteps
            }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.timesteps = checkpoint["timesteps"]
