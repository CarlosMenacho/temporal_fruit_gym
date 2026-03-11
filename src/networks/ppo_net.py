import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict


class PPONetwork(nn.Module):

    def __init__(self,
                 propioceptive_dim: int = 18,
                 action_dim: int = 7,
                 hidden_dim: int = 256,
                 use_image: bool = True,
                 image_channels: int = 3):
        """
        Process: tcp_pose (7) + tcp_vel (6) + gripper_pos (1) + gripper_vec (4)
        """
        super().__init__()

        self.action_dim = action_dim
        self.use_image = use_image

        self.propioceptive_net = nn.Sequential(
            nn.Linear(propioceptive_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

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

            cnn_output_size = 64 * 56 * 56

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

    def forward(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            obs: Dictionary with keys:
                - 'proprioceptive': (batch_size, 18)
                - 'image': (batch_size, 3, 64, 64) [optional]
        
        Returns:
            action_mean: (batch_size, 7)
            value: (batch_size, 1)
        """
        prop_features = self.propioceptive_net(obs["state"])

        if self.use_image:
            image_features = self.cnn(obs["images"]["wrist2"])
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
        obs: Dict,
        action: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation dictionary
            action: Action for probability calculation (optional)
            deterministic: If True, return mean without noise
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Action entropy
            value: Value estimate
        """
        action_mean, value = self.forward(obs=obs)

        if deterministic:
            return action_mean, None, None, value

        # Reparameterization trick for continuous actions
        std = torch.exp(self.log_std)
        normal_dist = torch.distributions.Normal(action_mean, std)

        if action is None:
            action = normal_dist.rsample()

        action_squashed = torch.tanh(action)  # [-1, 1]

        # calculate log probability with tanh
        log_prob = normal_dist.log_prob(action)
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        entropy = normal_dist.entropy().sum(dim=-1, keepdim=True)

        return action_squashed, log_prob, entropy, value


class PPOAgent:

    def __init__(
        self,
        propioceptive_dim: int = 18,
        action_dim: int = 7,
        use_image: bool = True,
        device: str = "cuda",
        learning_rate: float = 3e-4,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim

        # network
        self.network = PPONetwork(
            propioceptive_dim=propioceptive_dim,
            action_dim=action_dim,
            use_image=use_image,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=learning_rate)

        self.timesteps = 0

    def predict(self, obs: Dict, deterministic: bool = False) -> np.ndarray:
        """Get action from observation.
        
        Args:
            obs: Dictionary with proprioceptive data (and image if available)
            deterministic: If True, use mean action without noise
        
        Returns:
            Action as numpy array in [-1, 1]
        """
        # Conver to tensor

        obs_tensor = {
            "state": torch.from_numpy(obs["state"]).float().to(self.device)
        }

        if "image" in obs:
            obs_tensor["image"] = torch.from_numpy(
                obs["images"]["wrist2"]).float().to(self.device)

        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(
                obs=obs_tensor, deterministic=deterministic)

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

        action_prob, log_probs, entropy, values = self.network.get_action_and_value(
            obs=obs,
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
