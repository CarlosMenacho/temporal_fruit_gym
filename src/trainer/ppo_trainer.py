import os
import torch
import logging
import numpy as np
from omegaconf import DictConfig
from gymnasium import Env
from typing import List, Tuple

from src.utils import Logger, flatten_state, image_to_tensor
from src.networks import PPOAgent

log = logging.getLogger(__name__)


class PPOTrainer:

    def __init__(self, cfg: DictConfig, logger: Logger, env: Env,
                 device: str) -> None:
        self.env = env
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.image_size = cfg.algorithm.image_size

        self.agent = PPOAgent(
            propioceptive_dim=cfg.algorithm.propioceptive_dim,
            action_dim=cfg.algorithm.action_dim,
            use_image=cfg.algorithm.use_image,
            image_channels=cfg.algorithm.image_channels,
            image_size=cfg.algorithm.image_size,
            device=device,
            hidden_dim=cfg.algorithm.hidden_dim,
            learning_rate=cfg.algorithm.learning_rate,
        )

    def run_PPO(self) -> None:
        cfg = self.cfg

        for episode in range(cfg.algorithm.max_episodes):
            states, actions, log_probs, values, rewards = self.collect_trajectory(
            )

            returns = self.calculate_returns(
                rewards, cfg.algorithm.discount_factor).to(self.device)
            values_tensor = torch.cat(values).squeeze(-1).detach()
            advantages = self.calculate_advantages(returns, values_tensor)

            old_log_probs = torch.cat(log_probs).detach()
            actions_tensor = torch.cat(actions)

            for _ in range(cfg.algorithm.ppo_steps):
                metrics = self.update_policy(
                    states=states,
                    actions=actions_tensor,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    returns=returns,
                    epsilon=cfg.algorithm.epsilon,
                    entropy_coeff=cfg.algorithm.entropy_coeff,
                )

            episode_reward = sum(rewards)
            self.agent.timesteps += len(rewards)

            self.logger.log("train/episode_reward", episode_reward, episode)
            self.logger.log_dict(metrics, episode)

            if episode % cfg.print_interval == 0:
                log.info(f"Episode {episode}/{cfg.algorithm.max_episodes} | "
                         f"reward={episode_reward:.2f} | "
                         f"steps={self.agent.timesteps}")

            if episode % cfg.save_interval == 0 and episode > 0:
                os.makedirs("checkpoints", exist_ok=True)
                self.agent.save_checkpoint(f"checkpoints/episode_{episode}.pt")
                log.info(f"Saved checkpoint at episode {episode}")

        self.logger.close()

    def collect_trajectory(self, ) -> Tuple[List, List, List, List, List]:
        """Run one episode and return collected experience."""
        states, actions, log_probs, values, rewards = [], [], [], [], []

        obs, _ = self.env.reset()
        done = False
        truncated = False

        self.agent.network.eval()

        while not (done or truncated):
            obs_tensor = flatten_state(state=obs["state"], device=self.device)
            img_tensor = None
            if self.cfg.algorithm.use_image and "images" in obs:
                img_tensor = image_to_tensor(obs["images"]["wrist2"],
                                             device=self.device,
                                             size=self.image_size)

            with torch.no_grad():
                action, log_prob, _, value = self.agent.network.get_action_and_value(
                    state=obs_tensor,
                    image=img_tensor,
                )

            action_np = action.cpu().numpy().squeeze()
            obs, reward, done, truncated, _ = self.env.step(action_np)

            states.append((obs_tensor, img_tensor))
            actions.append(action)
            log_probs.append(log_prob.unsqueeze(0))
            values.append(value)
            rewards.append(reward)

        return states, actions, log_probs, values, rewards

    def update_policy(
        self,
        states: List[Tuple],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epsilon: float,
        entropy_coeff: float,
        value_coef: float = 0.5,
    ) -> dict:
        self.agent.network.train()

        obs_tensors = torch.cat([s[0] for s in states])
        img_tensors = torch.cat([s[1] for s in states
                                 ]) if states[0][1] is not None else None

        _, new_log_probs, entropy, values = self.agent.network.get_action_and_value(
            state=obs_tensors,
            image=img_tensors,
            action=actions,
        )

        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * ((returns - values.squeeze(-1))**2).mean()
        entropy_loss = -entropy.mean()

        loss = policy_loss + value_coef * value_loss + entropy_coeff * entropy_loss

        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), 0.5)
        self.agent.optimizer.step()

        return {
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/entropy": entropy_loss.item(),
        }

    def evaluate(self, n_episodes: int = 5) -> float:
        """Run evaluation episodes and return mean reward."""
        self.agent.network.eval()
        total_rewards = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0.0

            while not (done or truncated):
                action = self.agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        mean_reward = float(np.mean(total_rewards))
        return mean_reward

    def calculate_returns(self, rewards: List[float],
                          discount_factor: float) -> torch.Tensor:
        returns = []
        cumulative_reward = 0.0
        for r in reversed(rewards):
            cumulative_reward = r + cumulative_reward * discount_factor
            returns.insert(0, cumulative_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def calculate_advantages(self, returns: torch.Tensor,
                             values: torch.Tensor) -> torch.Tensor:
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                         1e-8)
        return advantages
