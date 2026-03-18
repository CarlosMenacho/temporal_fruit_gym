import argparse
import gymnasium as gym
import fruit_gym
import torch

from src.networks import PPOAgent


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy")
    parser.add_argument("checkpoint",
                        type=str,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--episodes",
                        type=int,
                        default=5,
                        help="Number of eval episodes")
    parser.add_argument("--no-render",
                        action="store_true",
                        help="Disable rendering")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    render_mode = None if args.no_render else "human"

    env = gym.make("PickMultiStrawbEnv", render_mode=render_mode)

    agent = PPOAgent(device=device)
    agent.load_checkpoint(args.checkpoint)
    agent.network.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Timesteps trained: {agent.timesteps}\n")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

        print(f"Episode {ep + 1}: reward={episode_reward:.3f}  steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
