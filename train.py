import logging
import hydra
import torch
import gymnasium as gym
import fruit_gym  # noqa: F401 – registers fruit_gym envs

from omegaconf import OmegaConf, DictConfig

from src.utils import set_seed
from src.utils import Logger
from src.trainer import PPOTrainer

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    device = torch.device("cuda" if cfg.device == "cuda"
                          and torch.cuda.is_available() else "cpu")

    log.warning(f"Using device: {device}")

    log.info("Setting up Tensorboard Logger")
    logger = Logger(log_dir="tensorboard/")

    log.info("Creating environment")
    env = gym.make("PickMultiStrawbEnv")

    trainer = PPOTrainer(cfg=cfg, logger=logger, env=env, device=str(device))

    log.info("Starting PPO training")
    trainer.run_PPO()

    env.close()


if __name__ == "__main__":
    main()
