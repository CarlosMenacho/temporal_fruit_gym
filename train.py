import logging
import hydra
import torch

from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_original_cwd

from src.utils import set_seed
from src.utils import Logger

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    device = torch.device("cuda" if cfg.device == "cuda"
                          and torch.cuda.is_available() else "cpu")

    log.warning(f"Using device: {device}")

    log.info(f"Setting up Tensorboard Logger")
    logger = Logger(log_dir="tensorboard/")


if __name__ == "__main__":
    main()
