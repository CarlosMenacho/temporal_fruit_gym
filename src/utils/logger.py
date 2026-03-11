import logging
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self._log = logging.getLogger(__name__)

    def log(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def close(self) -> None:
        self.writer.close()
