from dataclasses import dataclass

import torch
from exptune.exptune import ExperimentConfig
from exptune.summaries.final_run_summaries import TestMetricSummaries
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiments.utils import seed_all


@dataclass
class Extra:
    device: torch.device
    lr_scheduler: ReduceLROnPlateau


class BaseGraphConfig(ExperimentConfig):
    def __init__(self, debug_mode) -> None:
        super().__init__(debug_mode=debug_mode)

    def configure_seeds(self, seed):
        seed_all(seed)

    def extra_setup(self, model, optimizer, hparams):
        metric = self.trial_metric()
        return Extra(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr_scheduler=ReduceLROnPlateau(optimizer, mode=metric.mode),
        )

    def persist_trial(self, checkpoint_dir, model, optimizer, hparams, extra):
        out = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "hparams": hparams,
            "lr_scheduler": extra.lr_scheduler.state_dict(),
        }
        torch.save(out, str(checkpoint_dir / "checkpoint.pt"))

    def restore_trial(self, checkpoint_dir):
        checkpoint = torch.load(str(checkpoint_dir / "checkpoint.pt"))
        hparams = checkpoint["hparams"]

        model = self.model(hparams)
        model.load_state_dict(checkpoint["model"])

        opt = self.optimizer(model, hparams)
        opt.load_state_dict(checkpoint["opt"])

        extra = self.extra_setup(model, opt, hparams)
        extra.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        return model, opt, hparams, extra

    def final_runs_summaries(self):
        return [TestMetricSummaries()]
