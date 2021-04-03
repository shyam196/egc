import torch
import torch.nn.functional as F
from exptune.exptune import ExperimentSettings, Metric, TrialResources
from exptune.hyperparams import (
    ChoiceHyperParam,
    LogUniformHyperParam,
)
from exptune.search_strategies import RandomSearchStrategy
from exptune.summaries.final_run_summaries import TestMetricSummaries, TrialCurvePlotter
from exptune.utils import PatientStopper
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC

from experiments.exp_config import BaseGraphConfig, Extra
from experiments.utils import (
    data_location,
    load_pretrained,
    print_model_parameters,
)
from experiments.zinc.models import EgcZincNet, Gatv2ZincNet

REPEATS = 10
ITERS = 200


PRETRAINED_CONF = {
    "gatv2": (104, "https://www.dropbox.com/s/o520nlixkrji9p9/checkpoint.pt?dl=1"),
    "egc_s": (168, "https://www.dropbox.com/s/tao6e8zqxwk582x/checkpoint.pt?dl=1"),
    "egc_m": (124, "https://www.dropbox.com/s/5zsnv0zt1hqbvw4/checkpoint.pt?dl=1"),
}


def zinc_data(root, batch_size=128):
    data = dict()
    for split in ["train", "val", "test"]:
        data[split] = DataLoader(
            ZINC(root, subset=True, split=split),
            batch_size=batch_size,
            shuffle=(split == "train"),
        )

    return data


def zinc_loss(out, batch_y):
    batch_y = batch_y.view_as(out)
    return F.l1_loss(out, batch_y)


def train(model, optimizer, data, device):
    model = model.to(device)
    model.train()

    num_batches = 0
    loss_total = 0.0

    for batch in data["train"]:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = zinc_loss(out, batch.y)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        num_batches += 1

    return {"train_loss": loss_total / num_batches}


@torch.no_grad()
def evaluate(model, data, device, split):
    model = model.to(device)
    model.eval()

    loss_total = 0.0
    num_batches = 0

    for batch in data[split]:
        batch = batch.to(device)
        out = model(batch)

        loss_total += zinc_loss(out, batch.y).item()
        num_batches += 1

    return {f"{split}_loss": loss_total / num_batches}


class ZincConfig(BaseGraphConfig):
    def __init__(self, num_samples=50) -> None:
        super().__init__(debug_mode=False)
        self.num_samples = num_samples

    def settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            "zinc",
            final_repeats=REPEATS,
            final_max_iterations=ITERS,
        )

    def resource_requirements(self) -> TrialResources:
        return TrialResources(cpus=2, gpus=0.25)

    def search_strategy(self):
        return RandomSearchStrategy(self.num_samples)

    def trial_scheduler(self):
        metric = self.trial_metric()
        return AsyncHyperBandScheduler(
            metric=metric.name, mode=metric.mode, max_t=ITERS, grace_period=20
        )

    def trial_metric(self) -> Metric:
        return Metric("val_loss", "min")

    def stoppers(self):
        metric = self.trial_metric()
        return [
            PatientStopper(
                metric=metric.name, mode=metric.mode, patience=20, max_iters=ITERS
            )
        ]

    def optimizer(self, model, hparams):
        return Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])

    def extra_setup(self, model, optimizer, hparams):
        print_model_parameters(model)
        metric = self.trial_metric()
        return Extra(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr_scheduler=ReduceLROnPlateau(
                optimizer, metric.mode, factor=0.5, patience=10, min_lr=1e-5
            ),
        )

    def data(self, pinned_objs, hparams):
        return zinc_data(data_location(), batch_size=hparams["batch_size"])

    def train(self, model, optimizer, data, extra, iteration: int):
        return train(model, optimizer, data, extra.device), None

    def val(self, model, data, extra, iteration: int):
        trial_metric = self.trial_metric()
        metrics = evaluate(model, data, extra.device, "val")
        extra.lr_scheduler.step(metrics[trial_metric.name])
        return metrics, None

    def test(self, model, data, extra):
        return evaluate(model, data, extra.device, "test"), None

    def persist_trial(self, checkpoint_dir, model, optimizer, hparams, extra):
        out = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "lr_scheduler": extra.lr_scheduler.state_dict(),
            "hparams": hparams,
        }
        torch.save(out, str(checkpoint_dir / "checkpoint.pt"))

    def restore_trial(self, checkpoint_dir, map_location=None):
        checkpoint = torch.load(
            str(checkpoint_dir / "checkpoint.pt"), map_location=map_location
        )
        hparams = checkpoint["hparams"]

        model = self.model(hparams)
        model.load_state_dict(checkpoint["model"])

        opt = self.optimizer(model, hparams)
        opt.load_state_dict(checkpoint["opt"])

        extra = self.extra_setup(model, opt, hparams)
        extra.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        return model, opt, hparams, extra

    def final_runs_summaries(self):
        return [
            TrialCurvePlotter(["train_loss", "val_loss"], name="loss_curves"),
            TestMetricSummaries(),
        ]


class ZincGatv2Config(ZincConfig):
    def __init__(self, num_samples, hidden) -> None:
        super().__init__(num_samples=num_samples)
        self.hidden = hidden

    def hyperparams(self):
        return {
            "lr": LogUniformHyperParam(0.0001, 0.01, default=0.001),
            "batch_size": ChoiceHyperParam([64, 128], default=128),
            "wd": LogUniformHyperParam(0.0001, 0.001, default=0.0005),
        }

    def model(self, hparams):
        return Gatv2ZincNet(
            self.hidden,
            num_graph_layers=4,
            in_feat_drop=0.0,
            residual=True,
            readout="mean",
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="zinc",
            model_name="gatv2",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class ZincEgcConfig(ZincConfig):
    def __init__(
        self,
        num_samples,
        softmax,
        num_bases,
        num_heads,
        aggrs,
        hidden,
        sigmoid,
        hardtanh,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.softmax = softmax
        self.num_bases = num_bases
        self.num_heads = num_heads
        self.aggrs = aggrs.split(",")
        self.hidden = hidden
        self.sigmoid = sigmoid
        self.hardtanh = hardtanh

    def hyperparams(self):
        return {
            "lr": LogUniformHyperParam(0.0001, 0.01, default=0.001),
            "batch_size": ChoiceHyperParam([64, 128], default=128),
            "wd": LogUniformHyperParam(0.0001, 0.001, default=0.0005),
        }

    def model(self, hparams):
        return EgcZincNet(
            hidden_dim=self.hidden,
            num_graph_layers=4,
            in_feat_drop=0.0,
            residual=True,
            readout="mean",
            softmax=self.softmax,
            sigmoid=self.sigmoid,
            hardtanh=self.hardtanh,
            bases=self.num_bases,
            heads=self.num_heads,
            aggrs=self.aggrs,
        )

    def pretrained(self, model_dir):
        assert not self.softmax
        if len(self.aggrs) == 1:
            assert "symadd" in self.aggrs
            assert self.hidden == 168 and self.num_heads == 8 and self.num_bases == 4
            model = "egc_s"
        elif len(self.aggrs) == 3:
            assert set(self.aggrs).issuperset({"add", "std", "max"})
            assert self.hidden == 124 and self.num_heads == 4 and self.num_bases == 4
            model = "egc_m"
        else:
            raise ValueError

        return load_pretrained(
            self,
            dataset_name="zinc",
            model_name=model,
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )
