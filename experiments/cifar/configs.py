import torch
import torch.nn.functional as F
from exptune.exptune import ExperimentSettings, Metric, TrialResources
from exptune.hyperparams import (
    ChoiceHyperParam,
    LogUniformHyperParam,
    UniformHyperParam,
)
from exptune.search_strategies import RandomSearchStrategy
from exptune.summaries.final_run_summaries import TestMetricSummaries, TrialCurvePlotter
from exptune.utils import PatientStopper
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset

from experiments.cifar.models import EgcCifarNet, Gatv2CifarNet
from experiments.exp_config import BaseGraphConfig, Extra
from experiments.utils import (
    data_location,
    load_pretrained,
    print_model_parameters,
)

REPEATS = 10
ITERS = 200


PRETRAINED_CONF = {
    "gatv2": (104, "https://www.dropbox.com/s/64pfty38hai1r48/checkpoint.pt?dl=1"),
    "egc_s": (168, "https://www.dropbox.com/s/fmudwmf9vb2r2st/checkpoint.pt?dl=1"),
    "egc_m": (128, "https://www.dropbox.com/s/w2thynj5xi389n4/checkpoint.pt?dl=1"),
}


def _transform_pos(data):
    data.x = torch.cat((data.x, data.pos), dim=1)
    return data


def cifar_data(root, batch_size=128):
    data = dict()
    for split in ["train", "val", "test"]:
        dataset = GNNBenchmarkDataset(root, name="CIFAR10", split=split)
        dataset.transform = _transform_pos
        data[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )

    return data


def cifar_loss(out, batch_y):
    loss = F.cross_entropy(out, batch_y)
    pred = F.log_softmax(out, dim=1).argmax(
        dim=1, keepdim=True
    )  # get the index of the max log-probability
    correct = pred.eq(batch_y.view_as(pred)).sum().item()

    return loss, correct


def train(model, optimizer, data, device):
    model = model.to(device)
    model.train()

    num_batches = 0
    loss_total = 0.0
    correct_total = 0
    elems_total = 0

    for batch in data["train"]:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss, correct = cifar_loss(out, batch.y)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        correct_total += correct
        elems_total += len(batch.y)
        num_batches += 1

    return {
        "train_loss": loss_total / num_batches,
        "train_acc": correct_total / elems_total,
    }


@torch.no_grad()
def evaluate(model, data, device, split):
    model = model.to(device)
    model.eval()

    loss_total = 0.0
    num_batches = 0
    correct_total = 0
    elems_total = 0

    for batch in data[split]:
        batch = batch.to(device)
        out = model(batch)

        loss, correct = cifar_loss(out, batch.y)

        loss_total += loss.item()
        num_batches += 1
        correct_total += correct
        elems_total += len(batch.y)

    return {
        f"{split}_loss": loss_total / num_batches,
        f"{split}_acc": correct_total / elems_total,
    }


class CifarConfig(BaseGraphConfig):
    def __init__(self, num_samples=50) -> None:
        super().__init__(debug_mode=False)
        self.num_samples = num_samples

    def settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            "cifar",
            final_repeats=REPEATS,
            final_max_iterations=ITERS,
        )

    def resource_requirements(self) -> TrialResources:
        return TrialResources(cpus=4, gpus=0.5)

    def search_strategy(self):
        return RandomSearchStrategy(self.num_samples)

    def hyperparams(self):
        return {
            "lr": LogUniformHyperParam(0.0001, 0.01, default=0.001),
            "batch_size": ChoiceHyperParam([32, 64], default=64),
            "wd": LogUniformHyperParam(0.0001, 0.001, default=0.0005),
            "dropout": UniformHyperParam(0.0, 0.5, default=0.0),
        }

    def trial_scheduler(self):
        metric = self.trial_metric()
        return AsyncHyperBandScheduler(
            metric=metric.name, mode=metric.mode, max_t=ITERS, grace_period=20
        )

    def trial_metric(self) -> Metric:
        return Metric("val_acc", "max")

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
        return cifar_data(data_location(), batch_size=hparams["batch_size"])

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
            TrialCurvePlotter(
                ["train_loss", "val_loss", "train_acc", "val_acc"], name="loss_curves"
            ),
            TestMetricSummaries(),
        ]


class CifarGatv2Config(CifarConfig):
    def __init__(self, num_samples, hidden) -> None:
        super().__init__(num_samples=num_samples)
        self.hidden = hidden

    def model(self, hparams):
        return Gatv2CifarNet(
            self.hidden,
            num_graph_layers=4,
            residual=True,
            readout="mean",
            dropout=hparams["dropout"],
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="cifar",
            model_name="gatv2",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class CifarEgcConfig(CifarConfig):
    def __init__(
        self, num_samples, softmax, num_bases, num_heads, aggrs, hidden
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.softmax = softmax
        self.num_bases = num_bases
        self.num_heads = num_heads
        self.aggrs = aggrs.split(",")
        self.hidden = hidden

    def model(self, hparams):
        return EgcCifarNet(
            hidden_dim=self.hidden,
            num_graph_layers=4,
            residual=True,
            readout="mean",
            softmax=self.softmax,
            bases=self.num_bases,
            heads=self.num_heads,
            aggrs=self.aggrs,
            dropout=hparams["dropout"],
        )

    def pretrained(self, model_dir):
        assert not self.softmax
        if len(self.aggrs) == 1:
            assert "symadd" in self.aggrs
            assert self.hidden == 168 and self.num_heads == 8 and self.num_bases == 4
            model = "egc_s"
        elif len(self.aggrs) == 3:
            assert set(self.aggrs).issuperset({"symadd", "std", "max"})
            assert self.hidden == 128 and self.num_heads == 4 and self.num_bases == 4
            model = "egc_m"
        else:
            raise ValueError

        return load_pretrained(
            self,
            dataset_name="cifar",
            model_name=model,
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )
