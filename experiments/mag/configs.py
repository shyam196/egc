"""Code in this file is heavily modelled off OGB examples"""
import torch
import torch.nn.functional as F
from exptune.exptune import ExperimentSettings, Metric, TrialResources
from exptune.hyperparams import (
    ChoiceHyperParam,
)
from exptune.search_strategies import GridSearchStrategy
from exptune.summaries.final_run_summaries import TestMetricSummaries, TrialCurvePlotter
from exptune.utils import PatientStopper
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import torch_geometric.transforms as T

from experiments.exp_config import BaseGraphConfig, Extra
from experiments.utils import data_location, print_model_parameters
from experiments.mag.models import EGC

REPEATS = 10
ITERS = 200
PATIENCE = 50
NUM_LAYERS = 2


def train(model, data, train_idx, optimizer, device):
    model = model.to(device)
    data = data.to(device)
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, device):
    model = model.to(device)
    data = data.to(device)
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["train"]["paper"]],
            "y_pred": y_pred[split_idx["train"]["paper"]],
        }
    )["acc"]
    valid_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["valid"]["paper"]],
            "y_pred": y_pred[split_idx["valid"]["paper"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["test"]["paper"]],
            "y_pred": y_pred[split_idx["test"]["paper"]],
        }
    )["acc"]

    return train_acc, valid_acc, test_acc


def mag_data(root):
    dataset = PygNodePropPredDataset(name="ogbn-mag", root=root)
    rel_data = dataset[0]

    # We are only interested in paper <-> paper relations.
    data = Data(
        x=rel_data.x_dict["paper"],
        edge_index=rel_data.edge_index_dict[("paper", "cites", "paper")],
        y=rel_data.y_dict["paper"],
    )

    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()

    return data, split_idx


class MagConfig(BaseGraphConfig):
    def __init__(self, hidden) -> None:
        super().__init__(debug_mode=False)
        self.hidden = hidden

    def settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            "mag",
            final_repeats=REPEATS,
            final_max_iterations=ITERS,
            checkpoint_at_end=False,
            checkpoint_freq=0,
        )

    def resource_requirements(self) -> TrialResources:
        return TrialResources(cpus=8, gpus=1)

    def search_strategy(self):
        return GridSearchStrategy(dict())

    def trial_scheduler(self):
        return FIFOScheduler()

    def trial_metric(self) -> Metric:
        return Metric("val_acc", "max")

    def stoppers(self):
        metric = self.trial_metric()
        return [
            PatientStopper(
                metric=metric.name, mode=metric.mode, patience=PATIENCE, max_iters=ITERS
            )
        ]

    def hyperparams(self):
        return {
            "lr": ChoiceHyperParam([0.005, 0.01, 0.05, 0.1], default=0.01),
            "wd": ChoiceHyperParam([5e-5, 1e-4, 5e-4, 1e-3], default=1e-3),
            "dropout": ChoiceHyperParam([0.3, 0.5, 0.7], default=0.5),
        }

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
        return mag_data(data_location())

    def train(self, model, optimizer, data, extra, iteration: int):
        data, split_idx = data
        return (
            {
                "train_loss": train(
                    model, data, split_idx["train"]["paper"], optimizer, extra.device
                )
            },
            None,
        )

    def val(self, model, data, extra, iteration: int):
        data, split_idx = data
        trial_metric = self.trial_metric()
        train_acc, val_acc, test_acc = test(
            model, data, split_idx, Evaluator(name="ogbn-mag"), extra.device
        )
        metrics = dict(train_acc=train_acc, val_acc=val_acc, test_acc=test_acc)
        extra.lr_scheduler.step(metrics[trial_metric.name])
        return metrics, None

    def test(self, model, data, extra):
        return self.val(model, data, extra, 0)

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
                ["train_loss", "train_acc", "val_acc", "test_acc"], name="loss_curves"
            ),
            TestMetricSummaries(),
        ]


class EgcMagConfig(MagConfig):
    def __init__(self, hidden, heads, bases, aggrs) -> None:
        super().__init__(hidden=hidden)
        assert heads is not None and bases is not None

        self.heads = heads
        self.bases = bases
        self.aggrs = aggrs.split(",")

    def model(self, hparams):
        return EGC(
            self.hidden,
            NUM_LAYERS,
            dropout=hparams["dropout"],
            num_heads=self.heads,
            num_bases=self.bases,
            aggrs=self.aggrs,
        )
