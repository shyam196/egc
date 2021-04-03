import torch
import torch.nn.functional as F
from exptune.exptune import ExperimentSettings, Metric, TrialResources
from exptune.hyperparams import LogUniformHyperParam, UniformHyperParam
from exptune.search_strategies import GridSearchStrategy
from exptune.summaries.final_run_summaries import TestMetricSummaries, TrialCurvePlotter
from exptune.utils import PatientStopper
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree, to_undirected
import torch_geometric.transforms as T

from experiments.arxiv.norm_models import (
    EgcArxivNet,
    GatArxivNet,
    GcnArxivNet,
    GinArxivNet,
    MpnnArxivNet,
    PnaArxivNet,
    SageArxivNet,
)
from experiments.exp_config import BaseGraphConfig, Extra
from experiments.utils import data_location, load_pretrained, print_model_parameters

REPEATS = 10
ITERS = 1000
NUM_LAYERS = 3


PRETRAINED_CONF = {
    "gcn": (156, "https://www.dropbox.com/s/g18go108mghf813/checkpoint.pt?dl=1"),
    "gat": (152, "https://www.dropbox.com/s/wu6uoamw63orsx3/checkpoint.pt?dl=1"),
    "gatv2": (112, "https://www.dropbox.com/s/x2t2cp0ukta7mca/checkpoint.pt?dl=1"),
    "gin": (156, "https://www.dropbox.com/s/97d9o3famfofudb/checkpoint.pt?dl=1"),
    "sage": (115, "https://www.dropbox.com/s/xdchn8gnsovfmx4/checkpoint.pt?dl=1"),
    "mpnn_max": (116, "https://www.dropbox.com/s/cq8uj4gc35ihpn2/checkpoint.pt?dl=1"),
    "mpnn_add": (116, "https://www.dropbox.com/s/03ooc86sb2ezv12/checkpoint.pt?dl=1"),
    "pna": (76, "https://www.dropbox.com/s/40c7p0lxlbmlf2o/checkpoint.pt?dl=1"),
    "egc_s": (184, "https://www.dropbox.com/s/a5ptoecuxnal1rd/checkpoint.pt?dl=1"),
    "egc_m": (136, "https://www.dropbox.com/s/3jv50leuivg1zy1/checkpoint.pt?dl=1"),
}


def train(model, data, train_idx, optimizer, device):
    model = model.to(device)
    data = data.to(device)
    train_idx = train_idx.to(device)
    model.train()

    optimizer.zero_grad()
    out = model(x=data.x, edge_index=data.edge_index)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, device):
    model = model.to(device)
    data = data.to(device)
    model.eval()

    out = model(x=data.x, edge_index=data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    valid_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, valid_acc, test_acc


def arxiv_data(root):
    # keep the same data loading logic for all architectures
    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        root=root,
    )
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    split_idx = dataset.get_idx_split()

    return data, split_idx


class ArxivConfig(BaseGraphConfig):
    def __init__(self, hidden) -> None:
        super().__init__(debug_mode=False)
        self.hidden = hidden

    def settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            "arxiv",
            final_repeats=REPEATS,
            final_max_iterations=ITERS,
        )

    def resource_requirements(self) -> TrialResources:
        return TrialResources(cpus=8, gpus=1)

    def search_strategy(self):
        return GridSearchStrategy({"lr": 10, "wd": 2, "dropout": 2})

    def trial_scheduler(self):
        return FIFOScheduler()

    def trial_metric(self) -> Metric:
        return Metric("val_acc", "max")

    def stoppers(self):
        metric = self.trial_metric()
        return [
            PatientStopper(
                metric=metric.name, mode=metric.mode, patience=80, max_iters=ITERS
            )
        ]

    def hyperparams(self):
        return {
            "lr": LogUniformHyperParam(0.001, 0.05, default=0.01),
            "wd": LogUniformHyperParam(0.0001, 0.001, default=0.0005),
            "dropout": UniformHyperParam(0.0, 0.2, default=0.2),
        }

    def optimizer(self, model, hparams):
        return Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["wd"])

    def extra_setup(self, model, optimizer, hparams):
        print_model_parameters(model)
        metric = self.trial_metric()
        return Extra(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr_scheduler=ReduceLROnPlateau(
                optimizer, metric.mode, factor=0.5, patience=40, min_lr=1e-5
            ),
        )

    def data(self, pinned_objs, hparams):
        return arxiv_data(data_location())

    def train(self, model, optimizer, data, extra, iteration: int):
        data, split_idx = data
        return (
            {
                "train_loss": train(
                    model, data, split_idx["train"], optimizer, extra.device
                )
            },
            None,
        )

    def val(self, model, data, extra, iteration: int):
        data, split_idx = data
        trial_metric = self.trial_metric()
        train_acc, val_acc, test_acc = test(
            model, data, split_idx, Evaluator(name="ogbn-arxiv"), extra.device
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


class GcnArxivConfig(ArxivConfig):
    def model(self, hparams):
        return GcnArxivNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            dropout=hparams["dropout"],
            residual=True,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name="gcn",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class Gatv2ArxivConfig(ArxivConfig):
    def model(self, hparams):
        return GatArxivNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            dropout=hparams["dropout"],
            residual=True,
            gat_version=2
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name="gatv2",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )

class GatArxivConfig(ArxivConfig):
    def model(self, hparams):
        return GatArxivNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            dropout=hparams["dropout"],
            residual=True,
            gat_version=1
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name="gat",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class GinArxivConfig(ArxivConfig):
    def model(self, hparams):
        return GinArxivNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            dropout=hparams["dropout"],
            residual=True,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name="gin",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class EgcArxivConfig(ArxivConfig):
    def __init__(
        self, num_heads, num_bases, softmax, aggrs, hidden
    ) -> None:
        super().__init__(hidden=hidden)
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.softmax = softmax
        self.aggrs = aggrs.split(",")
        self.hidden = hidden

    def model(self, hparams):
        return EgcArxivNet(
            self.hidden,
            NUM_LAYERS,
            dropout=hparams["dropout"],
            heads=self.num_heads,
            bases=self.num_bases,
            softmax=self.softmax,
            aggrs=self.aggrs,
            residual=True,
        )

    def pretrained(self, model_dir):
        assert not self.softmax
        if len(self.aggrs) == 1:
            assert "symadd" in self.aggrs
            assert self.hidden == 184 and self.num_heads == 8 and self.num_bases == 4
            model = "egc_s"
        elif len(self.aggrs) == 3:
            assert set(self.aggrs).issuperset({"symadd", "max", "mean"})
            assert self.hidden == 136 and self.num_heads == 4 and self.num_bases == 4
            model = "egc_m"
        else:
            raise ValueError

        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name=model,
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class MpnnArxivConfig(ArxivConfig):
    def __init__(self, hidden, aggr) -> None:
        super().__init__(hidden)
        self.aggr = aggr

    def model(self, hparams):
        return MpnnArxivNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            dropout=hparams["dropout"],
            residual=True,
            aggr=self.aggr,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name=f"mpnn_{self.aggr}",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class PnaArxivConfig(ArxivConfig):
    def __init__(self, hidden) -> None:
        super().__init__(hidden)
        print("Calculating degree histogram required for PNA")
        data, _ = self.data(None, {})
        degs = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        self.deg = torch.bincount(degs)

    def model(self, hparams):
        return PnaArxivNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            dropout=hparams["dropout"],
            residual=True,
            deg=self.deg,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name="pna",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class SageArxivConfig(ArxivConfig):
    def model(self, hparams):
        return SageArxivNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            dropout=hparams["dropout"],
            residual=True,
        )
    
    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="arxiv",
            model_name="sage",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )
