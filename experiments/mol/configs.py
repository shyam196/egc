import torch
import torch.nn.functional as F
from exptune.exptune import ExperimentSettings, Metric, TrialResources
from exptune.hyperparams import (
    ChoiceHyperParam,
    LogUniformHyperParam,
    UniformHyperParam,
)
from exptune.search_strategies import GridSearchStrategy
from exptune.summaries.final_run_summaries import TestMetricSummaries, TrialCurvePlotter
from exptune.utils import PatientStopper
from ogb.graphproppred.evaluate import Evaluator
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiments.exp_config import BaseGraphConfig, Extra
from experiments.mol.pna_style_models import (
    EgcHIVNet,
    GatHIVNet,
    GcnHIVNet,
    GinHIVNet,
    MpnnHIVNet,
    SageHIVNet,
)
from experiments.mol.utils import mol_data
from experiments.utils import (
    data_location,
    load_pretrained,
    print_model_parameters,
)


REPEATS = 10
ITERS = 100
NUM_LAYERS = 4


PRETRAINED_CONF = {
    "gcn": (240, "https://www.dropbox.com/s/h0gvaydnuig90hp/checkpoint.pt?dl=1"),
    "gat": (240, "https://www.dropbox.com/s/etygnsb0qrbd117/checkpoint.pt?dl=1"),
    "gatv2": (184, "https://www.dropbox.com/s/yhywch0icc6c5wv/checkpoint.pt?dl=1"),
    "gin": (240, "https://www.dropbox.com/s/y02zzumv8sfpfr0/checkpoint.pt?dl=1"),
    "sage": (180, "https://www.dropbox.com/s/yzxfc0yaowglcsv/checkpoint.pt?dl=1"),
    "mpnn_max": (180, "https://www.dropbox.com/s/naq6g9qjrayzsol/checkpoint.pt?dl=1"),
    "mpnn_add": (180, "https://www.dropbox.com/s/3ne9ghput6rkes8/checkpoint.pt?dl=1"),
    "egc_s": (296, "https://www.dropbox.com/s/vr8fd4u6gw0433n/checkpoint.pt?dl=1"),
    "egc_m": (224, "https://www.dropbox.com/s/qmcxaugn8a7jsrb/checkpoint.pt?dl=1"),
}


def train(model, optimizer, data, device):
    model = model.to(device)
    model.train()

    num_batches = 0
    loss_total = 0.0

    for batch in data["train"]:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        # nan targets (unlabeled) should be ignored when computing training loss
        is_labeled = batch.y == batch.y
        loss = F.binary_cross_entropy_with_logits(
            out.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
        )
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        num_batches += 1

    return {"train_loss": loss_total / num_batches}


@torch.no_grad()
def evaluate(model, data, device, split):
    model = model.to(device)
    model.eval()

    evaluator = Evaluator(f"ogbg-molhiv")
    y_true = []
    y_pred = []
    loss_total = 0.0
    num_batches = 0

    for batch in data[split]:
        batch = batch.to(device)
        pred = model(batch)

        is_labeled = batch.y == batch.y
        loss_total += F.binary_cross_entropy_with_logits(
            pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
        ).item()

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
        num_batches += 1

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return {
        f"{split}_metric": evaluator.eval(input_dict)["rocauc"],
        f"{split}_loss": loss_total / num_batches,
    }


class MolConfig(BaseGraphConfig):
    def __init__(self, hidden) -> None:
        super().__init__(debug_mode=False)
        self.hidden = hidden

    def settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            f"mol-hiv", final_repeats=REPEATS, final_max_iterations=ITERS,
        )

    def resource_requirements(self) -> TrialResources:
        return TrialResources(cpus=2, gpus=0.25)

    def search_strategy(self):
        return GridSearchStrategy({"lr": 5, "wd": 2, "dropout": 2})

    def trial_scheduler(self):
        metric = self.trial_metric()
        return AsyncHyperBandScheduler(
            metric=metric.name, mode=metric.mode, max_t=ITERS, grace_period=30
        )

    def trial_metric(self) -> Metric:
        return Metric("valid_metric", "max")

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
        return Extra(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr_scheduler=ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5
            ),
        )

    def data(self, pinned_objs, hparams):
        return mol_data(
            data_location(), dataset="hiv", batch_size=hparams["batch_size"]
        )

    def hyperparams(self):
        return {
            "lr": LogUniformHyperParam(0.0001, 0.01, default=0.001),
            "batch_size": ChoiceHyperParam([32, 64], default=32),
            "wd": LogUniformHyperParam(0.0001, 0.001, default=0.0005),
            "dropout": UniformHyperParam(0.0, 0.2, default=0.2),
        }

    def train(self, model, optimizer, data, extra, iteration: int):
        return train(model, optimizer, data, extra.device), None

    def val(self, model, data, extra, iteration: int):
        v_metrics = evaluate(
            model,
            data,
            extra.device,
            "valid",
        )
        t_metrics = evaluate(
            model,
            data,
            extra.device,
            "test",
        )

        extra.lr_scheduler.step(v_metrics["valid_metric"])

        return {**v_metrics, **t_metrics}, None

    def test(self, model, data, extra):
        return (
            evaluate(
                model,
                data,
                extra.device,
                "test",
            ),
            None,
        )

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
                [
                    "train_loss",
                    "valid_metric",
                    "test_metric",
                    "valid_loss",
                    "test_loss",
                ],
                name="loss_curves",
            ),
            TestMetricSummaries(),
        ]


class GcnMolConfig(MolConfig):
    def model(self, hparams):
        return GcnHIVNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=hparams["dropout"],
            residual=True,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="hiv",
            model_name="gcn",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class Gatv2MolConfig(MolConfig):
    def model(self, hparams):
        return GatHIVNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=hparams["dropout"],
            residual=True,
            gat_version=2,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="hiv",
            model_name="gatv2",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )

class GatMolConfig(MolConfig):
    def model(self, hparams):
        return GatHIVNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=hparams["dropout"],
            residual=True,
            gat_version=1,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="hiv",
            model_name="gat",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class GinMolConfig(MolConfig):
    def model(self, hparams):
        return GinHIVNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=hparams["dropout"],
            residual=True,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="hiv",
            model_name="gin",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class EgcMolConfig(MolConfig):
    def __init__(
        self, hidden, softmax, num_bases, num_heads, aggrs
    ) -> None:
        super().__init__(hidden=hidden)
        self.softmax = softmax
        self.num_bases = num_bases
        self.num_heads = num_heads
        self.aggrs = aggrs.split(",")

    def model(self, hparams):
        return EgcHIVNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=hparams["dropout"],
            residual=True,
            readout="mean",
            softmax=self.softmax,
            bases=self.num_bases,
            heads=self.num_heads,
            aggrs=self.aggrs,
        )

    def pretrained(self, model_dir):
        assert not self.softmax
        if len(self.aggrs) == 1:
            assert "symadd" in self.aggrs
            assert self.hidden == 296 and self.num_heads == 8 and self.num_bases == 4
            model = "egc_s"
        elif len(self.aggrs) == 3:
            assert set(self.aggrs).issuperset({"add", "max", "mean"})
            assert self.hidden == 224 and self.num_heads == 4 and self.num_bases == 4
            model = "egc_m"
        else:
            raise ValueError

        return load_pretrained(
            self,
            dataset_name="hiv",
            model_name=model,
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class MpnnMolConfig(MolConfig):
    def __init__(self, hidden, aggr) -> None:
        super().__init__(hidden)
        self.aggr = aggr

    def model(self, hparams):
        return MpnnHIVNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=hparams["dropout"],
            residual=True,
            aggr=self.aggr,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="hiv",
            model_name=f"mpnn_{self.aggr}",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class SageMolConfig(MolConfig):
    def model(self, hparams):
        return SageHIVNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=hparams["dropout"],
            residual=True,
        )

    def pretrained(self, model_dir):
        return load_pretrained(
            self,
            dataset_name="hiv",
            model_name="sage",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )
