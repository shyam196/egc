import torch
import torch.nn.functional as F
from exptune.exptune import ExperimentSettings, Metric, TrialResources
from exptune.hyperparams import LogUniformHyperParam
from exptune.search_strategies import GridSearchStrategy
from exptune.summaries.final_run_summaries import TestMetricSummaries, TrialCurvePlotter
from exptune.utils import PatientStopper
from ogb.graphproppred.evaluate import Evaluator
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree

from experiments.code.models import (
    EgcCodeNet,
    GatCodeNet,
    GcnCodeNet,
    GinCodeNet,
    MpnnCodeNet,
    PnaCodeNet,
    SageCodeNet,
)
from experiments.code.utils import code_data, decode_arr_to_seq
from experiments.exp_config import BaseGraphConfig, Extra
from experiments.utils import data_location, load_pretrained, print_model_parameters

REPEATS = 10
ITERS = 25
PATIENCE = 5
NUM_LAYERS = 4


PRETRAINED_CONF = {
    "gcn": (304, "https://www.dropbox.com/s/wnur02753ieqeaq/checkpoint.pt?dl=1"),
    "gat": (304, "https://www.dropbox.com/s/3vko0r40zf0vov3/checkpoint.pt?dl=1"),
    "gatv2": (296, "https://www.dropbox.com/s/otpa7einx12zdkq/checkpoint.pt?dl=1"),
    "gin": (304, "https://www.dropbox.com/s/0nkity46sxwohxn/checkpoint.pt?dl=1"),
    "sage": (293, "https://www.dropbox.com/s/xmrfzfvw4rodmvb/checkpoint.pt?dl=1"),
    "mpnn_max": (292, "https://www.dropbox.com/s/yc0hbtx9kqowlti/checkpoint.pt?dl=1"),
    "mpnn_add": (292, "https://www.dropbox.com/s/bsh2puwta2w2oxz/checkpoint.pt?dl=1"),
    "pna": (272, "https://www.dropbox.com/s/ly15tsl0lt1mvs8/checkpoint.pt?dl=1"),
    "egc_s": (304, "https://www.dropbox.com/s/dm05frxah897zft/checkpoint.pt?dl=1"),
    "egc_m": (300, "https://www.dropbox.com/s/7w7795tv2gkrrj0/checkpoint.pt?dl=1"),
}


def train(model, optimizer, data, device):
    """Modelled off the OGB repo"""
    model = model.to(device)
    model.train()

    num_batches = 0
    loss_total = 0.0

    for batch in data["train"]:
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass

        optimizer.zero_grad()
        pred_list = model(batch)

        loss = 0
        for i in range(len(pred_list)):
            loss += F.cross_entropy(pred_list[i].to(torch.float32), batch.y_arr[:, i])
        loss /= len(pred_list)

        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        num_batches += 1

    return {"train_loss": loss_total / num_batches}


@torch.no_grad()
def evaluate(model, data, device, split, evaluator, idx2vocab):
    """Modelled off the OGB repo"""
    model = model.to(device)
    model.eval()

    seq_ref_list = []
    seq_pred_list = []

    for batch in data[split]:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass

        pred_list = model(batch)
        mat = []
        for i in range(len(pred_list)):
            mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
        mat = torch.cat(mat, dim=1)

        seq_pred = [decode_arr_to_seq(arr, idx2vocab) for arr in mat]
        seq_ref = [batch.y[i] for i in range(len(batch.y))]

        seq_ref_list.extend(seq_ref)
        seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
    return {
        f"{split}_metric": evaluator.eval(input_dict)["F1"],
    }


class CodeConfig(BaseGraphConfig):
    def __init__(self, hidden, use_old_code_dataset) -> None:
        super().__init__(debug_mode=False)
        self.hidden = hidden
        self.use_old_code_dataset = use_old_code_dataset
        self.evaluator = Evaluator(
            "ogbg-code" if use_old_code_dataset else "ogbg-code2"
        )

    def settings(self) -> ExperimentSettings:
        return ExperimentSettings(
            "code",
            final_repeats=REPEATS,
            final_max_iterations=ITERS,
        )

    def resource_requirements(self) -> TrialResources:
        return TrialResources(cpus=7, gpus=1)

    def search_strategy(self):
        return GridSearchStrategy({"lr": 6})

    def trial_scheduler(self):
        metric = self.trial_metric()
        return AsyncHyperBandScheduler(
            metric=metric.name, mode=metric.mode, max_t=ITERS, grace_period=15
        )

    def trial_metric(self) -> Metric:
        return Metric("valid_metric", "max")

    def stoppers(self):
        metric = self.trial_metric()
        return [
            PatientStopper(
                metric=metric.name, mode=metric.mode, patience=PATIENCE, max_iters=ITERS
            )
        ]

    def optimizer(self, model, hparams):
        return Adam(model.parameters(), lr=hparams["lr"])

    def extra_setup(self, model, optimizer, hparams):
        print_model_parameters(model)
        return Extra(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr_scheduler=ReduceLROnPlateau(
                optimizer, mode="max", factor=0.2, patience=10, min_lr=1e-5
            ),
        )

    def data(self, pinned_objs, hparams):
        return code_data(
            data_location(),
            batch_size=128,
            use_old_code_dataset=self.use_old_code_dataset,
        )

    def hyperparams(self):
        return {
            "lr": LogUniformHyperParam(0.0001, 0.01, default=0.001),
        }

    def train(self, model, optimizer, data, extra, iteration: int):
        loader, _ = data
        return train(model, optimizer, loader, extra.device), None

    def val(self, model, data, extra, iteration: int):
        loader, idx2vocab = data
        v_metrics = evaluate(
            model, loader, extra.device, "valid", self.evaluator, idx2vocab
        )
        t_metrics = evaluate(
            model, loader, extra.device, "test", self.evaluator, idx2vocab
        )

        extra.lr_scheduler.step(v_metrics["valid_metric"])

        return {**v_metrics, **t_metrics}, None

    def test(self, model, data, extra):
        loader, idx2vocab = data
        return (
            evaluate(model, loader, extra.device, "test", self.evaluator, idx2vocab),
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
                ],
                name="loss_curves",
            ),
            TestMetricSummaries(),
        ]


class GcnCodeConfig(CodeConfig):
    def model(self, hparams):
        return GcnCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            use_old_code_dataset=self.use_old_code_dataset,
        )

    def pretrained(self, model_dir):
        assert not self.use_old_code_dataset
        return load_pretrained(
            self,
            dataset_name="code2",
            model_name="gcn",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class GatCodeConfig(CodeConfig):
    def model(self, hparams):
        return GatCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            use_old_code_dataset=self.use_old_code_dataset,
            gat_version=1,
        )

    def pretrained(self, model_dir):
        assert not self.use_old_code_dataset
        return load_pretrained(
            self,
            dataset_name="code2",
            model_name="gat",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class Gatv2CodeConfig(CodeConfig):
    def model(self, hparams):
        return GatCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            use_old_code_dataset=self.use_old_code_dataset,
            gat_version=2,
        )

    def pretrained(self, model_dir):
        assert not self.use_old_code_dataset
        return load_pretrained(
            self,
            dataset_name="code2",
            model_name="gatv2",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class GinCodeConfig(CodeConfig):
    def model(self, hparams):
        return GinCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            use_old_code_dataset=self.use_old_code_dataset,
        )

    def pretrained(self, model_dir):
        assert not self.use_old_code_dataset
        return load_pretrained(
            self,
            dataset_name="code2",
            model_name="gin",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class EgcCodeConfig(CodeConfig):
    def __init__(
        self,
        hidden,
        softmax,
        num_bases,
        num_heads,
        aggrs,
        use_old_code_dataset,
    ) -> None:
        super().__init__(hidden=hidden, use_old_code_dataset=use_old_code_dataset)
        self.softmax = softmax
        self.num_bases = num_bases
        self.num_heads = num_heads
        self.aggrs = aggrs.split(",")

    def model(self, hparams):
        return EgcCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            readout="mean",
            softmax=self.softmax,
            bases=self.num_bases,
            heads=self.num_heads,
            aggrs=self.aggrs,
            use_old_code_dataset=self.use_old_code_dataset,
        )

    def pretrained(self, model_dir):
        assert not self.softmax
        if len(self.aggrs) == 1:
            assert "symadd" in self.aggrs
            assert self.hidden == 304 and self.num_heads == 8 and self.num_bases == 8
            model = "egc_s"
        elif len(self.aggrs) == 3:
            assert set(self.aggrs).issuperset({"symadd", "max", "min"})
            assert self.hidden == 300 and self.num_heads == 4 and self.num_bases == 4
            model = "egc_m"
        else:
            raise ValueError

        return load_pretrained(
            self,
            dataset_name="code2",
            model_name=model,
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class MpnnCodeConfig(CodeConfig):
    def __init__(self, hidden, aggr, use_old_code_dataset) -> None:
        super().__init__(hidden, use_old_code_dataset=use_old_code_dataset)
        self.aggr = aggr

    def model(self, hparams):
        return MpnnCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            aggr=self.aggr,
            use_old_code_dataset=self.use_old_code_dataset,
        )

    def pretrained(self, model_dir):
        assert not self.use_old_code_dataset
        return load_pretrained(
            self,
            dataset_name="code2",
            model_name=f"mpnn_{self.aggr}",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class PnaCodeConfig(CodeConfig):
    def __init__(self, hidden, use_old_code_dataset) -> None:
        super().__init__(hidden, use_old_code_dataset=use_old_code_dataset)
        print("Manually calculating degree histogram required by PNA")
        loaders, _ = self.data(None, {})
        deg = torch.zeros(800, dtype=torch.long)
        for data in loaders["train"]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        self.deg = deg

    def model(self, hparams):
        return PnaCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            deg=self.deg,
            use_old_code_dataset=self.use_old_code_dataset,
        )

    def pretrained(self, model_dir):
        assert not self.use_old_code_dataset
        return load_pretrained(
            self,
            dataset_name="code2",
            model_name="pna",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )


class SageCodeConfig(CodeConfig):
    def model(self, hparams):
        return SageCodeNet(
            hidden_dim=self.hidden,
            num_graph_layers=NUM_LAYERS,
            in_feat_drop=0.0,
            residual=True,
            use_old_code_dataset=self.use_old_code_dataset,
        )

    def pretrained(self, model_dir):
        assert not self.use_old_code_dataset
        return load_pretrained(
            self,
            dataset_name="code2",
            model_name="sage",
            hidden=self.hidden,
            model_dir=model_dir,
            pretrained_conf=PRETRAINED_CONF,
        )
