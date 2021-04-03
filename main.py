import os
from pathlib import Path

import click
import ray
from exptune import check_config, run_search, train_final_models
from exptune.utils import dump_invocation_state

from experiments.arxiv.configs import (
    EgcArxivConfig,
    GatArxivConfig,
    Gatv2ArxivConfig,
    GcnArxivConfig,
    GinArxivConfig,
    MpnnArxivConfig,
    PnaArxivConfig,
    SageArxivConfig,
    arxiv_data,
)
from experiments.cifar.configs import (
    CifarEgcConfig,
    CifarGatv2Config,
    cifar_data,
)
from experiments.code.configs import (
    EgcCodeConfig,
    GatCodeConfig,
    Gatv2CodeConfig,
    GcnCodeConfig,
    GinCodeConfig,
    MpnnCodeConfig,
    PnaCodeConfig,
    SageCodeConfig,
)
from experiments.code.utils import code_data
from experiments.mol.configs import (
    EgcMolConfig,
    GatMolConfig,
    Gatv2MolConfig,
    GcnMolConfig,
    GinMolConfig,
    MpnnMolConfig,
    SageMolConfig,
)
from experiments.mol.utils import mol_data
from experiments.utils import data_location
from experiments.zinc.configs import (
    ZincEgcConfig,
    ZincGatv2Config,
    zinc_data,
)
from experiments.rmag.configs import REgcMagConfig, rmag_data
from experiments.mag.configs import EgcMagConfig, mag_data


def _zinc(model, num_samples, egc_num_bases, egc_num_heads, aggrs, hidden):
    zinc_data(data_location())

    if model == "egc":
        config = ZincEgcConfig(
            num_samples=num_samples,
            softmax=False,
            sigmoid=False,
            hardtanh=False,
            num_bases=egc_num_bases,
            num_heads=egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )
    elif model == "gatv2":
        config = ZincGatv2Config(num_samples=num_samples, hidden=hidden)
    else:
        raise ValueError

    return config


def _mol(model, num_heads, num_bases, aggrs, hidden):
    mol_data(data_location(), "hiv")
    if model == "egc":
        return EgcMolConfig(
            hidden=hidden,
            num_bases=num_bases,
            num_heads=num_heads,
            softmax=False,
            aggrs=aggrs,
        )

    elif model == "gcn":
        return GcnMolConfig(hidden)
    elif model == "gat":
        return GatMolConfig(hidden)
    elif model == "gatv2":
        return Gatv2MolConfig(hidden)
    elif model == "gin":
        return GinMolConfig(hidden)
    elif model in ["mpnn-sum", "mpnn-max"]:
        return MpnnMolConfig(hidden, aggr="add" if "sum" in model else "max")
    elif model == "sage":
        return SageMolConfig(hidden)
    else:
        raise ValueError


def _arxiv(model, num_heads, num_bases, aggrs, hidden):
    arxiv_data(data_location())

    if model == "gcn":
        return GcnArxivConfig(hidden)
    elif model == "gat":
        return GatArxivConfig(hidden)
    elif model == "gatv2":
        return Gatv2ArxivConfig(hidden)
    elif model == "gin":
        return GinArxivConfig(hidden)
    elif model == "egc":
        return EgcArxivConfig(
            num_heads=num_heads,
            num_bases=num_bases,
            softmax=False,
            aggrs=aggrs,
            hidden=hidden,
        )
    elif model in ["mpnn-sum", "mpnn-max"]:
        return MpnnArxivConfig(hidden, aggr="add" if "sum" in model else "max")
    elif model == "pna":
        return PnaArxivConfig(hidden)
    elif model == "sage":
        return SageArxivConfig(hidden)
    else:
        raise ValueError


def _cifar(model, num_samples, egc_num_bases, egc_num_heads, aggrs, hidden):
    cifar_data(data_location())

    if model == "egc":
        config = CifarEgcConfig(
            num_samples=num_samples,
            softmax=False,
            num_bases=egc_num_bases,
            num_heads=egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )
    elif model == "gatv2":
        config = CifarGatv2Config(num_samples=num_samples, hidden=hidden)
    else:
        raise ValueError

    return config


def _code(
    model,
    num_heads,
    num_bases,
    aggrs,
    hidden,
    use_old_code_dataset,
):
    code_data(data_location(), use_old_code_dataset=use_old_code_dataset)
    if model == "egc":
        return EgcCodeConfig(
            hidden=hidden,
            num_bases=num_bases,
            num_heads=num_heads,
            softmax=False,
            aggrs=aggrs,
            use_old_code_dataset=use_old_code_dataset,
        )

    if model == "gcn":
        return GcnCodeConfig(hidden, use_old_code_dataset=use_old_code_dataset)
    elif model == "gat":
        return GatCodeConfig(hidden, use_old_code_dataset=use_old_code_dataset)
    elif model == "gatv2":
        return Gatv2CodeConfig(hidden, use_old_code_dataset=use_old_code_dataset)
    elif model == "gin":
        return GinCodeConfig(hidden, use_old_code_dataset=use_old_code_dataset)
    elif model in ["mpnn-sum", "mpnn-max"]:
        return MpnnCodeConfig(
            hidden,
            aggr="add" if "sum" in model else "max",
            use_old_code_dataset=use_old_code_dataset,
        )
    elif model == "pna":
        return PnaCodeConfig(hidden, use_old_code_dataset=use_old_code_dataset)
    elif model == "sage":
        return SageCodeConfig(hidden, use_old_code_dataset=use_old_code_dataset)
    else:
        raise ValueError


def _rmag(model, num_heads, num_bases, aggrs, hidden):
    rmag_data(data_location())
    if model == "egc":
        return REgcMagConfig(hidden, num_heads, num_bases)
    else:
        raise ValueError


def _mag(model, num_heads, num_bases, aggrs, hidden):
    mag_data(data_location())
    if model == "egc":
        return EgcMagConfig(hidden, num_heads, num_bases, aggrs=aggrs)
    else:
        raise ValueError


@click.command()
@click.argument("exp_directory", type=click.Path(file_okay=False))
@click.argument(
    "model",
    type=click.Choice(
        [
            "gcn",
            "gat",
            "egc",
            "gin",
            "mpnn-sum",
            "mpnn-max",
            "pna",
            "sage",
            "gatv2",
        ]
    ),
)
@click.argument(
    "dataset",
    type=click.Choice(
        ["zinc", "hiv", "arxiv", "cifar", "code", "rmag", "mag"]
    ),
)
@click.option("--num-samples", type=int, default=50)
@click.option("--check", is_flag=True)
@click.option("--check-epochs", type=int, default=200)
@click.option("--use-default-hparams", is_flag=True)
@click.option("--hparams", type=str, default=None)
@click.option("--egc-num-bases", type=int, default=None)
@click.option("--egc-num-heads", type=int, default=None)
@click.option("--final-runs", type=int, default=None)
@click.option("--aggrs", type=str, default=None)
@click.option("--hidden", type=int, default=None)
@click.option("--seed-base", type=int, default=0)
@click.option("--use-old-code-dataset", is_flag=True)
@click.option("--pretrained", is_flag=True)
def main(
    exp_directory,
    model,
    dataset,
    num_samples,
    check,
    check_epochs,
    use_default_hparams,
    hparams,
    egc_num_bases,
    egc_num_heads,
    final_runs,
    aggrs,
    hidden,
    seed_base,
    use_old_code_dataset,
    pretrained,
):
    exp_directory = Path(exp_directory).expanduser()
    if not exp_directory.exists():
        exp_directory.mkdir(parents=True)

    if dataset == "zinc":
        config = _zinc(
            model,
            num_samples,
            egc_num_bases,
            egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )
    elif dataset == "cifar":
        config = _cifar(
            model,
            num_samples,
            egc_num_bases,
            egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )
    elif dataset == "hiv":
        config = _mol(
            model,
            num_bases=egc_num_bases,
            num_heads=egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )
    elif dataset == "arxiv":
        config = _arxiv(
            model,
            num_bases=egc_num_bases,
            num_heads=egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )
    elif dataset == "code":
        config = _code(
            model,
            num_bases=egc_num_bases,
            num_heads=egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
            use_old_code_dataset=use_old_code_dataset,
        )

    elif dataset == "rmag":
        config = _rmag(
            model,
            num_bases=egc_num_bases,
            num_heads=egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )

    elif dataset == "mag":
        config = _mag(
            model,
            num_bases=egc_num_bases,
            num_heads=egc_num_heads,
            aggrs=aggrs,
            hidden=hidden,
        )

    else:
        raise ValueError

    if pretrained:
        model, opt, hparams, extra = config.pretrained(exp_directory)
        print(model)
        print(hparams)
        data = config.data([], hparams)
        print(config.test(model, data, extra))
        return

    if check:
        print(check_config(config, check_epochs))
        return

    dump_invocation_state(exp_directory)

    if "SLURM_CPUS_ON_NODE" in os.environ.keys():
        num_cpus = int(os.getenv("SLURM_CPUS_ON_NODE"))
    else:
        num_cpus = None

    ray.init(num_cpus=num_cpus)
    if hparams is not None:
        best_hparams = eval(hparams)
        print("Using given hyperparams:", best_hparams)
    elif use_default_hparams:
        best_hparams = {k: v.default() for k, v in config.hyperparams().items()}
        print("Using default hyperparams: ", best_hparams)

    else:
        best_hparams = run_search(config, exp_directory)
        print("Best hparams: ", best_hparams)

    train_final_models(
        config,
        best_hparams,
        exp_directory,
        override_repeats=final_runs,
        seed_base=seed_base,
    )


if __name__ == "__main__":
    main()
