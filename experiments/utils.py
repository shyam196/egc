import os
import random
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from torch_sparse import SparseTensor


def seed_all(seed):
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DATA_LOC_KEY = "DATASET_LOC"


def data_location():
    if DATA_LOC_KEY in os.environ.keys():
        return os.getenv(DATA_LOC_KEY)
    else:
        return str(Path.home() / "datasets")


def mlp(layers, act=nn.ReLU, dropout=0.0):
    modules = []
    for i, last in enumerate(layers[:-2]):
        current = layers[i + 1]
        modules.append(nn.Linear(last, current))
        modules.append(nn.BatchNorm1d(current))
        modules.append(act())
        modules.append(nn.Dropout(dropout))

    modules.append(nn.Linear(layers[-2], layers[-1]))
    return nn.Sequential(*modules)


def print_model_parameters(model, full=False):
    cnt = 0
    for k, v in model.named_parameters():
        if full:
            print(k, v.numel())
        cnt += v.numel()
    print("Total Params:", cnt)


def download(url: str, dest_file: Path):
    print(f"Downloading from {url}")
    if not dest_file.parent.exists():
        dest_file.parent.mkdir(parents=True)

    r = requests.get(url, stream=True)
    if r.ok:
        with open(dest_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        raise ValueError("Failed to download file")


def load_pretrained(conf, dataset_name, model_name, hidden, model_dir, pretrained_conf):
    req_hidden, url = pretrained_conf[model_name]
    if hidden != req_hidden:
        raise ValueError

    model_dir = model_dir / dataset_name / model_name
    model_path = model_dir / "checkpoint.pt"
    if not model_path.exists():
        download(url, model_path)

    return conf.restore_trial(model_dir, map_location=torch.device("cpu"))


class ToSparseTensor:
    """Additional checks over the PyG version"""

    def __init__(self, remove_edge_index: bool = True, fill_cache: bool = True):
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache

    def __call__(self, data):
        assert data.edge_index is not None

        (row, col), N, E = data.edge_index, data.num_nodes, data.num_edges
        perm = (col * N + row).argsort()
        row, col = row[perm], col[perm]

        if self.remove_edge_index:
            data.edge_index = None

        # we aren't interested in tracking edge attributes, so drop
        value = None

        for key, item in data:
            # additional check
            if hasattr(item, "size") and item.size(0) == E:
                data[key] = item[perm]

        data.adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(N, N), is_sorted=True
        )

        if self.fill_cache:  # Pre-process some important attributes.
            data.adj_t.storage.rowptr()
            data.adj_t.storage.csr2csc()

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"
