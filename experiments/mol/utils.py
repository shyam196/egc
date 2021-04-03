from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader


def mol_data(root, dataset, batch_size=32, num_workers=4):
    dataset = PygGraphPropPredDataset(name=f"ogbg-mol{dataset}", root=root)
    split_idx = dataset.get_idx_split()
    loaders = dict()
    for split in ["train", "valid", "test"]:
        loaders[split] = DataLoader(
            dataset[split_idx[split]],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders
