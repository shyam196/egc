# Much of the code in this file is borrowed from the OGB repo and examples.
import numpy as np
import torch
import torch_geometric.transforms as transforms
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

from experiments.utils import ToSparseTensor


VOCAB_SIZE = 5000
SEQ_LEN = 5
NUM_NODETYPES = 98
NUM_NODEATTRIBUTES_1 = 10003
NUM_NODEATTRIBUTES_2 = 10030
MAX_DEPTH = 20


def decode_arr_to_seq(arr, idx2vocab):
    """Taken from OGB repo"""
    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1)

    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]  # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


def get_vocab_mapping(seq_list, num_vocab):
    """Taken from OGB repo"""

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind="stable")[:num_vocab]

    # print("Coverage of top {} vocabulary:".format(num_vocab))
    # print(float(np.sum(cnt_list[topvocab])) / np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx["__UNK__"] = num_vocab
    idx2vocab.append("__UNK__")

    vocab2idx["__EOS__"] = num_vocab + 1
    idx2vocab.append("__EOS__")

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert idx == vocab2idx[vocab]

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert vocab2idx["__EOS__"] == len(idx2vocab) - 1

    return vocab2idx, idx2vocab


def augment_edge(data):
    """Taken from OGB Repo"""

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [
            torch.zeros(edge_index_ast_inverse.size(1), 1),
            torch.ones(edge_index_ast_inverse.size(1), 1),
        ],
        dim=1,
    )

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(
        data.node_is_attributed.view(-1,) == 1
    )[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack(
        [attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]],
        dim=0,
    )
    edge_attr_nextoken = torch.cat(
        [
            torch.ones(edge_index_nextoken.size(1), 1),
            torch.zeros(edge_index_nextoken.size(1), 1),
        ],
        dim=1,
    )

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack(
        [edge_index_nextoken[1], edge_index_nextoken[0]], dim=0
    )
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat(
        [
            edge_index_ast,
            edge_index_ast_inverse,
            edge_index_nextoken,
            edge_index_nextoken_inverse,
        ],
        dim=1,
    )
    data.edge_attr = torch.cat(
        [
            edge_attr_ast,
            edge_attr_ast_inverse,
            edge_attr_nextoken,
            edge_attr_nextoken_inverse,
        ],
        dim=0,
    )

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    """Taken from OGB Repo"""

    augmented_seq = seq[:max_seq_len] + ["__EOS__"] * max(0, max_seq_len - len(seq))
    return torch.tensor(
        [
            [
                vocab2idx[w] if w in vocab2idx else vocab2idx["__UNK__"]
                for w in augmented_seq
            ]
        ],
        dtype=torch.long,
    )


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    """Taken from OGB repo"""

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def code_data(
    root,
    batch_size=128,
    num_vocab=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    use_old_code_dataset=False,
):
    dataset = PygGraphPropPredDataset(
        "ogbg-code" if use_old_code_dataset else "ogbg-code2", root=root
    )
    split_idx = dataset.get_idx_split()
    vocab2idx, idx2vocab = get_vocab_mapping(
        [dataset.data.y[i] for i in split_idx["train"]], num_vocab
    )
    ts = [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, seq_len)]
    dataset.transform = transforms.Compose(ts)

    loaders = dict()
    for split in ["train", "valid", "test"]:
        loaders[split] = DataLoader(
            dataset[split_idx[split]],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
        )
    return loaders, idx2vocab
