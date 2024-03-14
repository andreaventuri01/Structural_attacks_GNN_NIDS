from collections import defaultdict
from typing import Any, Iterable, List, Optional, Tuple, Union
import pandas as pd

import scipy.sparse
import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack, to_dlpack

import torch_geometric
from torch_geometric.utils.num_nodes import maybe_num_nodes

import networkx as nx
from dgl import from_networkx
from dgl.transforms import LineGraph

def from_dgl(
    g: Any,
) -> Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']:
    r"""Converts a :obj:`dgl` graph object to a
    :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance.

    Args:
        g (dgl.DGLGraph): The :obj:`dgl` graph object.

    Example:

        >>> g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
        >>> g.ndata['x'] = torch.randn(g.num_nodes(), 3)
        >>> g.edata['edge_attr'] = torch.randn(g.num_edges(), 2)
        >>> data = from_dgl(g)
        >>> data
        Data(x=[6, 3], edge_attr=[4, 2], edge_index=[2, 4])

        >>> g = dgl.heterograph({
        >>> g = dgl.heterograph({
        ...     ('author', 'writes', 'paper'): ([0, 1, 1, 2, 3, 3, 4],
        ...                                     [0, 0, 1, 1, 1, 2, 2])})
        >>> g.nodes['author'].data['x'] = torch.randn(5, 3)
        >>> g.nodes['paper'].data['x'] = torch.randn(5, 3)
        >>> data = from_dgl(g)
        >>> data
        HeteroData(
        author={ x=[5, 3] },
        paper={ x=[3, 3] },
        (author, writes, paper)={ edge_index=[2, 7] }
        )
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")

    if g.is_homogeneous:
        with torch.no_grad():
            data = Data()
            data.edge_index = torch.stack(g.edges(), dim=0)

            for attr, value in g.ndata.items():
                data[attr] = value
            for attr, value in g.edata.items():
                data[attr] = value

            return data

    data = HeteroData()

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    return data

def get_batch(dataset, sampling_size = 1000, benign_ratio = 10):
    """Function to get the batches for training the LineGraphSAGE algorithms. Given a dataset it yields a batch formed by selecting sampling_size malicious
    and sampling_size * benign_ratio benigns.

    Args:
        dataset (pd.DataFrame): Original dataset to be separated
        sampling_size (int, optional): Number of malicious samples in each batch. Defaults to 1000.
        benign_ratio (int, optional): Ben : Mal ratio in each batch. Defaults to 10.

    Yields:
        pd.DataFrame: batch
    """
    remaining_mal = dataset[dataset.Label == 1].copy()
    remaining_ben = dataset[dataset.Label == 0].copy()
    
    while len(remaining_mal) > sampling_size:
        mal_batch = remaining_mal.sample(sampling_size)
        if len(remaining_ben) >= sampling_size * benign_ratio:
            ben_batch = remaining_ben.sample(sampling_size * benign_ratio)
        else:
            ben_batch = remaining_ben.copy() # This is for safety reasons, we shouldn't get here
        
        remaining_mal.drop(mal_batch.index, inplace=True)
        if len(remaining_ben) >= sampling_size * benign_ratio: # This is for safety reasons, we shouldn't get here
            remaining_ben.drop(ben_batch.index, inplace=True)
        
        batch_data = pd.concat([mal_batch, ben_batch]).sample(frac=1)
        yield batch_data
    else:
        mal_batch = remaining_mal.copy()
        if len(remaining_ben) >= len(mal_batch) * benign_ratio:
            ben_batch = remaining_ben.sample(len(mal_batch) * benign_ratio)
        else:
            ben_batch = remaining_ben.copy() # This is for safety reasons, we shouldn't get here
        batch_data = pd.concat([mal_batch, ben_batch]).sample(frac=1)
        yield batch_data
        
def get_batch_test(dataset, size = 11000):
    remaining_dset = dataset.copy()
    while len(remaining_dset) > 0:
        if len(remaining_dset) > size:
            batch_data = remaining_dset.sample(size)
            remaining_dset.drop(batch_data.index, inplace=True)
        else:
            batch_data = remaining_dset.copy()
            remaining_dset.drop(batch_data.index, inplace=True)
        yield batch_data


def create_graph(batch):
    
    '''
    Networkx Graph

    train_data    : pandas dataframe
    IPV4_SRC_ADDR : source node column name format "IPAddr:Port"
    IPV4_DST_ADDR : dst node column name format "IPAddr:Port"
    ['h', 'Label']: edge attributes. h --> normalized columns, Label = flow label
    MultiGraph()  : undirected graph that can store multiedges (multiple edges between two nodes)
    '''
    G = nx.from_pandas_edgelist(batch, "Src IP", "Dst IP", ['h', 'Label'],  create_using=nx.MultiGraph())
    G = G.to_directed()
    # print(f"Number of nodes in train: {len(list(G.nodes))}")
    # print(f"Number of edges in train: {len(list(G.edges))}")
    
    G = from_networkx(G, edge_attrs=['h', 'Label'])
    
    transform = LineGraph()
    G = transform(G)

    # print(f"Number of nodes in train line graph: {G.number_of_nodes()}")
    # print(f"Number of edges in train line graph: {G.number_of_edges()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Device: {device}")
    G = G.to(device)
    G_torch = from_dgl(G)
    return G_torch