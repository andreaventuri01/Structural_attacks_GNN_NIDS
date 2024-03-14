import pandas as pd

from dgl import from_networkx

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx

from dgl.transforms import LineGraph
from sklearn.metrics import classification_report
from utils import from_dgl

import timeit
import argparse
import os
import sys
from tqdm.auto import tqdm
from sklearn import preprocessing
import joblib

def test_model(test_data, model, scaler_file):
    scaler = joblib.load(scaler_file)
    cols_to_norm = None
    if 'ToN_IoT' in scaler_file:
        cols_to_norm = list(test_data.columns)
        cols_to_norm.remove('Src IP')
        cols_to_norm.remove('Dst IP')
        cols_to_norm.remove('Label')
        # cols_to_norm = list(set(test_data.columns) - set(['Src IP', 'Dst IP', 'Label']))
    else:
        cols_to_norm = ['Dur','TotPkts','TotBytes','SrcBytes','BytesPerPkt','PktsPerSec','RatioOutIn', 'DstBytes']
    # print(data[cols_to_norm])
    test_data[cols_to_norm] = scaler.transform(test_data[cols_to_norm])

    feature_cols = list(test_data.columns)
    feature_cols.remove("Label")
    feature_cols.remove("Src IP")
    feature_cols.remove("Dst IP")

    test_data['h'] = test_data[feature_cols].values.tolist()
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    G_test = nx.from_pandas_edgelist(test_data, "Src IP", "Dst IP", ['h', 'Label'],  create_using=nx.MultiGraph())
    G_test = G_test.to_directed()
    # print(f"Number of nodes in test flow graph: {len(list(G_test.nodes))}")
    # print(f"Number of edges in test flow graph: {len(list(G_test.edges))}")
    
    G_test = from_networkx(G_test, edge_attrs=['h', 'Label'])
    transform = LineGraph()
    new_G_test = transform(G_test)
    
    # print(f"Number of nodes in test line graph {new_G_test.number_of_nodes()}")
    # print(f"Number of edges in test line graph {new_G_test.number_of_edges()}")
    
    new_G_test = new_G_test.to(device)
    G_torch_test = from_dgl(new_G_test)
    
    pred_test = model(G_torch_test.h, G_torch_test.edge_index)
    pred_test = pred_test.argmax(1)
    pred_test = th.Tensor.cpu(pred_test).detach().numpy()
    
    actual = G_torch_test.Label.cpu()
    
    report = classification_report(actual, pred_test, digits=3, output_dict=True)
    report = pd.DataFrame(report).transpose()
    return report