import pandas as pd

from dgl import from_networkx

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx

from EGraphSAGE import EGraphSAGE
from sklearn.metrics import classification_report

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
    if "ToN_IoT" in scaler_file:
        cols_to_norm = list(test_data.columns)
        cols_to_norm.remove('Src IP')
        cols_to_norm.remove('Dst IP')
        cols_to_norm.remove('Label')
    else:
        cols_to_norm = ['Dur','TotPkts','TotBytes','SrcBytes','BytesPerPkt','PktsPerSec','RatioOutIn', 'DstBytes']
    test_data[cols_to_norm] = scaler.transform(test_data[cols_to_norm])
    
    feature_cols = list(test_data.columns)
    feature_cols.remove("Label")
    feature_cols.remove("Src IP")
    feature_cols.remove("Dst IP")

    test_data['h'] = test_data[feature_cols].values.tolist()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    G_test = nx.from_pandas_edgelist(test_data, "Src IP", "Dst IP", ['h', 'Label'], create_using=nx.MultiGraph())
    G_test = G_test.to_directed()
    G_test = from_networkx(G_test, edge_attrs=['h', 'Label'])
    actual = G_test.edata.pop('Label')
    G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G_test.edata['h'].shape[1])

    G_test.ndata['feature'] = th.reshape(G_test.ndata['feature'], 
                                        (G_test.ndata['feature'].shape[0],
                                        1,
                                        G_test.ndata['feature'].shape[1]))
    G_test.edata['h'] = th.reshape(G_test.edata['h'],
                                (G_test.edata['h'].shape[0],
                                    1,
                                    G_test.edata['h'].shape[1]))

    G_test = G_test.to(device)

    start_time = timeit.default_timer()
    node_features_test = G_test.ndata['feature']
    edge_features_test = G_test.edata['h']
    test_pred = model(G_test, node_features_test, edge_features_test).cuda()
    elapsed = timeit.default_timer() - start_time

    # print(f"Prediction elapsed time: {elapsed} seconds")

    test_pred = test_pred.argmax(1)
    test_pred = th.Tensor.cpu(test_pred).detach().numpy()

    actual_label = ["Normal" if i == 0 else "Attack" for i in actual]
    test_pred_label = ["Normal" if i == 0 else "Attack" for i in test_pred]

    report = classification_report(actual, test_pred, digits=3, output_dict=True)
    report = pd.DataFrame(report).transpose()
    return report