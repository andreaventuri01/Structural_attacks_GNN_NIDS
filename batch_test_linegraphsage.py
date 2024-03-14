import pandas as pd

from dgl import from_networkx

import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx

from dgl.transforms import LineGraph
from sklearn.metrics import classification_report
from utils import from_dgl, create_graph, get_batch, get_batch_test

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
    
    actuals = []
    predictions = []
    batch_number = 1
    with torch.no_grad():
        for test_batch_data in get_batch_test(test_data):
            model.eval()
            print(f'Batch number {batch_number}')
            batch_number += 1
            
            G_torch_test = create_graph(test_batch_data)

            pred_test = model(G_torch_test.h, G_torch_test.edge_index)
            pred_test = pred_test.argmax(1)
            pred_test = th.Tensor.cpu(pred_test).detach().numpy().tolist()
            predictions += pred_test 

            actual = G_torch_test.Label.cpu().tolist()
            actuals += actual

        print(len(predictions))
        print(len(actuals))

    report = classification_report(actuals, predictions, digits=3, output_dict=True)
    report = pd.DataFrame(report).transpose()
    return report