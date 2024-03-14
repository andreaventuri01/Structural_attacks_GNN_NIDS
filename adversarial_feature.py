import pandas as pd
import numpy as np
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx
from tqdm.auto import tqdm

pd.options.mode.chained_assignment = None  # default='warn'

#Vocabulary storing how features should be modified at each step
steps = {
    1 : {
        'Dur' : 1, 'SrcBytes' : 1, 'DstBytes' : 1, 'TotPkts' : 1 },
    2 : {
        'Dur' : 2, 'SrcBytes' : 2, 'DstBytes' : 2, 'TotPkts' : 2 },
    3 : {
        'Dur' : 5, 'SrcBytes' : 8, 'DstBytes' : 8, 'TotPkts' : 5 },
    4 : {
        'Dur' : 10, 'SrcBytes' : 16, 'DstBytes' : 16, 'TotPkts' : 10 },
    5 : {
        'Dur' : 15, 'SrcBytes' : 64, 'DstBytes' : 64, 'TotPkts' : 15 },
    6 : {
        'Dur' : 30, 'SrcBytes' : 128, 'DstBytes' : 128, 'TotPkts' : 20 },
    7 : {
        'Dur' : 45, 'SrcBytes' : 256, 'DstBytes' : 256, 'TotPkts' : 30 },
    8 : {
        'Dur' : 60, 'SrcBytes' : 512, 'DstBytes' : 512, 'TotPkts' : 50 },
    9 : {
        'Dur' : 120, 'SrcBytes' : 1024, 'DstBytes' : 1024, 'TotPkts' : 100 }
}

#Groups of altered features
altered_features = {
    '1a' : ['Dur'],
    '1b' : ['SrcBytes'],
    '1c' : ['DstBytes'],
    '1d' : ['TotPkts'],
    '2a' : ['Dur', 'SrcBytes'],
    '2b' : ['Dur', 'DstBytes'],
    '2c' : ['Dur', 'TotPkts'],
    '2d' : ['SrcBytes', 'TotPkts'],
    '2e' : ['SrcBytes', 'DstBytes'],
    '2f' : ['DstBytes', 'TotPkts'],
    '3a' : ['Dur', 'SrcBytes', 'DstBytes'],
    '3b' : ['Dur', 'SrcBytes', 'TotPkts'],
    '3c' : ['Dur', 'DstBytes', 'TotPkts'],
    '3d' : ['SrcBytes','DstBytes','TotPkts'],
    '4a' : ['Dur', 'SrcBytes', 'DstBytes', 'TotPkts']
}

#Function returning the modified dataset according to the list of features to alter
def get_malicious_modified(feature_key, malicious, step):
    for i in altered_features[feature_key]:
        modified = malicious[i] + steps[step][i]
        malicious.loc[:, i] = modified
    return malicious

def correct_derived_features(malicious, ton_iot=False):
    malicious.loc[:, 'TotBytes'] = malicious['SrcBytes'] + malicious['DstBytes']
    if not ton_iot: # ToN_IoT does not have these derived features
        malicious.loc[:, 'BytesPerPkt'] = malicious['TotBytes'] / malicious['TotPkts']
        malicious.loc[:, 'PktsPerSec'] = malicious['TotPkts'] / malicious['Dur']
        max_value = malicious.loc[malicious['PktsPerSec'] != np.inf, 'PktsPerSec'].max()
        malicious.loc[:, 'PktsPerSec'].replace(np.inf, max_value, inplace=True)
        malicious.loc[:, 'RatioOutIn'] = (malicious['DstBytes']) / malicious['SrcBytes']
        max_value = malicious.loc[malicious['RatioOutIn'] != np.inf, 'RatioOutIn'].max()
        malicious.loc[:, 'RatioOutIn'].replace(np.inf, max_value, inplace=True)
    return malicious

def adversarial_feature_attack(test_data, model, scaler_file, test_model, ton_iot = False):
    recall_dict = {}
    precision_dict = {}
    f1_dict = {}
    
    pbar = tqdm(range(len(altered_features.keys())*len(steps.keys())))
    for k in altered_features.keys():
        group_recalls = []
        group_precisions = []
        group_f1 = []
        for s in steps.keys():
            data = test_data.copy()
            data[data.Label == 1] = get_malicious_modified(k, data[data.Label == 1], s)
            data[data.Label == 1] = correct_derived_features(data[data.Label == 1], ton_iot)
            
            report = test_model(data, model, scaler_file)
            
            recall = report.loc['1', 'recall']
            precision = report.loc['1', 'precision']
            f1 = report.loc['1', 'f1-score']
            
            group_recalls.append(recall)
            group_precisions.append(precision)
            group_f1.append(f1)
            
            pbar.write(f"Group {k}, step {s}: {report.loc['1', 'recall']}")
            pbar.update(1)
        
        recall_dict[k] = group_recalls
        precision_dict[k] = group_precisions
        f1_dict[k] = group_f1
        
    recall_df = pd.DataFrame.from_dict(recall_dict)
    precision_df = pd.DataFrame.from_dict(precision_dict)
    f1_df = pd.DataFrame.from_dict(f1_dict)
    
    return recall_df.T, precision_df.T, f1_df.T