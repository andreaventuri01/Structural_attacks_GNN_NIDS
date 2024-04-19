import pandas as pd
import numpy as np
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx
from tqdm.auto import tqdm
import sys

# from test_egraphsage import test_model

from random import randint

def generate_random_node():
    ip = '.'.join(str(randint(0, 255)) for _ in range(4))
    port = str(randint(49152, 66536))
    return ':'.join([ip, port])

def generate_benign_from_new_nodes(data, num_nodes, amount):
    """Generate <num_nodes> new nodes and send them <amount> benign netflows from controlled hosts

    Args:
        data (pd.DataFrame): netflow dataframe
        num_nodes (int): number of new nodes to create
        amount (int): how many new benign flows to send to each new node
    """
    src_nodes = {x for x in data['Src IP'].to_list()}
    dst_nodes = {x for x in data['Dst IP'].to_list()}
    nodes = list(src_nodes.union(dst_nodes))
    
    new_nodes = []
    tries = 0
    max_try = 100
    while len(new_nodes) < num_nodes:
        new_node = generate_random_node()
        tries += 1
        if new_node not in nodes:
            new_nodes.append(new_node)
            tries = 0
        if tries > max_try:
            print("ERROR: Maximum tries for generating new nodes reached. Try with a smaller number of nodes")
            sys.exit()
            
    sampled_malicious = data[data.Label == 1].sample(num_nodes * amount, replace=True)
    sampled_ben = data[data.Label == 0].sample(num_nodes * amount, replace=True)
    
    new_nodes = new_nodes * amount # Repeating multiple Dst IPs for sending <amount> benign flows
    
    print(f'Generating {num_nodes} new nodes and sending them {amount} of benign flows each...')
    sampled_ben['Src IP'] = sampled_malicious['Src IP'].values
    sampled_ben['Dst IP'] = new_nodes
    
    # print(sampled_ben)
    return sampled_ben.copy()

def aa_add_nodes(test_data, model, scaler_file, test_model):
    amounts = [1, 2, 5, 10, 20]
    num_nodes = [1, 5, 10, 20, 50, 100, 200, 1000]
    
    results_list = []
    
    for n_nodes in num_nodes:
        mal_recall_scores = []
        mal_precision_scores = []
        mal_f1_scores = []
        
        ben_recall_scores = []
        ben_precision_scores = []
        ben_f1_scores = []
        
        for amount in amounts:
            
            new_X = generate_benign_from_new_nodes(test_data, n_nodes, amount)
            new_data = pd.concat([test_data, new_X], ignore_index=True)
            report = test_model(new_data, model, scaler_file)
            
            mal_recall_scores.append(report.loc['1', 'recall'])
            mal_precision_scores.append(report.loc['1', 'precision'])
            mal_f1_scores.append(report.loc['1', 'f1-score'])
            
            ben_recall_scores.append(report.loc['0', 'recall'])
            ben_precision_scores.append(report.loc['0', 'precision'])
            ben_f1_scores.append(report.loc['0', 'f1-score'])
        
        data_dict = {
            'Mal Recall': mal_recall_scores,
            'Mal Precision': mal_precision_scores,
            'Mal F1-Score': mal_f1_scores,
            'Ben Recall': ben_recall_scores,
            'Ben Precision': ben_precision_scores,
            'Ben F1-Score': ben_f1_scores
        }
        
        results_df = pd.DataFrame.from_dict(data_dict)
        results_df['Num Nodes'] = n_nodes
        results_df.index = amounts
        
        results_list.append(results_df)
    
    all_results = pd.concat(results_list)
    
    return all_results

def generate_malicious_from_C(data, amount):
    """For each Controlled node, add <amount> malicious samples to a random destination. 

    Args:
        data (pd.DataFrame): netflow dataframe
        amount (int): how many new benign flows for each malicious
    """
    benigns = data[data.Label == 0]
    malicious = data[data.Label == 1]
    sampled_mal = malicious.sample(len(malicious['Src IP'].unique()) * amount, replace=True)
    
    dst_ip = benigns.sample(len(sampled_mal), replace=True).loc[:, 'Dst IP'].to_list()
    sampled_mal["Dst IP"] = dst_ip
    return sampled_mal.copy()

def generate_malicious_from_C2(data, amount):
    """For each Controlled node, add <amount> malicious samples to a random destination. 

    Args:
        data (pd.DataFrame): netflow dataframe
        amount (int): how many new benign flows for each malicious
    """
    benigns = data[data.Label == 0]
    malicious = data[data.Label == 1]
    
    sampled_mal = malicious.sample(len(malicious['Src IP'].unique()) * amount, replace=True)
    src_ip = malicious['Src IP'].unique().tolist() * amount
    sampled_mal['Src IP'] = src_ip #This is to force to have <amount> malicious from each controlled node.
    
    dst_ip = benigns.sample(len(sampled_mal), replace=True).loc[:, 'Dst IP'].to_list()
    sampled_mal["Dst IP"] = dst_ip

    return sampled_mal.copy()

def generate_benign_from_C(data, amount):
    """For each Controlled node, add <amount> benign samples to a random destination. 

    Args:
        data (pd.DataFrame): netflow dataframe
        amount (int): how many new benign flows for each malicious
    """
    malicious = data[data.Label == 1]
    sampled_ben = data[data.Label == 0].sample(len(malicious['Src IP'].unique()) * amount, replace = True)
    src_ip = list(malicious['Src IP'].unique())

    print(f"Generating {len(src_ip) * amount} new benigns..")
    src_ip = src_ip * amount
    sampled_ben['Src IP'] = src_ip
    return sampled_ben.copy()

def aa_add_edge_C2X(test_data, model, scaler_file, test_model, attack):
    mal_recall_scores = []
    mal_precision_scores = []
    mal_f1_scores = []
    
    ben_recall_scores = []
    ben_precision_scores = []
    ben_f1_scores = []
    
    amounts = [1, 2, 5, 10, 20]
    
    gen_func = None
    if attack == 'benign_from_C':
        gen_func = generate_benign_from_C
    elif attack == 'malicious_from_C':
        gen_func = generate_malicious_from_C
    else:
        print("No correct attack specified.")
        print("Valid options are: ['benign_from_C', 'malicious_from_C']")
        sys.exit()
    
    for amount in amounts:
        
        new_X = gen_func(test_data, amount)
        new_data = pd.concat([test_data, new_X], ignore_index=True)
        
        report = test_model(new_data, model, scaler_file)
        
        mal_recall_scores.append(report.loc['1', 'recall'])
        mal_precision_scores.append(report.loc['1', 'precision'])
        mal_f1_scores.append(report.loc['1', 'f1-score'])
        
        ben_recall_scores.append(report.loc['0', 'recall'])
        ben_precision_scores.append(report.loc['0', 'precision'])
        ben_f1_scores.append(report.loc['0', 'f1-score'])
    
    data_dict = {
        'Mal Recall': mal_recall_scores,
        'Mal Precision': mal_precision_scores,
        'Mal F1-Score': mal_f1_scores,
        'Ben Recall': ben_recall_scores,
        'Ben Precision': ben_precision_scores,
        'Ben F1-Score': ben_f1_scores
    }
    
    results_df = pd.DataFrame.from_dict(data_dict)
    results_df.index = amounts
    # print(results_df)
    return results_df
    
    
if __name__ == '__main__':
    data = pd.read_csv("preprocessed_data/murlo_test.csv")
    
    # model = th.load("results/models/murlo/model.pth")
    # # generate_benign_from_C(data, 2)
    # scaler_file = "results/models/murlo/scaler.skl"
    # # generate_benign_from_new_nodes(data, 2, 2)
    # res = aa_add_nodes(data, model, scaler_file, test_model)
    # print(res)
    # aa_add_edge_C2X(data, model, scaler_file, attack="benign_from_C")
    # generate_malicious_from_C(data, 1)