import pandas as pd

from dgl import from_networkx

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx

from EGraphSAGE import EGraphSAGE
from sklearn.metrics import classification_report
from test_egraphsage import test_model
from adversarial_feature import adversarial_feature_attack
from adversarial_structure import aa_add_edge_C2X, aa_add_nodes

import timeit
import argparse
import os
import sys
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Test E-GraphSAGE model")
# Naming convention: <attack>_test.csv
parser.add_argument('--test', required=True, help="CSV file with test baseline data", dest="test_path")
parser.add_argument('--savedir', default="./results/", dest="save_dir", help="Directory for saving results")

group = parser.add_mutually_exclusive_group()
group.add_argument('--feature_attack', action='store_true', dest='feature_attack')
group.add_argument('--structure_attack', choices = ['benign_from_C', 'benign_from_each_mal', 'malicious_from_C', 'add_node'], dest='structure_attack')

# args = parser.parse_args(['--test', 'preprocessed_data/ToN_IoT/ddos_test.csv', '--feature_attack'])
args = parser.parse_args()
    
baseline = True
feature_attack = False
structure_attack = False
atk_type = None

if args.feature_attack:
    feature_attack = True
    baseline = False

if args.structure_attack is not None:
    structure_attack = True
    atk_type = args.structure_attack
    baseline = False

test_path = None
if os.path.exists(args.test_path) and os.path.isfile(args.test_path):
    test_path = args.test_path
else:
    print("Error with test data path. File may not exist")
    sys.exit()
    
SEED = 20230515

ton_iot = False
save_dir = args.save_dir
if not save_dir.endswith("/"):
    save_dir += "/"
if 'ToN_IoT' in test_path:
    save_dir += "ToN_IoT/"
    ton_iot = True
else:
    save_dir += "CTU/"

save_results_dir = save_dir
mal_name = os.path.basename(test_path).split("_")[0]
save_results_dir += f"scores/egraphsage/{mal_name}/"
scaler_file = save_dir + f"models/egraphsage/{mal_name}/scaler.skl"

if not os.path.exists(save_results_dir):
    os.makedirs(save_results_dir)
    
save_results_baseline_file = save_results_dir + "baseline.csv"

test_data = pd.read_csv(test_path)

model_file = save_dir + f"models/egraphsage/{mal_name}/model.pth"
model = None
if not (os.path.exists(model_file) and os.path.isfile(model_file)):
    print("Error with trained model file. Make sure it exists before testing")
    sys.exit()
    
model = th.load(model_file)
if baseline and not feature_attack and not structure_attack:
    print("--------------------------------------------------------")
    print("----------------- Baseline Test ------------------------")
    print("--------------------------------------------------------")
    report = test_model(test_data, model, scaler_file)
    print(report)
    report.to_csv(save_results_baseline_file)
    
if not baseline and feature_attack:
    print("--------------------------------------------------------")
    print("------------Adversarial Attack Feature------------------")
    print("--------------------------------------------------------")
    
    recall_df, precision_df, f1_df = adversarial_feature_attack(test_data,
                                                                model,
                                                                scaler_file,
                                                                test_model=test_model,
                                                                ton_iot=ton_iot)
    
    recall_df.to_csv(save_results_dir + "aa_feature_recall.csv")
    precision_df.to_csv(save_results_dir + "aa_feature_precision.csv")
    f1_df.to_csv(save_results_dir + "aa_feature_f1.csv")
    print("------------ Recall DF --------------")
    print(recall_df)
    print("\n------------ Precision --------------")
    print(precision_df)
    print("\n------------ F1-Score ---------------")
    print(f1_df)

if not baseline and structure_attack:
    print("--------------------------------------------------------")
    print("------------Adversarial Attack Structure----------------")
    print("--------------------------------------------------------")
    
    results_df = None
    if atk_type == "add_node":
        results_df = aa_add_nodes(test_data, model, scaler_file,
                               test_model=test_model)
    else:
        results_df = aa_add_edge_C2X(test_data, model, scaler_file, 
                                    test_model=test_model,
                                    attack=atk_type)
    
    print(results_df)
    results_df.to_csv(save_results_dir + f'aa_structure_{atk_type}.csv')