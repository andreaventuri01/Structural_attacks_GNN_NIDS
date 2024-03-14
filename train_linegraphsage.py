# DEPRECATED --> USE batch_train_linegraphsage.py


import pandas as pd

from dgl import from_networkx
# from dgl.transforms import LineGraph
from torch_geometric.transforms import LineGraph

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx
import joblib

from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import NeighborLoader
from utils import from_dgl

from sklearn.metrics import classification_report

from sklearn import preprocessing
from test_linegraphsage import test_model

import timeit
import argparse
import os
import sys
from tqdm.auto import tqdm

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

parser = argparse.ArgumentParser(description="Train E-GraphSAGE model")
parser.add_argument('--train', required=True,
                    help="CSV file with train data",
                    dest="train_path")
parser.add_argument('--test', required=True,
                    help="CSV file with test data",
                    dest="test_path")
parser.add_argument('--savedir', default="./results/",
                    dest="save_dir", 
                    help="Directory for saving model and results")

# args = parser.parse_args(['--train', 'preprocessed_data/ToN_IoT/ddos_train.csv',
#                           '--test', 'preprocessed_data/ToN_IoT/ddos_test.csv'])
args = parser.parse_args()

train_path = None
if os.path.exists(args.train_path) and os.path.isfile(args.train_path):
    train_path = args.train_path
else:
    print("Error with train data path. File may not exist")
    sys.exit()
    
test_path = None
if os.path.exists(args.test_path) and os.path.isfile(args.test_path):
    test_path = args.test_path
else:
    print("Error with test data path. File may not exist")
    sys.exit()
    
SEED = 20230515

save_dir = args.save_dir
if not save_dir.endswith("/"):
    save_dir += "/"
    
if 'ToN_IoT' in train_path:
    save_dir += 'ToN_IoT/'
else:
    save_dir += 'CTU/'
    
save_results_dir = save_dir
mal_name = os.path.basename(train_path).split("_")[0]
save_dir += f"models/linegraphsage/{mal_name}/"
save_results_dir += f"scores/linegraphsage/{mal_name}/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(save_results_dir):
    os.makedirs(save_results_dir)
    
save_dict_path = save_dir + "dict.pth"
save_model_path = save_dir + "model.pth"
save_results_path = save_results_dir + "test_scores_after_train.csv"

print("Reading train and test data...")
train_data = pd.read_csv(train_path) # data is already 20:1 ben:mal
test_data = pd.read_csv(test_path) # data is already 20:1 ben:mal

seed = 20230806
ben = train_data[train_data.Label == 0]
mal = train_data[train_data.Label == 1]

# Questo codice l'ho inserito perchè un dataset troppo grande fa andare la GPU "out-of-memory".
# CONSIGLIO: USA UN DATASET CON POCHI MALEVOLI (A MENO CHE NON SI HA A DISPOSIZIONE UNA GPU CON TANTA MEMORIA!!)
if len(mal) > 1000:
    mal = mal.sample(1000, random_state=seed)
    ben = ben.sample(len(mal) * 10, random_state=seed)
    train_data = pd.concat([ben, mal]).sample(frac=1, random_state=seed)
    


print(f"New ben:mal ratio = {len(ben) / len(mal)}")


print("Normalizing numerical features...")
scaler = preprocessing.MinMaxScaler()
cols_to_norm = None
if 'ToN_IoT' in train_path:
    cols_to_norm = list(train_data.columns)
    cols_to_norm.remove('Src IP')
    cols_to_norm.remove('Dst IP')
    cols_to_norm.remove('Label')
    # cols_to_norm = list(set(train_data.columns) - set(['Src IP', 'Dst IP', 'Label']))
else:
    cols_to_norm = ['Dur','TotPkts','TotBytes','SrcBytes','BytesPerPkt','PktsPerSec','RatioOutIn', 'DstBytes']
# print(data[cols_to_norm])
train_data[cols_to_norm] = scaler.fit_transform(train_data[cols_to_norm])
joblib.dump(scaler, save_dir + "scaler.skl")

feature_cols = list(train_data.columns)
feature_cols.remove("Label")
feature_cols.remove("Src IP")
feature_cols.remove("Dst IP")

train_data['h'] = train_data[feature_cols].values.tolist()

'''
Networkx Graph

train_data    : pandas dataframe
IPV4_SRC_ADDR : source node column name format "IPAddr:Port"
IPV4_DST_ADDR : dst node column name format "IPAddr:Port"
['h', 'Label']: edge attributes. h --> normalized columns, Label = flow label
MultiGraph()  : undirected graph that can store multiedges (multiple edges between two nodes)
'''
G = nx.from_pandas_edgelist(train_data, "Src IP", "Dst IP", ['h', 'Label'],  create_using=nx.MultiGraph())
G = G.to_directed()
print(f"Number of nodes in train flow graph: {len(list(G.nodes))}")
print(f"Number of edges in train flow graph: {len(list(G.edges))}")

G = from_networkx(G, edge_attrs=['h', 'Label'])

# transform = LineGraph()
# new_G = transform(G)

print(f"Number of nodes in train line graph {G.number_of_nodes()}")
print(f"Number of edges in train line graph {G.number_of_edges()}")

G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f"Device: {device}")
G = G.to(device)

model = GraphSAGE(
    in_channels = G.edata['h'].shape[1],
    hidden_channels = 64,
    out_channels = 2,
    num_layers = 2,
    dropout=0.2    
).cuda()

criterion = nn.CrossEntropyLoss()
opt = th.optim.Adam(model.parameters())

G_torch = from_dgl(G)

train_loader = NeighborLoader(
    data = G_torch,
    num_neighbors=[5, 5],
    batch_size=16,
    transform=LineGraph()
    )

print("\n\n-----------TRAINING-------------")
pbar = tqdm(range(1, 5))
for epoch in range(1, 5):
    
    model.train()
    train_loss = 0
    train_acc = 0
    acc = 0
    
    for batch in train_loader:
        pred = model(batch.h, batch.edge_index)
        loss = criterion(pred, batch.Label)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    pbar.update(1)

    if epoch % 1 == 0:
        pbar.write(f"Epoch: {epoch}")
        pbar.write(f'Training acc: {compute_accuracy(pred, batch.Label)}')
    
th.save(model.state_dict(), save_dict_path)
th.save(model, save_model_path)

# report = test_model(test_data, model, save_dir+"scaler.skl")
# print(report)