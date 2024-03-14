import pandas as pd

from dgl import from_networkx

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx
import joblib

from EGraphSAGE import EGraphSAGE
from sklearn.metrics import classification_report

from sklearn import preprocessing
from test_egraphsage import test_model

import timeit
import argparse
import os
import sys
from tqdm.auto import tqdm

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

parser = argparse.ArgumentParser(description="Train E-GraphSAGE model")
parser.add_argument('--train', required=True, help="CSV file with train data", dest="train_path")
parser.add_argument('--test', required=True, help="CSV file with test data", dest="test_path")
parser.add_argument('--savedir', default="./results/", dest="save_dir", help="Directory for saving model and results")

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
save_dir += f"models/egraphsage/{mal_name}/"
save_results_dir += f"scores/egraphsage/{mal_name}/"

print(save_dir)

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

print("Normalizing numerical features...")
scaler = preprocessing.MinMaxScaler()
cols_to_norm = None
if 'ToN_IoT' in train_path:
    cols_to_norm = list(train_data.columns)
    cols_to_norm.remove('Src IP')
    cols_to_norm.remove('Dst IP')
    cols_to_norm.remove('Label')
else:
    cols_to_norm = ['Dur','TotPkts','TotBytes','SrcBytes','BytesPerPkt','PktsPerSec','RatioOutIn', 'DstBytes']

train_data[cols_to_norm] = scaler.fit_transform(train_data[cols_to_norm])
joblib.dump(scaler, save_dir + "scaler.skl")

feature_cols = list(train_data.columns)
feature_cols.remove("Label")
feature_cols.remove("Src IP")
feature_cols.remove("Dst IP")

if mal_name == 'neris':
    mal = train_data[train_data.Label == 1]
    ben = train_data[train_data.Label == 0]
    
    mal = mal.sample(30000)
    ben = ben.sample(len(mal) * 10)
    train_data = pd.concat([mal, ben]).sample(frac=1)

# print(feature_cols)
# sys.exit()
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

print(f"Number of nodes in train: {len(list(G.nodes))}")
print(f"Number of edges in train: {len(list(G.edges))}")

'''
DGL Graph
'''
G = from_networkx(G, edge_attrs=['h', 'Label'])

'''
Node's feature named 'h' : assign 1 dimensional feature vector to each node
Size : Number of Nodes x Number of Netflow Features
'''
G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])

'''
New Shape: Number of Nodes x 1 x Number of Netflow Features
'''
G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1, G.ndata['h'].shape[1]))

G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1, G.edata['h'].shape[1]))

'''
Edges flagged as training (all edges in train set)
'''
G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f"Device: {device}")
G = G.to(device)

criterion = nn.CrossEntropyLoss()

node_features = G.ndata['h']
edge_features = G.edata['h']

edge_label = G.edata['Label']
train_mask = G.edata['train_mask']

# (ndim_in, ndim_out, edim, activation, dropout)
# 'ndim_in'     :   8
# 'ndim_out'    :   128
# 'edim'        :   8
# 'activation'  :   ReLu
# 'dropout'     :   0.2
model = EGraphSAGE(G.ndata['h'].shape[2], 128, G.ndata['h'].shape[2], F.relu, 0.2).cuda()
opt = th.optim.Adam(model.parameters())

print("\n\n-----------TRAINING-------------")

pbar = tqdm(range(1, 201))
for epoch in range(1, 201):
    pred = model(G, node_features, edge_features).cuda()
    # entropy loss between input and target
    loss = criterion(pred[train_mask], edge_label[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    pbar.update(1)
    
    if epoch % 10 == 0:
        pbar.write(f'Epoch: {epoch}')
        pbar.write(f'Training acc: {compute_accuracy(pred[train_mask], edge_label[train_mask])}')
        
th.save(model.state_dict(), save_dict_path)
th.save(model, save_model_path)