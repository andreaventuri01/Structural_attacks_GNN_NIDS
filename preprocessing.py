import pandas as pd
import numpy as np
import re

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Preprocess CTU data for E-GraphSAGE algorithm")
parser.add_argument('--mal', required=True, help="CSV file for malicious netflows", dest="mal_data_path")
parser.add_argument('--ben', required=True, help="CSV file for benign netflows", dest="ben_data_path")
parser.add_argument('--savedir', default="./preprocessed_data/CTU/", dest="save_dir")

args = parser.parse_args()

mal_data_path = None
if os.path.exists(args.mal_data_path) and os.path.isfile(args.mal_data_path):
    mal_data_path = args.mal_data_path
else:
    print("Error with malicious data file. File may not exist")
    sys.exit()

ben_data_path = None
if os.path.exists(args.ben_data_path) and os.path.isfile(args.ben_data_path):
    ben_data_path = args.ben_data_path
else:
    print("Error with benign data file. File may not exist")
    sys.exit()

print("Reading CSV files...")
mal_data = pd.read_csv(mal_data_path)
ben_data = pd.read_csv(ben_data_path)

print("Generating 20:1 ben:mal dataset")
if len(ben_data) >= 20*len(mal_data):
    ben_data = ben_data.sample(20*len(mal_data))
else:
    mal_data = mal_data.sample(len(ben_data)//20)
    
print(f"Ben/Mal samples: {len(ben_data)/len(mal_data)}")

print("Generating saving directories...")
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_dir = args.save_dir
if not save_dir.endswith('/'):
    save_dir += "/"

basename = os.path.basename(mal_data_path).split('.')[0]
train_save_path = save_dir + f"{basename}_train.csv"
test_save_path = save_dir + f"{basename}_test.csv"

# Preprocessing CTU
print("Starting Preprocessing...")
data = pd.concat([mal_data, ben_data], ignore_index=True)

print("Dropping unused columns...")
data = data.drop(columns=["StartTime"])

data.drop(data[data.Proto != 'tcp'].index, inplace=True)
data['Proto'].replace({'tcp': 0}, inplace=True)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data['sTos'].replace(np.nan, 0, inplace=True)
data['dTos'].replace(np.nan, 0, inplace=True)

# Drop samples with NaN values
data = data.dropna(how='any', axis=0)

# Add Derived Features
print("Computing derived features...")
data['BytesPerPkt'] = data['TotBytes'] / data['TotPkts']

data['PktsPerSec'] = data['TotPkts'] / data['Dur']
max_value = data.loc[data['PktsPerSec'] != np.inf, 'PktsPerSec'].max()
data['PktsPerSec'].replace(np.inf, max_value, inplace=True)

data['RatioOutIn'] = (data['TotBytes'] - data['SrcBytes']) / data['SrcBytes']
max_value = data.loc[data['RatioOutIn'] != np.inf, 'RatioOutIn'].max()
data['RatioOutIn'].replace(np.inf, max_value, inplace=True)

data['DstBytes'] = data['TotBytes'] - data['SrcBytes']

# 95-th Percentile threshold
print("Filtering outliers...")
data.drop(data[data.Dur > 300].index, inplace=True)
data.drop(data[data.SrcBytes >= 60000].index, inplace=True)
data.drop(data[data.TotPkts >= 100].index, inplace=True)
data.drop(data[data.PktsPerSec >= 10000].index, inplace=True)

# Encoding Src/Dst ports
print("Encoding ports...")
data.Sport = data.Sport.astype(np.int64)
data.Dport = data.Dport.astype(np.int64)

data['SrcPort'] = np.where(data.Sport.between(0, 1023, inclusive='both'),
                      'SrcPortWellKnown',
                      np.where(data.Sport.between(1024, 49151, inclusive='both'),
                               'SrcPortRegistered', 'SrcPortPrivate'))

data['DstPort'] = np.where(data.Dport.between(0, 1023, inclusive='both'),
                      'DstPortWellKnown',
                      np.where(data.Dport.between(1024, 49151, inclusive='both'),
                               'DstPortRegistered', 'DstPortPrivate'))

data = pd.get_dummies(data, columns=["SrcPort", "DstPort"], prefix="", prefix_sep="")

# One Hot Encoding
new_cols = pd.get_dummies(data.Dir, prefix='Dir')
data[new_cols.columns] = new_cols
data = data.drop('Dir', axis=1)

print("One hot encoding states")
state_flags = ['P', 'A', 'S', 'C', 'F', 'U', 'R', 'E']
data.loc[:, "State_Split"] = data.State.str.split("_")
state_l = data["State_Split"].to_list()
src_state = []
dst_state = []
for i in state_l:
    src_state.append(i[0])
    dst_state.append(i[1])
    
data.loc[:, "SrcState"] = src_state
data.loc[:, "DstState"] = dst_state

for s in state_flags:
    data["SrcState_" + s] = data["SrcState"].apply(lambda x: 1 if s in x else 0)
    data["DstState_" + s] = data["DstState"].apply(lambda x: 1 if s in x else 0)
    
data = data.drop(columns=["State", "State_Split", "SrcState", "DstState"])

print("Encoding explicit IPAddresses...")
data['IPSrcType'] = np.where(data.SrcAddr.str.startswith("147.32."), 1, 0)
data['IPDstType'] = np.where(data.DstAddr.str.startswith("147.32."), 1, 0)

# Concatenate IP Addr and Port data for graph nodes
print("Generating Node identifiers...")
data['Src IP'] = data['SrcAddr'].astype(str) + ":" + data["Sport"].astype(str)
data['Dst IP'] = data['DstAddr'].astype(str) + ":" + data["Dport"].astype(str)
data.drop(columns=["Sport", "Dport", "SrcAddr", "DstAddr"], inplace=True)

ben_data = data[data.Label == 0]
mal_data = data[data.Label == 1]

print(f"Len Malicious after preproc: {len(mal_data)}")
print(f'Len Benign after preproc: {len(ben_data)}')

print("Generating 20:1 ben:mal dataset")
if len(ben_data) >= 20*len(mal_data):
    ben_data = ben_data.sample(20*len(mal_data))
else:
    mal_data = mal_data.sample(len(ben_data)//20)
    
train_mal = mal_data.sample(frac=0.75)
test_mal = mal_data.drop(train_mal.index)
train_ben = ben_data.sample(frac=0.75)
test_ben = ben_data.drop(train_ben.index)

print(f"Train set: Mal {len(train_mal)}, Ben {len(train_ben)}, Ratio {len(train_ben)/len(train_mal)}")
print(f"Test set: Mal {len(test_mal)}, Ben {len(test_ben)}, Ratio {len(test_ben)/len(test_mal)}")

train_data = pd.concat([train_mal, train_ben], ignore_index=True).sample(frac=1)
test_data = pd.concat([test_mal, test_ben], ignore_index=True).sample(frac=1)

print("Saving Train and Test sets...")
train_data.to_csv(train_save_path, index = None)
test_data.to_csv(test_save_path, index = None)