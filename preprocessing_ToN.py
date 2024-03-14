import pandas as pd
import numpy as np
import re

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Preprocess ToN-IoT data for E-GraphSAGE algorithm")
parser.add_argument('--mal', required=True, help="CSV file for malicious netflows", dest="mal_data_path")
parser.add_argument('--ben', required=True, help="CSV file for benign netflows", dest="ben_data_path")
parser.add_argument('--savedir', default="./preprocessed_data/ToN_IoT/", dest="save_dir")

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
mal_data = pd.read_csv(mal_data_path, index_col=0)
ben_data = pd.read_csv(ben_data_path, index_col=0)

print(f"Len Malicious: {len(mal_data)}")
print(f'Len benign: {len(ben_data)}')

print("Generating saving directories...")
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_dir = args.save_dir
if not save_dir.endswith('/'):
    save_dir += "/"

basename = os.path.basename(mal_data_path).split('.')[0]
train_save_path = save_dir + f"{basename}_train.csv"
test_save_path = save_dir + f"{basename}_test.csv"

print("Starting Preprocessing...")
data = pd.concat([mal_data, ben_data], ignore_index=True)

data = data.drop(columns=['ts'])

data = data[data.proto == 'tcp']
data['proto'] = 0

print("Encoding IP addresses and ports")
data['Src IP'] = data.src_ip.apply(str) + ':' + data.src_port.apply(str)
data['Dst IP'] = data.dst_ip.apply(str) + ':' + data.dst_port.apply(str)

data['IPSrcType'] = data['src_ip'].apply(str).apply(lambda x: 1 if 
                                                    (re.search('^10[.]', x) or 
                                                     re.search('^172[.][1-3][678901][.]', x) or
                                                     re.search('^192[.]168[.]', x)) else 0)
data['IPDstType'] = data['dst_ip'].apply(str).apply(lambda x: 1 if 
                                                    (re.search('^10[.]', x) or 
                                                     re.search('^172[.][1-3][678901][.]', x) or
                                                     re.search('^192[.]168[.]', x)) else 0)

data['SrcPort'] = np.where(data.src_port.between(0, 1023, inclusive='both'),
                      'SrcPortWellKnown',
                      np.where(data.src_port.between(1024, 49151, inclusive='both'),
                               'SrcPortRegistered', 'SrcPortPrivate'))

data['DstPort'] = np.where(data.dst_port.between(0, 1023, inclusive='both'),
                      'DstPortWellKnown',
                      np.where(data.dst_port.between(1024, 49151, inclusive='both'),
                               'DstPortRegistered', 'DstPortPrivate'))

data = pd.get_dummies(data, columns = ['SrcPort', 'DstPort'], prefix="", prefix_sep="")

print("Dropping some columns...")

data.drop(columns=['src_port', 'dst_port', 'src_ip', 'dst_ip',
                   'http_uri', 'weird_name', 'weird_addl', 'weird_notice',
                   'dns_query', 'ssl_version', 'ssl_cipher', 
                   'ssl_subject', 'ssl_issuer', 'http_user_agent',
                   'http_method', 'http_version', 'http_request_body_len',
                   'http_response_body_len', 'http_status_code', 'http_user_agent',
                   'http_orig_mime_types', 'http_resp_mime_types', 'http_trans_depth'], inplace=True)

print("Encoding boolean columns")
for c in ['dns_AA', 'dns_RA', 'dns_RD', 'dns_rejected', 'ssl_resumed', 'ssl_established']:
    data[c].replace('-', 'F', inplace=True)
    data.loc[data[c] == 'F', c] = 0
    data.loc[data[c] == 'T', c] = 1
    
print("One hot encoding categorical columns...")
data = pd.get_dummies(data, columns = ['conn_state', 'service'])

data.rename({'label': 'Label',
             'duration': 'Dur',
             'src_bytes': 'SrcBytes',
             'dst_bytes': 'DstBytes',
             'src_pkts': 'SrcPkts',
             'dst_pkts': 'DstPkts'
             }, axis=1, inplace=True)

data['TotBytes'] = data.SrcBytes + data.DstBytes
data['TotPkts'] = data.SrcPkts + data.DstPkts

data = data.drop(columns=['type'])
data.Label = data.Label.astype('int')

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

print("Saving Train and Test sets")
train_data.to_csv(train_save_path, index=None)
test_data.to_csv(test_save_path, index=None)