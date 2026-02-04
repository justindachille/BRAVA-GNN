import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
import argparse
import os
import collections
import fcntl
import time

#Loading graph data
parser = argparse.ArgumentParser()
parser.add_argument("--g",default="HY")
parser.add_argument("--top_k", action="store_true", help="Calculate and log Top-K accuracy metrics")
parser.add_argument("--seed", type=int, default=20, help="Random seed")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

gtype = args.g
print(gtype)
if gtype == "SF":
    data_path = "./datasets/data_splits/SF/betweenness/"
    print("Scale-free graphs selected.")

elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/betweenness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/betweenness/"
    print("Gaussian Random Partition graphs selected.")

elif gtype == "HY":
    data_path = "./datasets/data_splits/HY/betweenness/"
    print("Hyperbolic graphs selected.")



#Load training data
print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)


with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

# Define full test suite
TEST_GRAPHS = ["web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", \
        "email-EuAll", "road-luxembourg-osm", "wiki-Talk", "road-roadNet-PA", \
        "road-belgium-osm", "road-roadNet-CA", "road-netherlands-osm",
        "soc-LiveJournal1", "cit-Patents", "wiki-topcats", "soc-Pokec", \
        "amazon", "com-lj", "com-youtube", "dblp"]


real_data_dict = dict()
max_test_nodes = 0

for data in TEST_GRAPHS:
    path = f"./datasets/data_splits/{data}_bet.pickle"
    if os.path.exists(path):
        print(f"Loading {data} graph...")
        with open(path,"rb") as fopen:
            list_graph,list_n_seq,list_num_node,bc_mat = pickle.load(fopen)
            real_data_dict[data] = [list_graph,list_n_seq,list_num_node,bc_mat]
            # Track max nodes for model sizing
            if list_num_node:
                max_test_nodes = max(max_test_nodes, max(list_num_node))
    else:
        print(f"Skipping {data} (not found)")

# Determine model size based on largest graph in train or test
max_train_nodes = max(list_num_node_train) if list_num_node_train else 0
model_size = max(10000, max_train_nodes, max_test_nodes)
print(f"Model initialized with size: {model_size}")

#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)

# Prepare validation set (road-belgium-osm)
val_idx = 3
val_name = TEST_GRAPHS[val_idx] if len(TEST_GRAPHS) > val_idx else ""
list_adj_val, list_adj_t_val, list_num_node_val, bc_mat_val = [], [], [], []
if val_name in real_data_dict:
    print(f"Preparing validation set: {val_name}")
    g_val, seq_val, num_val, bc_val = real_data_dict[val_name]
    list_adj_val, list_adj_t_val = graph_to_adj_bet(g_val, seq_val, num_val, model_size)
    list_num_node_val, bc_mat_val = num_val, bc_val

def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test):
    model.eval()
    loss_val = 0
    list_kt = list()
    topk_lists = collections.defaultdict(list)
    total_inference_time = 0

    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        y_out = model(adj,adj_t)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.time()
        total_inference_time += (end - start)
    
        # Handle format difference: bc_mat can be matrix or list of arrays depending on dataset source
        if isinstance(bc_mat_test, np.ndarray) and bc_mat_test.ndim > 1:
             true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        else:
             true_arr = torch.from_numpy(bc_mat_test[j] if isinstance(bc_mat_test, list) else bc_mat_test).float()

        true_val = true_arr.to(device)
    
        if args.top_k:
            kt, topk_dict = ranking_correlation_topk(y_out, true_val, num_nodes, model_size)
            list_kt.append(kt)
            for p, acc in topk_dict.items():
                topk_lists[p].append(acc)
        else:
            kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
            list_kt.append(kt)

    mean_kt = np.mean(np.array(list_kt))
    avg_inference_time = total_inference_time / num_samples_test
    print(f"   Average KT score on test graphs is: {mean_kt:.4f} and std: {np.std(np.array(list_kt)):.4f}")
    
    topk_means = {}
    if args.top_k:
        topk_means = {p: np.mean(vals) for p, vals in topk_lists.items()}
        print(f"   Top 1%: {topk_means.get(0.01, 0):.4f} | Top 5%: {topk_means.get(0.05, 0):.4f} | Top 10%: {topk_means.get(0.1, 0):.4f}")

    return mean_kt, topk_means, avg_inference_time


#Model parameters
hidden = 12

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)
model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {num_params}")

optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
num_epoch = 10

print("Training")
print(f"Total Number of epoches: {num_epoch}")
training_start_time = time.time()
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)

    #to check test loss while training
    with torch.no_grad():
        test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)
training_end_time = time.time()
total_training_time = training_end_time - training_start_time

print("Testing on real datasets")
scores_per_dataset = collections.defaultdict(list)
topk_scores_per_dataset = collections.defaultdict(dict)
times_per_dataset = collections.defaultdict(list)

for data in TEST_GRAPHS:
    if data not in real_data_dict:
        continue
    print(f"Testing on {data} dataset")
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = real_data_dict[data]
    list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)
    with torch.no_grad():
        kt, topk_means, avg_time = test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)
        scores_per_dataset[data].append(kt)
        times_per_dataset[data].append(avg_time)
        if args.top_k:
            topk_scores_per_dataset[data] = topk_means

# CSV Output
print("\n" + "="*30)
print("Writing results to all_results.csv")
print("="*30)
alg_name = f"GNN_Bet_{gtype}_S{args.seed}"
header = "Algorithm," + ",".join(TEST_GRAPHS)

results_str = [alg_name]
for name in TEST_GRAPHS:
    if name in scores_per_dataset and scores_per_dataset[name]:
        results_str.append(f"{scores_per_dataset[name][0]:.4f}")
    else:
        results_str.append("")
csv_row = ",".join(results_str)

results_file = "all_results.csv"
with open(results_file, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(header + "\n")
    f.write(csv_row + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)

# 2. New Top-K Results Output
if args.top_k:
    results_file_topk = "all_results_topk.csv"
    print(f"Writing detailed top-k results to {results_file_topk}...")

    with open(results_file_topk, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(header + "\n")
        
        # Write KT Row (Same as standard file but in this new file too for completeness)
        f.write(csv_row + "\n")

        # Write Top-K Rows
        for p_label, p_val in [("Top1%", 0.01), ("Top5%", 0.05), ("Top10%", 0.1)]:
            row_parts = [f"{alg_name}_{p_label}"]
            for name in TEST_GRAPHS:
                if name in topk_scores_per_dataset and p_val in topk_scores_per_dataset[name]:
                    row_parts.append(f"{topk_scores_per_dataset[name][p_val]:.4f}")
                else:
                    row_parts.append("")
            f.write(",".join(row_parts) + "\n")
            
        fcntl.flock(f, fcntl.LOCK_UN)

# Generate Wallclock CSV
print("Writing wallclock results to all_results_wallclock.csv")
results_file_wallclock = "all_results_wallclock.csv"
results_str_time = [alg_name]
for name in TEST_GRAPHS:
    if name in times_per_dataset and times_per_dataset[name]:
        results_str_time.append(f"{times_per_dataset[name][0]:.6f}")
    else:
        results_str_time.append("")
csv_row_time = ",".join(results_str_time)

with open(results_file_wallclock, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(header + "\n")
    f.write(csv_row_time + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)

results_file_training = "all_results_training_time.csv"
with open(results_file_training, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    if f.tell() == 0:
        f.write("Algorithm,Time\n")
    f.write(f"{alg_name},{total_training_time:.4f}\n")
    fcntl.flock(f, fcntl.LOCK_UN)

print("Done.")