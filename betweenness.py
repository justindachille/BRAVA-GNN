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
import matplotlib.pyplot as plt
import subprocess
import sys
import fcntl
import time
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="baseline", choices=["baseline", "repeats", "more_layers"], help="Model architecture mode")
parser.add_argument("--repeats", type=int, default=1, help="Number of repeats for layers or number of layers in 'more_layers' mode")
parser.add_argument("--init_type", default="AW", help="Initialization type: AW, degree, degree0, degree1, degree2, degree3, degree_embedding")
parser.add_argument("--train_type", default="SF", help="Train on SF or HY graphs (e.g., HY_10, HY_50, SF_HY, SF_HY_160)")
parser.add_argument("--leverage", action="store_true", help="Include Leverage Centrality in node initialization")
parser.add_argument("--lcc", action="store_true", help="Include Local Clustering Coefficient in node initialization")
parser.add_argument("--run_all_tests", action="store_true", help="Whether to run full test suite")
parser.add_argument("--nhid", type=int, default=12, help="Number of hidden parameters")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers for baseline mode")
parser.add_argument("--skip_gen", action="store_true", help="Skip graph generation and wait for files to appear")
parser.add_argument("--top_k", action="store_true", help="Calculate and log Top-K accuracy metrics")
parser.add_argument("--normalize", action="store_true", help="Normalize input features by graph max to [0,1]")
parser.add_argument("--accumulate", type=int, default=1, help="Number of steps for gradient accumulation")
parser.add_argument("--seed", type=int, default=20, help="Random seed")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

TRAIN_GRAPHS = ["SF"]
if args.train_type == "SF_HY":
    TRAIN_GRAPHS = ["SF", "HY"]
elif args.train_type.startswith("SF_HY_"):
    TRAIN_GRAPHS = ["SF", args.train_type.replace("SF_", "")]
elif args.train_type.startswith("SF_") and "_HY_" in args.train_type:
    parts = args.train_type.split("_HY_")
    TRAIN_GRAPHS = [parts[0], "HY_" + parts[1]]
elif args.train_type.startswith("HY"):
    TRAIN_GRAPHS = [args.train_type]
elif args.train_type.startswith("SF_"):
    TRAIN_GRAPHS = [args.train_type]

# TEST_GRAPHS = ["wiki_vote", "web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", "road-minnesota", "road-euroroad"]
TEST_GRAPHS = ["road-belgium-osm", "road-roadNet-CA", "amazon", "cit-Patents", "com-lj"]
TEST_GRAPHS = ["web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", \
    "email-EuAll", "wiki-Talk", \
    "soc-LiveJournal1", "cit-Patents", "wiki-topcats", "soc-Pokec", \
    "amazon", "com-lj", "com-youtube", "dblp"] 

if args.run_all_tests:
    # TEST_GRAPHS = ["wiki_vote", "web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", "road-minnesota", \
                # "road-euroroad", "email-EuAll", "road-luxembourg-osm", "wiki-Talk", "road-roadNet-PA", "p2p-Gnutella05"]
    # TEST_GRAPHS = ["web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", \
        # "email-EuAll", "road-luxembourg-osm", "wiki-Talk", "road-roadNet-PA", \
        # "road-belgium-osm", "road-roadNet-CA", "road-netherlands-osm", \
        # "soc-LiveJournal1", "cit-Patents", "wiki-topcats", "soc-Pokec"]

    TEST_GRAPHS = ["web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", \
        "email-EuAll", "road-luxembourg-osm", "wiki-Talk", "road-roadNet-PA", \
        "road-belgium-osm", "road-roadNet-CA", "road-netherlands-osm", \
        "soc-LiveJournal1", "cit-Patents", "wiki-topcats", "soc-Pokec", \
        "amazon", "com-lj", "com-youtube", "dblp"] 
        # "road-italy-osm"
# TRAIN_GRAPHS = ["HY"]
# TEST_GRAPHS = ["road-euroroad", "wiki_vote", "road-minnesota", "soc-Epinions1"]

def is_synthetic(g):
    return g in ["SF", "ER", "GRP"] or g.startswith("HY") or g.startswith("SF_")

def check_and_generate(train_graphs, test_graphs):
    needed_gen = []
    
    for g in set(train_graphs + test_graphs):
        if is_synthetic(g):
            # Check for splits (training check is sufficient for existence)
            if not os.path.exists(f"./datasets/data_splits/{g}/betweenness/training.pickle"):
                needed_gen.append(g)
        else:
            if not os.path.exists(f"./datasets/data_splits/{g}_bet.pickle"):
                needed_gen.append(g)
            
    if needed_gen:
        print(f"Missing datasets detected: {needed_gen}. Running generation scripts...")
        
        cmd_gen = [sys.executable, "-u", "datasets/generate_graph.py", "--datasets"] + needed_gen
        subprocess.check_call(cmd_gen)
        
        cmd_create = [sys.executable, "-u", "datasets/create_dataset.py", "--datasets"] + needed_gen
        subprocess.check_call(cmd_create)

if not args.skip_gen:
    check_and_generate(TRAIN_GRAPHS, TEST_GRAPHS)
else:
    print("Skipping generation check (Waiting mode).")

def wait_for_file(filepath):
    """Waits for a file to appear before continuing, checking every 30 seconds."""
    if not os.path.exists(filepath):
        print(f"Waiting for {filepath} to be created by another job...")
        while not os.path.exists(filepath):
            time.sleep(30)
        print(f"Found {filepath}. Resuming.")
        # Give it a few extra seconds to ensure write completion
        time.sleep(5)

TURN_MODEL_PICKLING_OFF = True

# parser = argparse.ArgumentParser()
# parser.add_argument("--g",default="SF")
# args = parser.parse_args()
# gtype = args.g
gtype = args.train_type
print(f'Training on {gtype} | Mode: {args.mode} | Repeats: {args.repeats} | Init: {args.init_type} | Lev: {args.leverage} | LCC: {args.lcc} | Nhid: {args.nhid} | TopK: {args.top_k}')
print(f"Normalization: {args.normalize} | Accumulation Steps: {args.accumulate} | Layers: {args.num_layers} | Seed: {args.seed} | Dropout: {args.dropout} | Epochs: {args.epochs}")

#Load training data
print(f"Loading data...")
list_graph_train, list_n_seq_train, list_num_node_train = [], [], []
bc_mat_train = [] # Changed to list of arrays to handle dynamic sizes

latest_mtime = 0

for g in TRAIN_GRAPHS:
    if is_synthetic(g):
        path = f"./datasets/data_splits/{g}/betweenness/training.pickle"
    else:
        path = f"./datasets/data_splits/{g}_bet.pickle"
    
    if args.skip_gen:
        wait_for_file(path)

    if os.path.exists(path):
        print(f'Loading {path}')
        latest_mtime = max(latest_mtime, os.path.getmtime(path))
        with open(path,"rb") as fopen:
            data = pickle.load(fopen)
            # data structure: [list_graph, list_n_seq, list_node_num, cent_mat]
            list_graph_train.extend(data[0])
            list_n_seq_train.extend(data[1])
            list_num_node_train.extend(data[2])
            
            # Process centrality matrix
            # cent_mat is (max_nodes, num_samples)
            # We convert this to a list of 1D arrays, slicing only the valid nodes
            cent_mat_chunk = data[3]
            nodes_in_chunk = data[2]
            
            for k in range(cent_mat_chunk.shape[1]):
                valid_count = nodes_in_chunk[k]
                # Slice exactly valid_count to allow dynamic mixing without padding errors
                bc_mat_train.append(cent_mat_chunk[:valid_count, k])

unique_indices = []
seen_graph_ids = set()
for i, g in enumerate(list_graph_train):
    gid = id(g)
    if gid not in seen_graph_ids:
        seen_graph_ids.add(gid)
        unique_indices.append(i)

if len(unique_indices) < len(list_graph_train):
    print(f"Filtering duplicates from dataset: Reduced from {len(list_graph_train)} to {len(unique_indices)} unique graphs.")
    list_graph_train = [list_graph_train[i] for i in unique_indices]
    list_n_seq_train = [list_n_seq_train[i] for i in unique_indices]
    list_num_node_train = [list_num_node_train[i] for i in unique_indices]
    bc_mat_train = [bc_mat_train[i] for i in unique_indices]


test_data_dict = {}
for g in TEST_GRAPHS:
    if is_synthetic(g):
        path = f"./datasets/data_splits/{g}/betweenness/test.pickle"
    else:
        path = f"./datasets/data_splits/{g}_bet.pickle"
    
    if args.skip_gen:
        wait_for_file(path)

    if os.path.exists(path):
        print(f"  Loading {path}")
        latest_mtime = max(latest_mtime, os.path.getmtime(path))
        with open(path,"rb") as fopen:
            d = pickle.load(fopen)
            # Convert matrix to list of arrays for test data as well
            c_mat = d[3]
            l_nodes = d[2]
            c_list = [c_mat[:l_nodes[k], k] for k in range(c_mat.shape[1])]
            
            test_data_dict[g] = (d[0], d[1], d[2], c_list)
    else:
        print(f'dataset {g} not found at {path}, skipping')

# Determine model size (max nodes encountered) for initialization
max_train_size = max(list_num_node_train) if list_num_node_train else 0
max_test_size = 0
for key in test_data_dict:
    # test_data_dict[key][2] is list_num_node
    if test_data_dict[key][2]:
        max_test_size = max(max_test_size, max(test_data_dict[key][2]))

model_size = max(max_train_size, max_test_size)

print(f"Global Max Nodes: {model_size} (Train: {max_train_size}, Test Max: {max_test_size})")
print(f"Model initialized with size: {model_size}")

if not os.path.exists("pickles"): os.makedirs("pickles")

# Prepare validation set (use first available test graph for monitoring)
val_idx = 0
# val_idx = 8
val_key = TEST_GRAPHS[val_idx] if (len(TEST_GRAPHS) > val_idx and TEST_GRAPHS[val_idx] in test_data_dict) else None
if val_key:
    list_graph_val, list_n_seq_val, list_num_node_val, bc_mat_val = test_data_dict[val_key]
    print(f"Using {val_key} as validation set.")
else:
    list_graph_val, list_n_seq_val, list_num_node_val, bc_mat_val = [], [], [], []

adj_cache_path = f"pickles/adj_data_scipy_{gtype}_{model_size}_{val_key}.pickle"
lock_path = adj_cache_path + ".lock"
with open(lock_path, "w") as lock_file:
    print(f"Acquiring lock for adjacency conversion: {adj_cache_path}")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    
    cache_valid = os.path.exists(adj_cache_path) and os.path.getmtime(adj_cache_path) > latest_mtime

    if cache_valid:
        print(f"Loading cached adjacency conversion (Scipy) from {adj_cache_path}")
        with open(adj_cache_path, "rb") as f:
            list_adj_train, list_adj_t_train, list_adj_val, list_adj_t_val = pickle.load(f)
    else:
        print(f"No valid cache found. Starting Graphs to adjacency conversion (Scipy Sparse).")
        list_adj_train, list_adj_t_train = graph_to_adj_bet(list_graph_train, list_n_seq_train, list_num_node_train, model_size)
        list_adj_val, list_adj_t_val = graph_to_adj_bet(list_graph_val, list_n_seq_val, list_num_node_val, model_size)
        
        with open(adj_cache_path, "wb") as f:
            pickle.dump([list_adj_train, list_adj_t_train, list_adj_val, list_adj_t_val], f)
    
    fcntl.flock(lock_file, fcntl.LOCK_UN)


def train(list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train, model_size, accum_steps=1):
    model.train()
    
    gpu_graphs = []
    for i in range(len(list_adj_train)):
        adj = sparse_mx_to_torch_sparse_tensor(list_adj_train[i]).to(device).coalesce()
        adj_t = sparse_mx_to_torch_sparse_tensor(list_adj_t_train[i]).to(device).coalesce()
        
        true_val = torch.from_numpy(bc_mat_train[i]).float().to(device)
        gpu_graphs.append((adj, adj_t, true_val, list_num_node_train[i]))
        
    num_virtual_copies = 50 
    
    optimizer.zero_grad()
    
    for _ in range(num_virtual_copies):
        indices = torch.randperm(len(gpu_graphs))
        
        for i, idx in enumerate(indices):
            adj, adj_t, true_val, node_num = gpu_graphs[idx]
            
            perm = torch.randperm(node_num, device=device)
            
            batch_true_val = true_val[perm]
            inv_perm = torch.argsort(perm)
            
            def permute_adj(src_adj, map_idx, num_nodes):
                old_indices = src_adj.indices()
                new_indices = map_idx[old_indices]
                
                return torch.sparse_coo_tensor(new_indices, src_adj.values(), (num_nodes, num_nodes), device=device).coalesce()

            batch_adj = permute_adj(adj, inv_perm, node_num)
            batch_adj_t = permute_adj(adj_t, inv_perm, node_num)
            
            y_out = model(batch_adj, batch_adj_t)
            loss = loss_cal(y_out, batch_true_val, node_num, device, model_size)
            
            # Normalize loss for accumulation
            if accum_steps > 1:
                loss = loss / accum_steps
            
            loss.backward()
            
            # Update step logic: accumulate gradients until accum_steps is reached
            if (i + 1) % accum_steps == 0 or (i + 1) == len(indices):
                optimizer.step()
                optimizer.zero_grad()

def test(list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test, model_size):
    model.eval()
    list_kt = list()
    total_inference_time = 0
    
    # Store lists for topk
    topk_lists = collections.defaultdict(list)

    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj_sparse = list_adj_test[j]
        adj_t_sparse = list_adj_t_test[j]
        
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_sparse).to(device)
        adj_t_tensor = sparse_mx_to_torch_sparse_tensor(adj_t_sparse).to(device)
        
        num_nodes = list_num_node_test[j]
        
        # Timing Inference
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        y_out = model(adj_tensor, adj_t_tensor)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.time()
        total_inference_time += (end - start)
        
        true_arr = torch.from_numpy(bc_mat_test[j]).float()
        true_val = true_arr.to(device)
    
        if args.top_k:
            kt, topk_dict = ranking_correlation_topk(y_out, true_val, num_nodes, model_size)
            list_kt.append(kt)
            for p, acc in topk_dict.items():
                topk_lists[p].append(acc)
        else:
            # Use simpler basic function if Top-K not requested
            kt = ranking_correlation(y_out, true_val, num_nodes, model_size)
            list_kt.append(kt)

    mean_kt_score = np.mean(np.array(list_kt))
    std_kt_score = np.std(np.array(list_kt))
    avg_inference_time = total_inference_time / num_samples_test
    
    topk_means = {}
    if args.top_k:
        topk_means = {p: np.mean(vals) for p, vals in topk_lists.items()}

    print(f"   Average KT score on test graphs is: {mean_kt_score:.4f} and std: {std_kt_score:.4f}")
    if args.top_k:
        print(f"   Top 1%: {topk_means.get(0.01, 0):.4f} | Top 5%: {topk_means.get(0.05, 0):.4f} | Top 10%: {topk_means.get(0.1, 0):.4f}")
    
    return mean_kt_score, std_kt_score, topk_means, avg_inference_time

hidden = args.nhid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on device: {device}')
scores_per_dataset = collections.defaultdict(list)
topk_scores_per_dataset = collections.defaultdict(dict)
times_per_dataset = collections.defaultdict(list)

model = GNN_Bet(ninput=model_size, nhid=hidden, dropout=args.dropout, mode=args.mode, repeats=args.repeats, init_type=args.init_type, leverage=args.leverage, lcc=args.lcc, normalize=args.normalize, num_layers=args.num_layers)
model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {num_params}")

optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
num_epoch = args.epochs

print("Training")
print(f"Total Number of epoches: {num_epoch}")

PICKLE_FILEPATH = f"pickles/between_network_{args.mode}_{args.repeats}_{args.init_type}_lev{args.leverage}_lcc{args.lcc}_hid{args.nhid}_norm{args.normalize}_accum{args.accumulate}_lay{args.num_layers}_ep{args.epochs}_seed{args.seed}_drop{args.dropout}.pickle"
if os.path.exists(PICKLE_FILEPATH) and not TURN_MODEL_PICKLING_OFF:
    print("Loading network pickle...")
    model.load_state_dict(torch.load(PICKLE_FILEPATH))
else:
    training_start_time = time.time()
    for e in range(num_epoch):
        print(f"Epoch number: {e+1}/{num_epoch}")
        train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model_size, args.accumulate)

        #to check test loss while training
        with torch.no_grad():
            test(list_adj_val,list_adj_t_val,list_num_node_val,bc_mat_val,model_size)
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    torch.save(model.state_dict(), PICKLE_FILEPATH)

print("Testing on real datasets")
for data_name in TEST_GRAPHS:
    if data_name not in test_data_dict:
        continue
    print(f"Testing on {data_name} dataset")
    real_graph_test, real_n_seq_test, real_num_node_test, real_bc_mat_test = test_data_dict[data_name]
    
    real_adj_test, real_adj_t_test = graph_to_adj_bet(real_graph_test, real_n_seq_test, real_num_node_test, model_size)
    with torch.no_grad():
        mean, std, topk_means, avg_time = test(real_adj_test, real_adj_t_test, real_num_node_test, real_bc_mat_test, model_size)
        scores_per_dataset[data_name].append(mean)
        times_per_dataset[data_name].append(avg_time)
        if args.top_k:
            topk_scores_per_dataset[data_name] = topk_means

# Generate CSV output for spreadsheet
print("\n" + "="*30)
print("SPREADSHEET DATA (Copy & Paste)")
print("="*30)
alg_name = f"{args.mode}_{args.init_type}_{args.train_type}_{args.nhid}"
if args.normalize: alg_name += "_norm"
if args.accumulate > 1: alg_name += f"_accum{args.accumulate}"
if args.num_layers != 4: alg_name += f"_L{args.num_layers}"
if abs(args.dropout - 0.6) > 1e-6: alg_name += f"_drop{args.dropout}"
if args.epochs != 10: alg_name += f"_E{args.epochs}"
# Append seed for differentiation
alg_name += f"_S{args.seed}"

header = "Algorithm," + ",".join(TEST_GRAPHS)
print(header)

results_str = [alg_name]
for name in TEST_GRAPHS:
    if name in scores_per_dataset and scores_per_dataset[name]:
        results_str.append(f"{scores_per_dataset[name][0]:.4f}")
    else:
        results_str.append("")
csv_row = ",".join(results_str)
print(csv_row)
print("="*30)

if not os.path.exists("results"):
    os.makedirs("results")

results_file = "results/all_results.csv"
with open(results_file, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(header + "\n")
    f.write(csv_row + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)

if args.top_k:
    results_file_topk = "results/all_results_topk.csv"
    print(f"Writing detailed top-k results to {results_file_topk}...")

    with open(results_file_topk, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(header + "\n")
        
        f.write(csv_row + "\n")

        for p_label, p_val in [("Top1%", 0.01), ("Top5%", 0.05), ("Top10%", 0.1)]:
            row_parts = [f"{alg_name}_{p_label}"]
            for name in TEST_GRAPHS:
                if name in topk_scores_per_dataset and p_val in topk_scores_per_dataset[name]:
                    row_parts.append(f"{topk_scores_per_dataset[name][p_val]:.4f}")
                else:
                    row_parts.append("")
            f.write(",".join(row_parts) + "\n")
            
        fcntl.flock(f, fcntl.LOCK_UN)

results_file_wallclock = "results/all_results_wallclock.csv"
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

results_file_training = "results/all_results_training_time.csv"
with open(results_file_training, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    if f.tell() == 0:
        f.write("Algorithm,Time\n")
    f.write(f"{alg_name},{total_training_time:.4f}\n")
    fcntl.flock(f, fcntl.LOCK_UN)