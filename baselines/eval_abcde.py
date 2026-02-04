import os
import pickle
import torch
import numpy as np
import networkx as nx
import fcntl
import time
import argparse
from scipy.stats import kendalltau
from torch_geometric.data import Data
from abcde.models import ABCDE

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="experiments/latest/models/best.ckpt", help="Path to model checkpoint")
parser.add_argument("--algorithm_name", default="ABCDE_Train", help="Algorithm name for CSV")
args = parser.parse_args()

MODEL_PATH = args.model_path
RESULTS_FILE = "results/all_results.csv"
RESULTS_FILE_TOPK = "results/all_results_topk.csv"
RESULTS_FILE_WALLCLOCK = "results/all_results_wallclock.csv"

TEST_GRAPHS = ["web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", \
        "email-EuAll", "road-luxembourg-osm", "wiki-Talk", "road-roadNet-PA", \
        "road-belgium-osm", "road-roadNet-CA", "road-netherlands-osm", \
        "soc-LiveJournal1", "cit-Patents", "wiki-topcats", "soc-Pokec", \
        "amazon", "com-lj", "com-youtube", "dblp"]

def get_graph_data(name):
    if name in ["SF", "HY", "ER", "GRP"] or name.startswith("SF_") or name.startswith("HY_"):
        path = f"./datasets/data_splits/{name}/betweenness/test.pickle"
    else:
        path = f"./datasets/data_splits/{name}_bet.pickle"

    if not os.path.exists(path):
        return None, None, None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            graphs = data[0]
            sequences = data[1]
            bc_mats = data[3]
            G = graphs[0]
            node_seq = sequences[0]
            if isinstance(bc_mats, np.ndarray):
                bc_scores = bc_mats[:, 0]
            elif isinstance(bc_mats, list):
                bc_scores = bc_mats[0]
            num_nodes = len(node_seq)
            if len(bc_scores) > num_nodes:
                bc_scores = bc_scores[:num_nodes]
            return G, node_seq, bc_scores
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None, None

def convert_to_pyg(G, node_seq, bc_scores):
    mapping = {original_id: i for i, original_id in enumerate(node_seq)}
    G = nx.relabel_nodes(G, mapping)
    edge_index = np.array(G.to_directed(as_view=True).edges).T
    degrees = nx.degree_centrality(G)
    degrees_arr = np.array([degrees[i] for i in range(len(G))], dtype='float32')
    degrees_arr = np.expand_dims(degrees_arr, -1)
    label = np.array(bc_scores, dtype='float32')
    label = np.expand_dims(label, axis=-1)
    data = Data(x=torch.from_numpy(degrees_arr), y=torch.from_numpy(label), edge_index=torch.from_numpy(edge_index))
    data.src_ids = torch.zeros(1, dtype=torch.long)
    data.tgt_ids = torch.zeros(1, dtype=torch.long)
    data.num_nodes = G.number_of_nodes()
    return data

def load_patched_model(path):
    print(f"Loading and patching model from {path}...")
    model = ABCDE(
        nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
        conv_sizes=(48, 48, 32, 32, 24, 24),
        drops=(0.3, 0.3, 0.2, 0.2, 0.1, 0.1),
        lr_reduce_patience=2, 
        dropout=0.1
    )
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    target_state_dict = model.state_dict()
    for key, value in state_dict.items():
        new_key = key
        new_val = value
        if "conv_blocks" in key and ".weight" in key and "lin.weight" not in key:
            parts = key.split('.')
            if len(parts) > 3 and parts[3] == '0': 
                new_key = key.replace(".weight", ".lin.weight")
                new_val = value.T
        if new_key in target_state_dict:
            target_shape = target_state_dict[new_key].shape
            if new_val.shape != target_shape:
                if new_val.shape == torch.Size([1]) and len(target_shape) == 1:
                    new_val = new_val.repeat(target_shape[0])
        new_state_dict[new_key] = new_val
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def calculate_topk(pred, label, p):
    k = int(len(label) * p)
    if k < 1: k = 1
    top_k_pred = np.argpartition(pred, -k)[-k:]
    top_k_true = np.argpartition(label, -k)[-k:]
    common = np.intersect1d(top_k_pred, top_k_true)
    return len(common) / k

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists("results"):
        os.makedirs("results")

    model = load_patched_model(MODEL_PATH)
    results_kt = []
    results_top1 = []
    results_top5 = []
    results_top10 = []
    times = []

    for name in TEST_GRAPHS:
        print(f"Processing {name}...", end=" ", flush=True)
        G, node_seq, bc_scores = get_graph_data(name)
        if G is None:
            results_kt.append("")
            results_top1.append("")
            results_top5.append("")
            results_top10.append("")
            times.append("")
            print("Skipped")
            continue
            
        pyg_data = convert_to_pyg(G, node_seq, bc_scores)
        try:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                out = model(pyg_data)
                pred_scores = out.view(-1).cpu().numpy()
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end = time.time()
            elapsed = end - start
            
            true_scores = bc_scores
            kt, _ = kendalltau(pred_scores, true_scores)
            
            t1 = calculate_topk(pred_scores, true_scores, 0.01)
            t5 = calculate_topk(pred_scores, true_scores, 0.05)
            t10 = calculate_topk(pred_scores, true_scores, 0.10)
            
            print(f"KT: {kt:.4f} | T1%: {t1:.4f} | Time: {elapsed:.4f}s")
            
            results_kt.append(f"{kt:.4f}")
            results_top1.append(f"{t1:.4f}")
            results_top5.append(f"{t5:.4f}")
            results_top10.append(f"{t10:.4f}")
            times.append(f"{elapsed:.6f}")
            
        except Exception as e:
            print(f"Error: {e}")
            results_kt.append("")
            results_top1.append("")
            results_top5.append("")
            results_top10.append("")
            times.append("")

    alg_name = args.algorithm_name
    
    header = "Algorithm," + ",".join(TEST_GRAPHS)
    csv_row = f"{alg_name}," + ",".join(results_kt)
    with open(RESULTS_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0: f.write(header + "\n")
        f.write(csv_row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    
    with open(RESULTS_FILE_TOPK, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0: f.write(header + "\n")
        f.write(csv_row + "\n")
        f.write(f"{alg_name}_Top1%," + ",".join(results_top1) + "\n")
        f.write(f"{alg_name}_Top5%," + ",".join(results_top5) + "\n")
        f.write(f"{alg_name}_Top10%," + ",".join(results_top10) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    csv_row_time = f"{alg_name}," + ",".join(times)
    with open(RESULTS_FILE_WALLCLOCK, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0: f.write(header + "\n")
        f.write(csv_row_time + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

if __name__ == "__main__":
    main()