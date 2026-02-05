import os
import pickle
import networkx as nx
import numpy as np
import sys
import subprocess

ABCDE_DATA_DIR = "./datasets/real"
if not os.path.exists(ABCDE_DATA_DIR):
    os.makedirs(ABCDE_DATA_DIR)

DATASETS = [
    "SF", "HY",
    "web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31",
    "email-EuAll", "road-luxembourg-osm", "wiki-Talk", "road-roadNet-PA",
    "road-belgium-osm", "road-roadNet-CA", "road-netherlands-osm",
    "soc-LiveJournal1", "cit-Patents", "wiki-topcats", "soc-Pokec",
    "amazon", "com-lj", "com-youtube", "dblp"
]

def export_to_abcde(name, G, bc_scores):
    """
    Writes graph and scores to text files expected by ABCDE.
    G: NetworkX graph
    bc_scores: Numpy array or list of floats
    """
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    edge_path = os.path.join(ABCDE_DATA_DIR, f"{name}.txt")
    nx.write_edgelist(G, edge_path, data=False)
    
    score_path = os.path.join(ABCDE_DATA_DIR, f"{name}-score.txt")
    
    if isinstance(bc_scores, dict):
        ordered_scores = [bc_scores[node] for node in G.nodes()]
    else:
        ordered_scores = bc_scores

    np.savetxt(score_path, ordered_scores, fmt='%.9f')
    print(f"Converted {name} -> {edge_path}")

def load_and_convert():
    for name in DATASETS:
        if name in ["SF", "HY", "ER", "GRP"] or name.startswith("SF_") or name.startswith("HY_"):
            path = f"./datasets/data_splits/{name}/betweenness/test.pickle"
        else:
            path = f"./datasets/data_splits/{name}_bet.pickle"

        if not os.path.exists(path):
            print(f"Skipping {name} (File not found: {path})")
            continue

        print(f"Loading {name}...")
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                
                graphs = data[0]
                bc_mats = data[3]
                
                if isinstance(bc_mats, np.ndarray):
                    bc_scores = bc_mats[:, 0]
                elif isinstance(bc_mats, list):
                    bc_scores = bc_mats[0]
                
                G = graphs[0]
                
                real_node_count = G.number_of_nodes()
                if len(bc_scores) > real_node_count:
                    bc_scores = bc_scores[:real_node_count]
                
                export_to_abcde(name, G, bc_scores)

        except Exception as e:
            print(f"Error converting {name}: {e}")

if __name__ == "__main__":
    print("Ensuring datasets are generated...")
    subprocess.check_call([sys.executable, "datasets/generate_graph.py", "--datasets"] + DATASETS)
    subprocess.check_call([sys.executable, "datasets/create_dataset.py", "--datasets"] + DATASETS)
    load_and_convert()