import os
import pickle
import networkx as nx
import numpy as np

# Define where to save files for ABCDE to read
ABCDE_DATA_DIR = "./datasets/real"
if not os.path.exists(ABCDE_DATA_DIR):
    os.makedirs(ABCDE_DATA_DIR)

# List of datasets you want to convert (matching your existing pickles)
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
    # 1. Remap nodes to be contiguous 0..N-1 (ABCDE requirement)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # 2. Save Edge List (dataset.txt)
    edge_path = os.path.join(ABCDE_DATA_DIR, f"{name}.txt")
    nx.write_edgelist(G, edge_path, data=False)
    
    # 3. Save Scores (dataset-score.txt)
    # Ensure scores align with the new mapping 0..N-1
    # bc_scores is assumed to be ordered by the original pickle sequence
    score_path = os.path.join(ABCDE_DATA_DIR, f"{name}-score.txt")
    
    # If bc_scores is a dict (unlikely in your pickles but good for safety)
    if isinstance(bc_scores, dict):
        ordered_scores = [bc_scores[node] for node in G.nodes()]
    else:
        # Assuming bc_scores corresponds to the node order in the pickle
        ordered_scores = bc_scores

    np.savetxt(score_path, ordered_scores, fmt='%.9f')
    print(f"Converted {name} -> {edge_path}")

def load_and_convert():
    for name in DATASETS:
        # Determine path based on your existing structure
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
                
                # Your pickles are usually: [list_graph, list_n_seq, list_num_node, bc_mat]
                # We only need the first graph for evaluation if multiple exist, 
                # or loop through them if you want average results.
                # For simplicity, let's take the first graph in the test set.
                
                graphs = data[0]
                bc_mats = data[3]
                
                # Handle matrix vs list format
                if isinstance(bc_mats, np.ndarray):
                    # It's a matrix (nodes x samples)
                    bc_scores = bc_mats[:, 0]
                elif isinstance(bc_mats, list):
                     # It's a list of arrays
                    bc_scores = bc_mats[0]
                
                # Take the first graph
                G = graphs[0]
                
                # Check for node count mismatch (padding issue handling)
                real_node_count = G.number_of_nodes()
                if len(bc_scores) > real_node_count:
                    bc_scores = bc_scores[:real_node_count]
                
                export_to_abcde(name, G, bc_scores)

        except Exception as e:
            print(f"Error converting {name}: {e}")

if __name__ == "__main__":
    load_and_convert()