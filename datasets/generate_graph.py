import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
import os
import argparse

np.random.seed(1)

def load_real_data(file_path, is_undirected=False):
    original_to_new = {}
    new_id = 0
    edges = []
    
    # Read file and remap IDs
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comments
            from_node, to_node = map(int, line.split())
            
            if from_node not in original_to_new:
                original_to_new[from_node] = new_id
                new_id += 1
            
            if to_node not in original_to_new:
                original_to_new[to_node] = new_id
                new_id += 1
            
            edges.append((original_to_new[from_node], original_to_new[to_node]))
            if is_undirected:
                edges.append((original_to_new[to_node], original_to_new[from_node]))
    
    # Create graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    return G

def create_graph(graph_type, idx=0):

    num_nodes =100000 #np.random.randint(100000,100000)

    if graph_type == "ER":
        #Erdos-Renyi random graphs
        p = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes,p = p,directed = True)
        return g_nx

    if graph_type == "SF" or graph_type.startswith("SF_"):
        #Scalefree graphs
        alpha = np.random.randint(40,60)*0.01
        gamma = 0.05
        beta = 1 - alpha - gamma
        g_nx = nx.scale_free_graph(num_nodes,alpha = alpha,beta = beta,gamma = gamma)
        return g_nx


    if graph_type == "GRP":
        #Gaussian-Random Partition Graphs
        s = np.random.randint(200,1000)
        v = np.random.randint(200,1000)
        p_in = np.random.randint(2,25)*0.0001
        p_out = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.gaussian_random_partition_graph(num_nodes,s = s, v = v, p_in = p_in, p_out = p_out, directed = True)
        assert nx.is_directed(g_nx)==True,"Not directed"
        return g_nx
    
    if graph_type.startswith("HY"):
        num_nodes = 100000
        
        # Determine generation mode
        # Mode 0: Fixed (Original)
        # Mode 1: Uniform Random
        # Mode 2: Real Data Stats (Current/Default)
        # k = average degree, gamma = power law exponent, T = temperature
        
        mode = 2
        
        parts = graph_type.split('_')
        # Check for suffix like HY_10_0, HY_10_1, etc.
        if len(parts) >= 3 and parts[-1].isdigit():
             mode = int(parts[-1])
        
        if graph_type == "HY":
            mode = 0

        if mode == 0:
            k = 25
            gamma = 3
            T = 0
        elif mode == 1:
            k = np.random.uniform(3, 28)
            gamma = np.random.uniform(2.1, 7.2)
            T = np.random.uniform(0, 0.5)
        else:
            REAL_DATA_STATS = [
                (12.1298, 2.2056),  # soc-Slashdot0811
                (32.4296, 2.4494),  # gemsec-Facebook
                (15.3317, 2.5023),  # musae-github
                (80.8684, 2.1334),  # twitch-gamers
                (76.2814, 2.2849),  # com-Orkut
                (6.6221, 3.5380),   # com-DBLP
                (19.4080, 2.6299),  # web-BerkStan
                (6.6933, 2.4715),   # web-NotreDame
                (10.0202, 2.3982),  # email-Enron
                (4.8159, 7.0192),   # p2p-Gnutella30
                (2.7852, 6.8398),   # roadNet-TX
                (2.4140, 6.9798),   # road-euroroad
                (2.5677, 2.9640),   # road-usroads-48
                (7.1955, 2.5423),   # road-usroads
                (2.0980, 7.9782),   # road-italy-osm
            ]            
            target_k, target_gamma = REAL_DATA_STATS[np.random.randint(len(REAL_DATA_STATS))]

            k = target_k * np.random.uniform(0.95, 1.05)
            gamma = target_gamma * np.random.uniform(0.95, 1.05)
            
            # Clamp gamma > 2 (Requirement for HyperbolicGenerator)
            gamma = max(2.1, gamma)
            T = np.random.uniform(0, 0.5)
        
        print(f" [DEBUG] HY Params (Mode {mode}): k={k:.4f}, gamma={gamma:.4f}, T={T:.4f}")
        hg = generators.HyperbolicGenerator(num_nodes, k, gamma, T)
        hgG = hg.generate()
        return nkit2nx(hgG)


def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    return g_nkit

def nkit2nx(g_nkit):
    return nxadapter.nk2nx(g_nkit)


def cal_exact_bet(g_nx):

    #exact_bet = nx.betweenness_centrality(g_nx,normalized=True)

    exact_bet = centrality.Betweenness(g_nkit,normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    return exact_bet_dict

def cal_exact_close(g_nx):
    
    #exact_close = nx.closeness_centrality(g_nx, reverse=False)

    exact_close = centrality.Closeness(g_nkit,True,1).run().ranking()

    exact_close_dict = dict()
    for j in exact_close:
        exact_close_dict[j[0]] = j[1]

    return exact_close_dict



print("Load and process data graphs")
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["Wiki-Vote", "soc-Epinions1", "web-Google", "soc-Slashdot0811","p2p-Gnutella31", "road-minnesota", "road-euroroad"])
args = parser.parse_args()

output_dir = "./datasets/graphs/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for data in args.datasets:
    # Check if this is a synthetic graph type
    is_synthetic = data in ["SF", "ER", "GRP"] or data.startswith("HY") or data.startswith("SF_")
    
    if is_synthetic:
        fname_bet = "./datasets/graphs/"+data+"_data_bet.pickle"
        if os.path.exists(fname_bet):
            print(f"Skipping {data} (exists)")
            continue
        
        # Determine number of graphs to generate
        if data.startswith("HY_") or data.startswith("SF_"):
            try:
                num_of_graphs = int(data.split("_")[1])
            except:
                num_of_graphs = 5
        elif data == "HY":
            num_of_graphs = 5
        elif data == "HY_10":
            num_of_graphs = 10
        elif data == "HY_20":
            num_of_graphs = 20
        else:
            num_of_graphs = 5
            
        print("###################")
        print(f"Generating synthetic graph type : {data}")
        print(f"Number of graphs to be generated:{num_of_graphs}")
        list_bet_data = list()
        list_close_data = list()
        print("Generating graphs and calculating centralities...")
        for i in range(num_of_graphs):
            print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
            g_nx = create_graph(data, i)
            
            num_before = g_nx.number_of_nodes()
            if nx.number_of_isolates(g_nx)>0:
                #print("Graph has isolates.")
                g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
                g_nx = nx.convert_node_labels_to_integers(g_nx)
            
            num_after = g_nx.number_of_nodes()
            print(f" -> Nodes: {num_before} -> {num_after}")

            g_nkit = nx2nkit(g_nx)
            bet_dict = cal_exact_bet(g_nkit)
            close_dict = cal_exact_close(g_nkit)
            list_bet_data.append([g_nx,bet_dict])
            list_close_data.append([g_nx,close_dict])

        fname_bet = "./datasets/graphs/"+data+"_data_bet.pickle"    
        fname_close = "./datasets/graphs/"+data+"_data_close.pickle"

        with open(fname_bet,"wb") as fopen:
            pickle.dump(list_bet_data,fopen)

        with open(fname_close,"wb") as fopen1:
            pickle.dump(list_close_data,fopen1)
        print("")
        print(f"{data} Graphs saved")
        
    else:
        # Real graph processing
        fname_bet = os.path.join(output_dir, data + "_bet.pickle")
        if os.path.exists(fname_bet):
            print(f"Skipping {data} (output exists at {fname_bet})")
            continue

        # Try Murata path
        input_path_murata = f"./datasets/real_graph/murata/{data}.txt"
        input_path_roads = f"./datasets/real_graph/roads/{data}.txt"
        
        G = None
        if os.path.exists(input_path_murata):
            print(f"Processing {data} (Murata)...")
            G = load_real_data(input_path_murata)
        elif os.path.exists(input_path_roads):
            print(f"Processing {data} (Roads)...")
            G = load_real_data(input_path_roads, is_undirected=True)
            if nx.number_of_isolates(G)>0:
                G.remove_nodes_from(list(nx.isolates(G)))
                G = nx.convert_node_labels_to_integers(G)
            components = nx.strongly_connected_components(G)
            for component in components:
                print(f'Diameter of component: {data}: D={nx.diameter(G.subgraph(component))})')
        else:
            print(f"Warning: Raw data for {data} not found in murata or roads folder")
            continue

        if nx.number_of_isolates(G)>0:
            G.remove_nodes_from(list(nx.isolates(G)))
            G = nx.convert_node_labels_to_integers(G)
        g_nkit = nx2nkit(G)
        bet_dict = cal_exact_bet(g_nkit)
        list_bet_data = [[G,bet_dict]]
        with open(fname_bet,"wb") as fopen:
            pickle.dump(list_bet_data,fopen)
        print(f"{data} graph saved")

print("End.")