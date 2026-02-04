import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
import os
import argparse
random.seed(10)

def reorder_list(input_list,serial_list):
    new_list_tmp = [input_list[j] for j in serial_list]
    return new_list_tmp

def create_dataset(list_data, num_copies, min_adj_size=10000):

    max_nodes = 0
    for g_data in list_data:
        max_nodes = max(max_nodes, g_data[0].number_of_nodes())
    
    adj_size = max(min_adj_size, max_nodes)

    num_data = len(list_data)
    total_num = num_data*num_copies
    cent_mat = np.zeros((adj_size,total_num),dtype=np.float32)
    list_graph = list()
    list_node_num = list()
    list_n_sequence = list()
    mat_index = 0
    for g_data in list_data:

        graph, cent_dict = g_data
        nodelist = [i for i in graph.nodes()]
        assert len(nodelist)==len(cent_dict),"Number of nodes are not equal"
        node_num = len(nodelist)

        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            random.shuffle(tmp_nodelist)
            list_graph.append(graph)
            list_node_num.append(node_num)
            list_n_sequence.append(tmp_nodelist)

            for ind,node in enumerate(tmp_nodelist):
                cent_mat[ind,mat_index] = cent_dict[node]
            mat_index +=  1


    if total_num > 0:
        serial_list = [i for i in range(total_num)]
        random.shuffle(serial_list)

        list_graph = reorder_list(list_graph,serial_list)
        list_n_sequence = reorder_list(list_n_sequence,serial_list)
        list_node_num = reorder_list(list_node_num,serial_list)
        cent_mat_tmp = cent_mat[:,np.array(serial_list)]
        cent_mat = cent_mat_tmp

    return list_graph, list_n_sequence, list_node_num, cent_mat


def get_split(source_file,num_train,num_test,num_copies,adj_size,save_path):

    with open(source_file,"rb") as fopen:
        list_data = pickle.load(fopen)

    num_graph = len(list_data)
    print(f"Total number of graphs in {source_file} file:{num_graph}")

    assert num_train+num_test == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
    #For training split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:num_train], num_copies=num_copies, min_adj_size=adj_size)

    with open(save_path+"training.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[num_train:num_train+num_test], num_copies=1, min_adj_size=adj_size)

    with open(save_path+"test.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)


#creating training/test dataset split for the model
adj_size = 10000
#Number of permutations for node sequence
#Can be raised higher to get more training graphs
num_copies = 1

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["web-Google", "soc-Epinions1", "soc-Slashdot0902", "p2p-Gnutella31", 
        "email-EuAll", "road-luxembourg-osm", "wiki-Talk", "road-roadNet-PA", 
        "road-belgium-osm", "road-roadNet-CA", "road-netherlands-osm", 
        "soc-LiveJournal1", "wiki-topcats", "soc-Pokec"])
args = parser.parse_args()

output_split_dir = "./datasets/data_splits/"
if not os.path.exists(output_split_dir):
    os.makedirs(output_split_dir)

for data in args.datasets:
    # Check if synthetic
    is_synthetic = data in ["SF", "ER", "GRP"] or data.startswith("HY") or data.startswith("SF_")
    
    if is_synthetic:
        print(f"Processing {data}...")
        bet_source_file = "./datasets/graphs/"+ data + "_data_bet.pickle"

        #paths for saving splits
        save_path_bet = "./datasets/data_splits/"+data+"/betweenness/"
        if not os.path.exists(save_path_bet): os.makedirs(save_path_bet)

        if os.path.exists(save_path_bet+"training.pickle") and os.path.exists(save_path_bet+"test.pickle"):
            print(f"Skipping {data} splits (exist)")
            continue

        # Determine split based on type
        cur_num_train = 5
        cur_num_test = 0

        if data.startswith(("HY_", "SF_")):
            try:
                cur_num_train = int(data.split("_")[1])
            except:
                pass
        
        #save betweenness split
        get_split(bet_source_file,cur_num_train,cur_num_test,num_copies,adj_size,save_path_bet)
        print(f" {data} Data split saved.")
        
    else:
        # Real graph logic
        source_file = "./datasets/graphs/"+data+"_bet.pickle"
        if not os.path.exists(source_file):
            print(f"Warning: Source {source_file} not found")
            continue

        dest_file = os.path.join(output_split_dir, data+"_bet.pickle")
        if os.path.exists(dest_file):
            print(f"Skipping {data} (output exists)")
            continue

        print(f"Loading and processing {data} graph...")
        with open(source_file,"rb") as fopen:
            list_data = pickle.load(fopen)
        
        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data,num_copies = 1, min_adj_size=adj_size)
        with open(dest_file,"wb") as fopen:
            pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)
        print(f"{data} graph saved.")

print("End.")