import pandas as pd
import numpy as np
import os
import argparse
import re

MANUAL_DEGREE_CODE = "degree_mix_mass_6"
MANUAL_TRAIN_TYPE = "SF_10_HY_10_2"
MANUAL_LAYERS = 2
MANUAL_NHID = 12
MANUAL_DROPOUT = 0.3
MANUAL_EPOCHS = 10

RESULTS_KT = "results/all_results.csv"
RESULTS_TIME = "results/all_results_wallclock.csv"
RESULTS_TRAINING = "results/all_results_training_time.csv"

ROAD_NETWORKS = [
    "road-luxembourg-osm",
    "road-roadNet-PA",
    "road-belgium-osm",
    "road-roadNet-CA",
    "road-netherlands-osm"
]

SOCIAL_NETWORKS = [
    "p2p-Gnutella31",
    "soc-Epinions1",
    "soc-Slashdot0902",
    "email-EuAll",
    "web-Google",
    "com-youtube",
    "soc-Pokec",
    "wiki-topcats",
    "amazon",
    "wiki-Talk",
    "cit-Patents",
    "com-lj",
    "dblp",
    "soc-LiveJournal1"
]

DATASETS = ROAD_NETWORKS + SOCIAL_NETWORKS

def get_algo_name(init="degree_mix_sum", train="SF_10_HY_10_2", nhid=12, layers=4, dropout=0.6, epochs=10):
    name = f"baseline_{init}_{train}_{nhid}"
    
    if layers != 4:
        name += f"_L{layers}"
    
    if abs(dropout - 0.6) > 1e-6:
        name += f"_drop{dropout}"
    
    if epochs != 10:
        name += f"_E{epochs}"
        
    return name

ONE_CHOSEN = {
    "1": 5,
    "2": 2,
    "3": 2,
    "4": 1,
    "5": 2,
    "6": 0,
}

def load_and_aggregate(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None, None

    df = pd.read_csv(filepath)
    
    if "Time" in df.columns and not any(ds in df.columns for ds in DATASETS):
        for ds in DATASETS:
            df[ds] = df["Time"]
    
    if "Algorithm" in df.columns:
        df["Algorithm"] = df["Algorithm"].apply(lambda x: re.sub(r"_S\d+$", "", str(x)))

    for c in df.columns:
        if c != "Algorithm":
            df[c] = pd.to_numeric(df[c], errors='coerce')

    grouped = df.groupby("Algorithm")
    mean_df = grouped.mean()
    std_df = grouped.std().fillna(0)
    
    return mean_df, std_df

def format_val_std(val, std, rank):
    if np.isnan(val): return "-"
    s = f"{val:.4f} \\pm {std:.4f}"
    if rank == 0: return f"$\\mathbf{{{s}}}$"
    elif rank == 1: return f"$\\underline{{{s}}}$"
    return f"${s}$"

def calculate_improvement(our_val, baseline_vals, lower_is_better=False):
    if np.isnan(our_val): return "-"
    valid_baselines = [v for v in baseline_vals if not np.isnan(v)]
    if not valid_baselines: return "-"

    if lower_is_better:
        best_baseline = min(valid_baselines)
        if our_val < 1e-9: return "Inf"
        ratio = best_baseline / our_val
        return f"{ratio:.1f}x"
    else:
        best_baseline = max(valid_baselines)
        pct = ((our_val - best_baseline) / abs(best_baseline)) * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}\\%"

def generate_combined_main_table(mean_kt, std_kt, mean_time, std_time, dotted=False):
    our_key = get_algo_name(init=MANUAL_DEGREE_CODE, train=MANUAL_TRAIN_TYPE, 
                            nhid=MANUAL_NHID, layers=MANUAL_LAYERS, dropout=MANUAL_DROPOUT, epochs=MANUAL_EPOCHS)

    keys = ["GNN_Bet_SF", "ABCDE_Train", our_key]
    
    print(r"\begin{table*}[t]")
    print(r"  \setlength{\tabcolsep}{3pt}")
    print(r"  \caption{Comparison of Kendall Tau Correlation ($\uparrow$) and Inference Time ($\downarrow$). We report Mean $\pm$ Std. Dev. Best results are \textbf{bolded}, second best \underline{underlined}. The \textbf{\% Imp.} column shows the percentage improvement of BRAVA-GNN over the best baseline.}")
    print(r"  \label{tab:combined_results}")
    print(r"  \centering")
    print(r"  \small")
    
    print(r"  \begin{tabular}{@{}lllllllll@{}}")
    print(r"    \toprule")
    print(r"    & \multicolumn{4}{c}{\textbf{Accuracy (Kendall Tau $\uparrow$)}} & \multicolumn{4}{c}{\textbf{Time (Seconds $\downarrow$)}} \\")
    print(r"    \cmidrule(r){2-5} \cmidrule(r){6-9}") 
    print(r"    & \textbf{GNN-Bet} & \textbf{ABCDE} & \textbf{BRAVA} & \textbf{\% Imp.} & \textbf{GNN-Bet} & \textbf{ABCDE} & \textbf{BRAVA} & \textbf{Speedup} \\")
    
    groups = [
        ("Road Networks", ROAD_NETWORKS),
        ("Social and Web Networks", SOCIAL_NETWORKS)
    ]

    for group_name, dataset_list in groups:
        print(r"    \midrule")
        print(fr"    \multicolumn{{9}}{{@{{}}l@{{}}}}{{\textbf{{\textit{{{group_name}}}}}}} \\")

        for i, ds in enumerate(dataset_list):
            display_ds = ds.replace("_", r"\_")
            
            kt_vals = [mean_kt.loc[k, ds] if (k in mean_kt.index and ds in mean_kt.columns) else np.nan for k in keys]
            kt_stds = [std_kt.loc[k, ds] if (k in std_kt.index and ds in std_kt.columns) else 0 for k in keys]
            
            valid_kt = [(v, idx) for idx, v in enumerate(kt_vals) if not np.isnan(v)]
            valid_kt.sort(key=lambda x: x[0], reverse=True)
            ranks_kt = [-1] * 3
            if len(valid_kt) > 0: ranks_kt[valid_kt[0][1]] = 0
            if len(valid_kt) > 1: ranks_kt[valid_kt[1][1]] = 1
            
            kt_strs = [format_val_std(kt_vals[idx], kt_stds[idx], ranks_kt[idx]) for idx in range(3)]
            kt_imp = calculate_improvement(kt_vals[2], kt_vals[:2], False)

            t_vals = [mean_time.loc[k, ds] if (k in mean_time.index and ds in mean_time.columns) else np.nan for k in keys]
            t_stds = [std_time.loc[k, ds] if (k in std_time.index and ds in std_time.columns) else 0 for k in keys]
            
            valid_t = [(v, idx) for idx, v in enumerate(t_vals) if not np.isnan(v)]
            valid_t.sort(key=lambda x: x[0], reverse=False)
            ranks_t = [-1] * 3
            if len(valid_t) > 0: ranks_t[valid_t[0][1]] = 0
            if len(valid_t) > 1: ranks_t[valid_t[1][1]] = 1
            
            t_strs = [format_val_std(t_vals[idx], t_stds[idx], ranks_t[idx]) for idx in range(3)]
            t_imp = calculate_improvement(t_vals[2], t_vals[:2], True)

            print(f"    \\hspace{{0.5em}}{display_ds} & " + " & ".join(kt_strs) + f" & {kt_imp} & " + " & ".join(t_strs) + f" & {t_imp} \\\\")
            
            if dotted and i < len(dataset_list) - 1:
                print(r"    \hdashline")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table*}")

def format_cell(mean, std, is_best=False, is_second=False, is_chosen=False):
    if np.isnan(mean): return "-"
    val_str = f"{mean:.4f} \\pm {std:.4f}"
    if is_chosen:
        val_str += "^{\\dagger}"
    if is_best: 
        return f"$\\mathbf{{{val_str}}}$"
    if is_second:
        return f"$\\underline{{{val_str}}}$"
    return f"${val_str}$"

def _resolve_chosen(table_id, chunk_keys):
    chosen = ONE_CHOSEN.get(table_id)
    if isinstance(chosen, int):
        keys = list(chunk_keys.keys())
        return keys[chosen] if chosen < len(keys) else None
    return chosen

def generate_transposed_latex_table(title, label, mean_df, std_df, column_keys, table_id, split_threshold=7, higher_is_better=True, dotted=False):
    row_bests = {}
    row_seconds = {}
    opt_func = max if higher_is_better else min
    
    for ds in DATASETS:
        vals = [(k, mean_df.loc[k, ds]) for k in column_keys.keys() if k in mean_df.index and ds in mean_df.columns]
        if vals:
            sorted_vals = sorted(vals, key=lambda x: x[1], reverse=higher_is_better)
            row_bests[ds] = sorted_vals[0][1]
            row_seconds[ds] = sorted_vals[1][1] if len(sorted_vals) > 1 else None
        else:
            row_bests[ds] = None
            row_seconds[ds] = None

    all_col_avgs = []
    for k in column_keys.keys():
        if k in mean_df.index:
             valid_ds = [d for d in DATASETS if d in mean_df.columns]
             if valid_ds:
                 all_col_avgs.append(mean_df.loc[k, valid_ds].mean())
             else:
                 all_col_avgs.append(None)
        else:
             all_col_avgs.append(None)
             
    valid_avgs = [(i, x) for i, x in enumerate(all_col_avgs) if x is not None]
    global_best_avg = opt_func([x[1] for x in valid_avgs]) if valid_avgs else None
    sorted_avgs = sorted(valid_avgs, key=lambda x: x[1], reverse=higher_is_better)
    global_second_avg = sorted_avgs[1][1] if len(sorted_avgs) > 1 else None

    all_items = list(column_keys.items())
    num_items = len(all_items)
    
    if num_items <= split_threshold:
        chunks = [all_items]
    else:
        chunks = [all_items[i:i + split_threshold] for i in range(0, num_items, split_threshold)]

    groups = [("Road Networks", ROAD_NETWORKS), ("Social and Web Networks", SOCIAL_NETWORKS)]

    for idx, chunk in enumerate(chunks):
        chunk_keys = dict(chunk)
        chosen_key = _resolve_chosen(table_id, column_keys)
        t_suffix = ""
        l_suffix = ""
        if len(chunks) > 1:
            t_suffix = f" (Part {idx+1}/{len(chunks)})"
            l_suffix = f"_{idx+1}"

        print(f"\n% --- {title}{t_suffix} ---")
        print(r"\begin{table*}[h]")
        print(r"  \centering")
        print(r"  \renewcommand{\arraystretch}{1.2}")
        print(r"  \setlength{\tabcolsep}{2pt}")
        print(f"  \\caption{{{title}{t_suffix}}}")
        print(f"  \\label{{{label}{l_suffix}}}")
        
        col_def = "@{}l" + "l" * len(chunk_keys) + "@{}"
        print(r"  \begin{tabular}{" + col_def + "}")
        print(r"    \toprule")
        
        headers = [v + ("$^{\\dagger}$" if k == chosen_key else "") for k, v in chunk_keys.items()]
        print(r"    \textbf{Graph} & \textbf{" + "} & \\textbf{".join(headers) + r"} \\")

        for group_name, dataset_list in groups:
            print(r"    \midrule")
            print(fr"    \multicolumn{{{len(chunk_keys) + 1}}}{{@{{}}l@{{}}}}{{\textbf{{\textit{{{group_name}}}}}}} \\")

            for i, ds in enumerate(dataset_list):
                display_ds = ds.replace("_", r"\_")
                row = f"    \\hspace{{0.5em}}{display_ds}"
                
                for k in chunk_keys.keys():
                    if k in mean_df.index and ds in mean_df.columns:
                        m = mean_df.loc[k, ds]
                        s = std_df.loc[k, ds]
                        is_best = (row_bests[ds] is not None and abs(m - row_bests[ds]) < 1e-6)
                        is_second = (not is_best and row_seconds[ds] is not None and abs(m - row_seconds[ds]) < 1e-6)
                        is_chosen = (chosen_key == k)
                        row += f" & {format_cell(m, s, is_best, is_second, is_chosen)}"
                    else:
                        row += " & -"
                print(row + r" \\")
                
                if dotted and i < len(dataset_list) - 1:
                    print(r"    \hdashline")

        print(r"    \midrule")
        row = r"    \textbf{AVG}"
        
        chunk_start_idx = idx * split_threshold
        
        for i, k in enumerate(chunk_keys.keys()):
            global_idx = chunk_start_idx + i
            val = all_col_avgs[global_idx]
            
            if val is not None:
                valid_ds = [d for d in DATASETS if d in std_df.columns]
                s = std_df.loc[k, valid_ds].mean() if k in std_df.index else 0
                is_best = (global_best_avg is not None and abs(val - global_best_avg) < 1e-6)
                is_second = (not is_best and global_second_avg is not None and abs(val - global_second_avg) < 1e-6)
                is_chosen = (chosen_key == k)
                row += f" & {format_cell(val, s, is_best, is_second, is_chosen)}"
            else:
                row += " & -"
                
        print(row + r" \\")
        print(r"    \bottomrule")
        print(r"  \end{tabular}")
        print(r"\end{table*}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="all", help="Options: main, all, 1, 2, 3, 4, 5, 6")
    parser.add_argument("--mode", default="accuracy", choices=["accuracy", "inference", "training"],
                        help="Select metric.")
    parser.add_argument("--dotted", action="store_true", help="Add dotted lines between rows (requires arydshln)")
    args = parser.parse_args()

    if args.mode == "inference":
        target_file = RESULTS_TIME
        higher_is_better = False
    elif args.mode == "training":
        target_file = RESULTS_TRAINING
        higher_is_better = False
    else:
        target_file = RESULTS_KT
        higher_is_better = True

    primary_mean, primary_std = load_and_aggregate(target_file)
    if primary_mean is None: return

    # Defaults
    start_L = 2
    start_F = "degree_mix_sum"
    start_H = 12
    start_D = 0.6
    

    best_L = start_L
    best_F = start_F
    best_H = start_H
    best_D = start_D
    
    # We ran the training distribution test first
    t1_cols = {}
    t1_cols = {
        get_algo_name(init="degree1", layers=best_L, dropout=best_D): "Degree (D1)",
        get_algo_name(init="degree_mix_sum_2", layers=best_L, dropout=best_D): "2-hop (D2)",
        get_algo_name(init="degree_mix_sum", layers=best_L, dropout=best_D): "3-hop (D3)",
    }
    for x in range(4, 13):
        t1_cols[get_algo_name(init=f"degree_mix_mass_{x}", layers=best_L, dropout=best_D)] = f"{x}-hop (D{x})"
    # for x in range(1, 11):
        # t1_cols[get_algo_name(init=f"degree_mix_independent_{x}", layers=best_L, dropout=best_D)] = f"{x}-hop (D{x})"
    best_F = MANUAL_DEGREE_CODE

    
    t2_cols = {}
    for l in range(11):
        lbl = f"{l} Layers" if l > 0 else "0 Layers (MLP)"
        t2_cols[get_algo_name(init="degree_mix_sum", layers=l, dropout=best_D)] = lbl
    best_L = MANUAL_LAYERS

    t3_cols = {
        get_algo_name(train="SF_10", init=best_F, layers=best_L, dropout=best_D): "SF Only",
        get_algo_name(train="HY_10_2", init=best_F, layers=best_L, dropout=best_D): "HY Only",
        get_algo_name(train="SF_10_HY_10_2", init=best_F, layers=best_L, dropout=best_D): "SF+HY (Mix)",
    }

    t4_cols = {}
    for h in [8, 12, 16, 24, 32, 64, 128]:
        t4_cols[get_algo_name(init=best_F, layers=best_L, nhid=h, dropout=best_D)] = f"Hidden {h}"
    best_H = MANUAL_NHID

    t5_cols = {}
    for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        t5_cols[get_algo_name(init=best_F, layers=best_L, nhid=best_H, dropout=d)] = f"Drop {d}"
    best_D = MANUAL_DROPOUT
        
    t6_cols = {}
    for e in range(10, 110, 10):
        t6_cols[get_algo_name(init=best_F, layers=best_L, nhid=best_H, dropout=best_D, epochs=e)] = f"Ep {e}"

    if args.table in ["main", "all"]:
        mean_kt, std_kt = load_and_aggregate(RESULTS_KT)
        mean_time, std_time = load_and_aggregate(RESULTS_TIME)
        if mean_kt is not None and mean_time is not None:
            generate_combined_main_table(mean_kt, std_kt, mean_time, std_time, dotted=args.dotted)

    print(f"% Using Config -> Layer: {best_L}, Feature: {best_F}, Nhid: {best_H}, Drop: {best_D} | Mode: {args.mode}")
    if args.table in ["1", "all"]:
        generate_transposed_latex_table(f"Ablation: Node Features ({args.mode})", "tab:feat", primary_mean, primary_std, t1_cols, "1", 6, higher_is_better, args.dotted)

    if args.table in ["2", "all"]:
        generate_transposed_latex_table(f"Ablation: Network Depth ({args.mode})", "tab:depth", primary_mean, primary_std, t2_cols, "2", 6, higher_is_better, args.dotted)

    if args.table in ["3", "all"]:
        generate_transposed_latex_table(f"Ablation: Training Distribution ({args.mode})", "tab:train", primary_mean, primary_std, t3_cols, "3", 7, higher_is_better, args.dotted)

    if args.table in ["4", "all"]:
        generate_transposed_latex_table(f"Ablation: Hidden Dimension ({args.mode})", "tab:nhid", primary_mean, primary_std, t4_cols, "4", 7, higher_is_better, args.dotted)

    if args.table in ["5", "all"]:
        generate_transposed_latex_table(f"Ablation: Dropout ({args.mode})", "tab:drop", primary_mean, primary_std, t5_cols, "5", 7, higher_is_better, args.dotted)

    # if args.table in ["6", "all"]:
    #     generate_transposed_latex_table(f"Ablation: Epochs ({args.mode})", "tab:epochs", primary_mean, primary_std, t6_cols, "6", 6, higher_is_better, args.dotted)

if __name__ == "__main__":
    main()