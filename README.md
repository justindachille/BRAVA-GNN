# BRAVA-GNN

### Degree-Mass Message Passing for Betweenness Ranking in Directed and Undirected Networks

[![arXiv](https://img.shields.io/badge/arXiv-2602.09716-b31b1b.svg)](https://arxiv.org/abs/2602.09716)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3799682.3840622-blue.svg)](https://doi.org/10.1145/3799682.3840622)

Official implementation of **BRAVA-GNN** (Betweenness Ranking Approximation via Degree Mass GNN), a compact graph neural network for ranking nodes by betweenness centrality in directed and undirected graphs. This README follows the CIKM '26 paper and its experimental setup.

BRAVA-GNN replaces graph-size-dependent node embeddings with six size-invariant degree-mass features, corresponding to orders 0 through 5. Two shared message-passing streams process the adjacency matrix and its transpose, and their scores are combined multiplicatively. The final model has 1,297 trainable parameters, independently of the input graph size.

## Results at a glance

The paper evaluates BRAVA-GNN on 14 real-world networks from eight domains. Results are averaged over three independent runs and measured with Kendall's $\tau_b$.

| Result | BRAVA-GNN |
| --- | ---: |
| Average gain over the strongest baseline | **+9.7%** |
| Average gain on undirected graphs | **+16.1%** |
| Average gain on directed graphs | **+5.4%** |
| Largest gain on one undirected / directed graph | **+24.6% / +10.9%** |
| Trainable parameters | **1,297** |
| Parameter reduction relative to ABCDE / DrBC | **56× / 96×** |
| Directed-graph inference speedup, median / maximum | **2.5× / 24.5×** |

BRAVA-GNN obtains the best Kendall $\tau_b$ on all 14 test graphs. Inference time is competitive with the fastest GNN baseline on undirected graphs; the speedups above refer specifically to the directed benchmark.


## Installation

Run the setup script from the repository root:

```bash
source setup.sh
```

The script creates and activates `.venv`, installs `uv` when necessary, and installs PyTorch 2.3.1 with CUDA 11.8, PyTorch Geometric, the packages in `requirements.txt`, and the vendored ABCDE package. Python 3.11 is preferred when available.

## Datasets

### Test graphs

The benchmark contains six undirected and eight directed real-world graphs:

- **Undirected:** `p2p-Gnutella31`, `com-youtube`, `amazon`, `cit-Patents`, `com-lj`, `dblp`.
- **Directed:** `soc-Epinions1`, `soc-Slashdot0902`, `email-EuAll`, `web-Google`, `soc-Pokec`, `wiki-topcats`, `wiki-Talk`, `soc-LiveJournal1`.

Download the raw test data from SNAP and the ABCDE release:

```bash
python datasets/download.py
```

The five graphs distributed with precomputed scores by ABCDE are imported directly:

```bash
python datasets/import_abcde_datasets.py
```

For the nine SNAP graphs, compute exact betweenness labels and create the model-ready splits:

```bash
python datasets/generate_graph.py --datasets \
    p2p-Gnutella31 soc-Epinions1 soc-Slashdot0902 email-EuAll \
    web-Google wiki-Talk wiki-topcats soc-Pokec soc-LiveJournal1

python datasets/create_dataset.py --datasets \
    p2p-Gnutella31 soc-Epinions1 soc-Slashdot0902 email-EuAll \
    web-Google wiki-Talk wiki-topcats soc-Pokec soc-LiveJournal1
```

Exact betweenness computation is expensive for the largest graphs. To prepare only a smaller evaluation subset, pass the same subset of names to both commands and later use `--test_graphs`.

Use `python datasets/download.py --calibration` to also download the ten reference networks used to parameterize the synthetic hyperbolic graphs and support the appendix analyses.

### Synthetic training set

The paper trains on 30 synthetic graphs, each with 100,000 nodes:

- 10 directed scale-free graphs (`SF_10_Dir`);
- 10 undirected scale-free graphs, stored as symmetric directed graphs (`SF_10_Sym`);
- 10 uniformly directed hyperbolic random graphs (`HY_10_Dir`, denoted UDHY in the paper).

Generate the graphs, exact labels, and training splits with:

```bash
python datasets/generate_graph.py --datasets SF_10_Dir SF_10_Sym HY_10_Dir
python datasets/create_dataset.py --datasets SF_10_Dir SF_10_Sym HY_10_Dir
```

## Training and evaluation

Run the final paper configuration with:

```bash
python betweenness.py \
    --init_type degree_mix_mass_6 \
    --num_layers 2 \
    --nhid 12 \
    --dropout 0.3 \
    --train_type SF_10_Dir+SF_10_Sym+HY_10_Dir \
    --seed 1 \
    --run_all_tests
```

The paper trains for 10 epochs with Adam and a learning rate of $5 \times 10^{-3}$; these are the code defaults. Repeat the run with seeds 1, 2, and 3 to reproduce the reported mean and standard deviation.

Useful options:

- `--run_all_tests` evaluates all 14 paper graphs; without it, the code uses a six-graph default subset.
- `--test_graphs GRAPH [GRAPH ...]` evaluates only the named graphs.
- `--top_k` also computes Top-1%, Top-5%, and Top-10% accuracy.
- `--no_preprocessing` disables the shortest-path preprocessing heuristic.
- `--dump_predictions` saves per-node predictions, labels, and degrees as compressed NumPy files.

## Baselines

The paper compares BRAVA-GNN with three learning-based baselines (GNN-Bet, ABCDE, and DrBC) and three sampling-based baselines (KADABRA, SILVAN, and Bavarian).

### ABCDE and DrBC

Seeded ABCDE checkpoints and the DrBC checkpoint are included in `baselines/abcde/`. For example:

```bash
python baselines/eval_abcde.py \
    --model abcde \
    --model_path baselines/abcde/seeds/best_S1.ckpt \
    --seed 1 \
    --test_graphs p2p-Gnutella31 com-youtube amazon cit-Patents com-lj dblp

python baselines/eval_abcde.py \
    --model drbc \
    --seed 1 \
    --test_graphs p2p-Gnutella31 com-youtube amazon cit-Patents com-lj dblp
```

ABCDE and DrBC are evaluated only on the undirected graphs, matching their supported setting in the paper.

### GNN-Bet

After preparing its legacy `SF` training split under `datasets/data_splits/SF/betweenness/`, run:

```bash
python baselines/GNN-Bet/betweenness.py --g SF --seed 1
```

### Sampling-based methods

Compile the C++ implementations as described in their bundled READMEs, then run the evaluation wrappers:

```bash
python baselines/eval_kadabra.py
python baselines/eval_silvan.py
python baselines/eval_bavarian.py
```

The wrapper defaults match the paper: KADABRA uses $\epsilon=0.01$ and $\delta=0.1$; SILVAN uses $\epsilon=0.01$ and $\delta=0.05$; Bavarian uses the RK estimator with $\epsilon=0.01$, $\delta=0.1$, 100 Monte Carlo trials, and a progressive scaling factor of 2.

## Outputs and paper tables

Runs append their results to `results/betweenness/`:

- `all_results.csv` — Kendall $\tau_b$;
- `all_results_topk.csv` — Top-K accuracy;
- `all_results_wallclock.csv` — per-graph inference time;
- `all_results_training_time.csv` — total training time;
- `all_results_flops.csv` — inference GFLOPs or the sampling method's compute proxy.

Generate the paper tables from these CSV files with:

```bash
python scripts/parse_results.py --table main
python scripts/parse_results.py --table all
python scripts/generate_dataset_table.py
python scripts/calc_params.py
```

Wall-clock results are hardware-dependent. The paper used one NVIDIA Quadro RTX 8000 GPU per run on a machine with two Intel Xeon Gold 6230R CPUs and reports approximately 18 minutes for the complete BRAVA-GNN training process. BRAVA-GNN inference timings include computation of the degree-mass features.

## Citation

If you use BRAVA-GNN in your research, please cite:

```bibtex
@inproceedings{dachille2026degree,
  author    = {Justin Dachille and Aurora Rossi and Sunil Kumar Maurya and
               Frederik Mallmann-Trenn and Xin Liu and Fr{\'e}d{\'e}ric Giroire and
               Tsuyoshi Murata and Emanuele Natale},
  title     = {Degree-Mass Message Passing for Betweenness Ranking in Directed
               and Undirected Networks},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  numpages  = {12},
  location  = {Rome, Italy},
  doi       = {10.1145/3799682.3840622},
  url       = {https://doi.org/10.1145/3799682.3840622}
}
```
