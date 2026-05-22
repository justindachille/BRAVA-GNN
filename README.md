# BRAVA-GNN: Betweenness Ranking Approximation Via Degree-MAss Inspired Graph Neural Network
<a href="https://arxiv.org/abs/2602.09716"><img src="https://img.shields.io/badge/arXiv-2602.09716-b31b1b.svg"></a>

Official code release for **BRAVA-GNN**, a parameter-efficient GNN for approximating node betweenness centrality on large real-world networks. BRAVA-GNN improves Kendall–Tau ranking correlation by up to 214% and delivers up to 66× faster inference than prior GNN-based approaches.

---

## Setup

Create the virtual environment (uses [`uv`](https://github.com/astral-sh/uv); installed automatically if missing):

```bash
source setup.sh
```

This installs PyTorch 2.3.1 (CUDA 11.8), PyTorch Geometric, and the Python dependencies in `requirements.txt`. The ABCDE baseline is installed as an editable package.

---

## Datasets

All test graphs are sourced from [SNAP](https://snap.stanford.edu/data/) and the [ABCDE](https://github.com/MartinXPN/abcde) release.

Download the 14 paper test graphs:

```bash
python datasets/download.py
```

Add `--calibration` to also fetch the parameter-tuning graphs used in the appendix.

Generate the synthetic training graphs and the per-graph centrality pickles, then build the training splits used by the headline model:

```bash
python datasets/generate_graph.py  --datasets SF_10_Dir SF_10_Sym HY_10_Dir
python datasets/create_dataset.py  --datasets SF_10_Dir SF_10_Sym HY_10_Dir
python datasets/generate_graph.py  --datasets p2p-Gnutella31 soc-Epinions1 soc-Slashdot0902 \
                                              email-EuAll web-Google wiki-Talk wiki-topcats \
                                              soc-Pokec soc-LiveJournal1
python datasets/create_dataset.py  --datasets p2p-Gnutella31 soc-Epinions1 soc-Slashdot0902 \
                                              email-EuAll web-Google wiki-Talk wiki-topcats \
                                              soc-Pokec soc-LiveJournal1
```

The 5 ABCDE-release graphs (amazon, cit-Patents, com-lj, com-youtube, dblp) are imported separately:

```bash
python datasets/import_abcde_datasets.py
```

---

## Training & Evaluation

Headline model (the configuration used in the paper):

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

Notable flags:

- `--init_type` — `AW`, `degree_mix_mass_N`, `degree_mix_independent_N` (see `layer.py` for the full set).
- `--train_type` — combine training sets with `+`. SF/HY suffixes are explicit: `_Dir` (directed) or `_Sym` (symmetric). The 3-regime headline mix is `SF_10_Dir+SF_10_Sym+HY_10_Dir`.
- `--run_all_tests` — evaluate on all 14 paper test graphs; otherwise the small smoke set.
- `--top_k` — also compute Top-1%/5%/10% accuracy.
- `--no_preprocessing` — skip the clique-mask preprocessing.

Results are appended to CSV files under `results/betweenness/`:

- `all_results.csv` — Kendall-τ correlation
- `all_results_topk.csv` — Top-K accuracy (`--top_k` required)
- `all_results_wallclock.csv` — per-graph inference time
- `all_results_training_time.csv` — total training wallclock
- `all_results_flops.csv` — inference GFLOPs

---

## Baselines

Five baselines from the paper, vendored under `baselines/` with their original licenses preserved.

### ABCDE / DrBC

```bash
# Default 14-graph paper sweep
python baselines/eval_abcde.py --model abcde --seed 1
python baselines/eval_abcde.py --model drbc  --seed 1
```

The ABCDE/DrBC model checkpoints ship inside `baselines/abcde/`.

### GNN-Bet

```bash
python baselines/GNN-Bet/betweenness.py --g SF
```

### Bavarian (sampling-based)

Compile the C++ sources (see `baselines/Bavarian/README.md`), then:

```bash
python baselines/eval_bavarian.py
```

### SILVAN (sampling-based)

Compile per `baselines/SILVAN/README.md`, then:

```bash
python baselines/eval_silvan.py
```

### KADABRA (sampling-based)

Build the C++ binary (see `baselines/kadabra/README.md`), then:

```bash
python baselines/eval_kadabra.py
```

### Degree-mass heuristic (sanity baseline)

```bash
python baselines/eval_degree_mass.py
```

---

## Paper tables

After populating the CSVs in `results/betweenness/`, regenerate the LaTeX tables used in the paper:

```bash
python scripts/parse_results.py                # main results table
python scripts/parse_results.py --table 1      # ablation tables (--table 1 … 6)
python scripts/generate_dataset_table.py       # Table 11 (test graphs)
python scripts/calc_params.py                  # parameter-count comparison
```

---

## Citation

If you find BRAVA-GNN useful, please cite our paper:

```bibtex
@misc{dachille2026bravagnn,
      title={BRAVA-GNN: Betweenness Ranking Approximation Via Degree-MAss Inspired Graph Neural Network},
      author={Justin Dachille and Aurora Rossi and Sunil Kumar Maurya and Frederik Mallmann-Trenn and Xin Liu and Frédéric Giroire and Tsuyoshi Murata and Emanuele Natale},
      year={2026},
      eprint={2602.09716},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.09716}
}
```
