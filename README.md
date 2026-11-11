# elecgnn

Graph neural networks, descriptor models and evaluation tools for predicting liquid-electrolyte
properties from molecular dynamics (MD) and experimental data.

This repository accompanies the article *Coverage, not architecture: what limits
machine-learning extrapolation for liquid battery electrolytes* (submitted to Digital
Discovery). It contains everything needed to reproduce the analyses and figures:

- **Models:** GAT (GATv2) and GCN mixture models in which each component's graph embedding is
  weighted by its mole fraction; nine descriptor models (RF, AdaBoost, XGBoost, LightGBM, SVR,
  ridge, lasso, GPR, MLP); hybrids of the two.
- **External information:** OEDB pretraining, OEDB-predicted electrolyte properties as features,
  and molecular pretraining on HOMO, LUMO, gap and dipole moment.
- **Evaluation:** one shared fold file for all models (random, leave-solvent-out, leave-class-out,
  leave-ethers-out), held-out early stopping, and paired statistical comparisons over seeds and folds.
- **Analysis:** coverage vs error, integrated gradients, degree-corrected attention, embedding
  checks against a random-weight control, and a Raman-spectra check of MD solvation structure.

## Installation

Python 3.10 or later. Install PyTorch and PyTorch Geometric for your platform first
(see https://pytorch.org and https://pytorch-geometric.readthedocs.io), then:

```bash
git clone https://github.com/minhee2043/elecgnn.git
cd elecgnn
pip install -e ".[dev]"            # package, tests and linting
pip install -e ".[baselines]"      # LightGBM, XGBoost, SHAP for the descriptor models
pip install -e ".[figures]"        # matplotlib for the figures
pytest -q
```

Tested with Python 3.11, PyTorch 2.4.1 and PyTorch Geometric 2.5.3.

## Data

`data/` contains the cleaned in-house MD dataset (268 compositions, 31 solvents, three Li salts,
363 K), its cleaning log and the fold assignments used for every result. External datasets
(OEDB MD and Raman data, the Coulombic-efficiency data and the molecular-pretraining set) are
downloaded separately; see [`data/README.md`](data/README.md). Then:

```bash
python scripts/compare_oedb.py --base data             # OEDB QC flags and in-house/OEDB overlap
python scripts/build_unified_and_folds.py --base data  # data/unified_md.csv
python scripts/build_kim_test_set.py --out data/LCE_test13.csv
```

## Quick start

```bash
# GAT, ionic conductivity (log10 sigma), new solvents held out
elecgnn train --task ioncon --data data/unified_md.csv --model gat \
  --split file --folds-file data/folds_inhouse.csv --fold-col fold_solvent \
  --seeds 0 1 2 --out results/gat_new_solvents

# All ether-type solvents held out, ten seeds
elecgnn train --task ioncon --data data/unified_md.csv --model gat \
  --split file --folds-file data/folds_inhouse.csv --fold-col fold_ethers --test-folds 0 \
  --seeds 0 1 2 3 4 5 6 7 8 9 --out results/gat_ethers

# Paired comparison of two runs on matching (seed, split)
elecgnn compare results/gat_ethers results/gcn_ethers --metric MAE_log10
```

Each run directory contains `run.json` (settings and summary), `fold_metrics.csv` (one row per
seed and split) and `predictions.csv`.

## Commands

| Command | Purpose |
| --- | --- |
| `elecgnn train` | Train and evaluate GNN models (`--task ioncon`, `lce` or `oedb`) |
| `elecgnn compare` | Paired test between two runs: corrected resampled *t*-test across folds, paired *t*-test over seeds when only seeds vary |
| `elecgnn oedb-features` | Train the OEDB property ensemble and write its predictions as features |
| `elecgnn interpret` | Integrated gradients, degree-corrected attention and embedding checks for a run trained with `--save-models` |
| `elecgnn raman` | Fit Raman anion bands and compare the coordinated-anion fraction with MD and the OEDB predictor |

Main `train` options:

| Option | Values | Default |
| --- | --- | --- |
| `--model` | `gat`, `gcn` | `gat` |
| `--split` | `file` (fold file), `original` (original random protocol), `repeated`, `external`, `group-solvent` | `file` |
| `--early-stop` | `inner` (held-out part of the training fold), `outer` (test fold; original protocol only) | `inner` |
| `--transform` | target transform; `log10` for conductivity | task-dependent |
| `--pretrain` | `none`, `dipole`, `molecular` (with `--pretrain-targets`), `oedb` | `none` |
| `--extra-features`, `--feature-set` | add OEDB-derived features or any feature file (`file`) | none |
| `--pooling` | `mean`, `sum` | `mean` |
| `--atom-features` | `standard` (26 features), `extended` (+ degree, attached H, formal charge, ring size) | `standard` |
| `--seeds` | initialization seeds | `0 1 2` |
| `--hidden`, `--layers`, `--head-hidden`, `--dropout`, `--lr` | model size and training settings | `64`, `4`, `128`, `0.2`, `1e-3` |
| `--save-models` | keep trained models for `elecgnn interpret` | off |

## Scripts

| Script | Purpose |
| --- | --- |
| `clean_solvent_db.py` | Cleaning of the raw in-house table (produces `solvent_db_1_clean.csv` and the changelog) |
| `compare_oedb.py`, `build_unified_and_folds.py` | OEDB QC, in-house/OEDB overlap, unified dataset and folds |
| `build_kim_test_set.py`, `kim_linear_baseline.py` | External CE test set and the four-descriptor linear baseline |
| `descriptor_baselines.py`, `build_descriptor_features.py` | Descriptor models on the shared folds; descriptor features for hybrid models |
| `coverage_vs_error.py` | Error against similarity to the nearest training solvent |
| `reproduce_original_protocol.sh` | Reproduce the original random-split protocol |
| `run_oedb_features.sh`, `run_scalar_ablation.sh` | Feature and scalar-input studies |
| `make_figures.py` | All figures of the article and SI, regenerated from `results/` |
| `migrate_legacy_names.py` | Update files written by earlier versions of this code |
| `add_family_fold.py` | Add a family hold-out column (for example all carbonates) to the fold file |
| `make_subset.py` | Copy of the unified data without some solvent classes, for nested model selection |
| `chemprop_baseline.py` | Chemprop (D-MPNN) baseline on the shared folds, in the elecgnn run format |
| `external_ce_bootstrap.py` | Bootstrap confidence intervals for the external CE comparison |
| `md_conductivity_uncertainty.py` | Block-averaged Einstein conductivity and its statistical error from a GROMACS trajectory |

## Reproducing the article

The experiments are listed as one command per line in `scripts/slurm/jobs_*.txt`
(descriptor baselines, molecular pretraining, hybrids, graph ablation, Raman). On a SLURM
cluster, set the partition in `scripts/slurm/*.sbatch` and run, for example:

```bash
export ELECGNN_ENV=elecgnn        # conda environment name or path
mkdir -p logs
sbatch --array=1-4 scripts/slurm/cpu_array_job.sbatch scripts/slurm/jobs_descriptors.txt
sbatch --array=1-34%4 scripts/slurm/array_job.sbatch scripts/slurm/jobs_pretraining.txt
python scripts/make_figures.py --results results --data data --out figures
```

Without SLURM, run the lines of the job files directly. Out-of-domain comparisons on a single
held-out group use ten seeds; fewer seeds can produce false positives.

## Notes

- Early stopping on the test fold (`--early-stop outer`) is provided only to reproduce the
  original protocol; it is optimistically biased.
- The Coulombic-efficiency training file encodes fluorobenzene as fluorocyclohexane; this is
  documented but not changed, to keep results comparable with published work.
- Raman decomposition is reliable for FSI⁻, BF₄⁻, ClO₄⁻ and TfO⁻, not for TFSI⁻ and PF₆⁻.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT; see [`LICENSE`](LICENSE). The external datasets keep their original licenses.
