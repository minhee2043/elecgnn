"""Command-line entry point: ``elecgnn train`` and ``elecgnn compare``."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from .data import (
    attach_scalars,
    concat_tables,
    load_ioncon,
    load_lce,
    load_oedb,
    load_oedb_task,
    scalar_lookup,
)
from .features import FEATURE_SETS, OEDBPredictor, attach_features, lce_features
from .featurize import GraphCache, node_dim
from .metrics import compare_runs
from .models import MixtureModel
from .train import (
    Split,
    TrainConfig,
    config_dict,
    original_kfold,
    original_lce_split,
    pretrain_dipole,
    pretrain_molecular,
    pretrain_oedb,
    repeated_kfold,
    run_cv,
    splits_from_column,
)

log = logging.getLogger("elecgnn")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="elecgnn", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="cross-validate one model configuration")
    t.add_argument("--task", choices=["ioncon", "lce", "oedb"], required=True)
    t.add_argument("--oedb-target", default="log_sigma",
                   help="target for --task oedb (e.g. log_sigma, log_eta, CN_anion)")  # fmt: skip
    t.add_argument("--data", type=Path, required=True,
                   help="unified_md.csv (ioncon) or Ref_LCEdata.xlsx (lce)")  # fmt: skip
    t.add_argument("--scalar-db", type=Path, help="cleaned in-house CSV with Solv_Ener and Dielec")
    t.add_argument("--model", choices=["gat", "gcn"], default="gat")
    t.add_argument("--scalars", choices=["none", "gsol", "dielec", "both"], default="none")
    t.add_argument("--share-encoder", action="store_true", help="one encoder for all slots")
    t.add_argument("--pooling", choices=["mean", "sum"], default="mean", help="sum keeps molecular size")
    # "thesis" is a legacy alias of "standard", kept so that old run.json files still load
    t.add_argument("--atom-features", choices=["standard", "thesis", "extended"], default="standard",
                   help="extended adds degree, attached H, formal charge, smallest ring size")  # fmt: skip
    splits = ["original", "thesis", "file", "repeated", "external", "group-solvent"]
    t.add_argument("--split", choices=splits, default="file",
                   help="repeated k-fold, or external (--test-file)")  # fmt: skip
    t.add_argument("--n-repeats", type=int, default=5, help="repeats for --split repeated")
    t.add_argument("--test-file", type=Path, help="external LCE test set (same layout as --data)")
    t.add_argument("--save-models", action="store_true", help="save every trained model (for interpretation)")
    t.add_argument("--folds-file", type=Path, help="CSV with a row_id column and fold columns")
    t.add_argument("--fold-col", help="fold column to use with --split file")
    t.add_argument("--test-folds", nargs="*", help="evaluate only these fold values (e.g. 0)")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--transform", choices=["identity", "log1p", "log10", "log1p_S_per_m"])
    t.add_argument("--pretrain", choices=["none", "dipole", "molecular", "oedb"], default="none",
                   help="dipole: original protocol; molecular: standardized --pretrain-targets")  # fmt: skip
    t.add_argument("--pretrain-targets", nargs="+", default=["HOMO", "LUMO", "GAP", "ELECTRIC MOMENT"],
                   help="columns of the molecular pretraining file (with --pretrain molecular)")  # fmt: skip
    t.add_argument("--pretrain-data", type=Path,
                   help="HOMO_LUMO_data.xlsx (dipole) or unified_md.csv (oedb)")  # fmt: skip
    t.add_argument("--oedb-cations", nargs="+", default=["Li"])
    t.add_argument("--pretrain-epochs", type=int, default=1000)
    t.add_argument("--early-stop", choices=["inner", "outer"], default="inner")
    t.add_argument(
        "--train-frac", type=float, default=1.0, help="learning curve: keep this training fraction"
    )
    t.add_argument("--add-back", type=float, default=0.0,
                   help="learning curve: move this fraction of each test fold into training")  # fmt: skip
    t.add_argument("--hidden", type=int, default=64, help="encoder channels (per head for GAT)")
    t.add_argument("--head-hidden", type=int, default=128, help="width of the first prediction-head layer")
    t.add_argument("--layers", type=int, default=4, help="message-passing layers before the output layer")
    t.add_argument("--dropout", type=float, default=0.2, help="dropout in the prediction head")
    t.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    t.add_argument("--epochs", type=int, default=5000)
    t.add_argument("--patience", type=int, default=500)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--device", default="auto")
    t.add_argument("--extra-features", type=Path, help="CSV from `elecgnn oedb-features` (row_id + columns)")
    t.add_argument(
        "--feature-set", choices=[*sorted(FEATURE_SETS), "file"], help="columns of --extra-features to use"
    )
    t.add_argument("--out", type=Path, required=True)

    f = sub.add_parser("oedb-features", help="train an OEDB property predictor and featurize task rows")
    f.add_argument("--unified", type=Path, required=True, help="unified_md.csv")
    f.add_argument("--lce", type=Path, help="Ref_LCEdata.xlsx (optional)")
    f.add_argument("--lce-test", type=Path, help="external LCE test set (optional; row ids lce_test_*)")
    f.add_argument("--model", choices=["gat", "gcn"], default="gat")
    f.add_argument("--n-models", type=int, default=5, help="ensemble size")
    f.add_argument("--oedb-cations", nargs="+", default=["Li"])
    f.add_argument("--epochs", type=int, default=1000)
    f.add_argument("--patience", type=int, default=100)
    f.add_argument("--batch-size", type=int, default=128)
    f.add_argument("--device", default="auto")
    f.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("raman", help="Raman coordinated-anion fractions vs MD and the OEDB predictor")
    r.add_argument("--raman", type=Path, required=True, help="oedb-experimental CSV (Raman spectra)")
    r.add_argument("--md", type=Path, required=True, help="oedb-electrolytes CSV (MD, with coordination)")
    r.add_argument("--unified", type=Path, help="unified_md.csv; needed with --predict")
    r.add_argument(
        "--predict", action="store_true", help="train the OEDB predictor and predict every Raman system"
    )
    r.add_argument("--model", choices=["gat", "gcn"], default="gat")
    r.add_argument("--n-models", type=int, default=5)
    r.add_argument("--epochs", type=int, default=300)
    r.add_argument("--patience", type=int, default=30)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--device", default="auto")
    r.add_argument("--out", type=Path, default=Path("results/raman"))
    r.add_argument("--summary-only", action="store_true",
                   help="recompute raman_summary.csv from raman_fractions.csv only")  # fmt: skip

    i = sub.add_parser("interpret", help="attributions, attention and embedding controls for a saved run")
    i.add_argument("run", type=Path, help="run directory trained with --save-models")
    i.add_argument("--out", type=Path, help="output directory (default: <run>/interpret)")
    i.add_argument("--seeds", type=int, nargs="*", help="only models with these seeds (default: all)")
    i.add_argument("--splits", help="only models whose split name matches this glob (e.g. 'r0k*')")
    i.add_argument("--ig-steps", type=int, default=32)
    i.add_argument("--device", default="auto")
    i.add_argument("--embeddings-only", action="store_true", help="skip attributions; only embedding checks")

    c = sub.add_parser("compare", help="paired corrected t-test between two runs")
    c.add_argument("run_a", type=Path)
    c.add_argument("run_b", type=Path)
    c.add_argument("--metric", default="MAE",
                   choices=["MAE", "RMSE", "R2", "Spearman", "MAE_log10", "median_fold_error"])  # fmt: skip
    return p


def _load_table(args):
    if args.task == "oedb":
        if args.pretrain == "oedb":
            raise SystemExit("--pretrain oedb with --task oedb would train on the test data")
        return load_oedb_task(args.data, getattr(args, "oedb_target", "log_sigma"),
                              tuple(getattr(args, "oedb_cations", ["Li"])))  # fmt: skip
    table = load_ioncon(args.data) if args.task == "ioncon" else load_lce(args.data)
    if args.split == "external":
        if args.task != "lce" or args.test_file is None:
            raise SystemExit("--split external needs --task lce and --test-file")
        table = concat_tables(table, load_lce(args.test_file, prefix="lce_test"))
    if args.scalars != "none":
        if args.scalar_db is None:
            raise SystemExit("--scalars requires --scalar-db")
        table = attach_scalars(table, scalar_lookup(args.scalar_db), args.scalars)
        real = table.conc > 0
        log.info("scalar coverage: %d / %d non-padding components have real values",
                 int(table.scalar_found[real].sum()), int(real.sum()))  # fmt: skip
    if args.extra_features is not None:
        if args.feature_set is None:
            raise SystemExit("--extra-features requires --feature-set")
        table = attach_features(table, pd.read_csv(args.extra_features), args.feature_set)
        log.info("extra features (%s): %s", args.feature_set, ", ".join(table.extra_names))
    return table


def _splits(args, table):
    if args.split == "group-solvent":
        groups = table.meta["solvent_smiles"].to_numpy()
        parts = GroupKFold(5).split(np.zeros(len(groups)), groups=groups)
        return [Split(f"solvent_group={k}", tr, te) for k, (tr, te) in enumerate(parts)], None
    if args.split == "repeated":
        return repeated_kfold(len(table), n_repeats=args.n_repeats), None
    if args.split == "external":
        is_test = np.char.startswith(table.row_id.astype(str), "lce_test_")
        return [Split("external", np.where(~is_test)[0], np.where(is_test)[0])], None
    if args.split == "inhouse":
        if args.task == "lce":
            return original_lce_split(len(table))
        return original_kfold(len(table)), None
    if args.folds_file is None or args.fold_col is None:
        raise SystemExit("--split file requires --folds-file and --fold-col")
    folds = pd.read_csv(args.folds_file).set_index("row_id")[args.fold_col]
    folds = folds.reindex(table.row_id)
    if folds.isna().any():
        raise SystemExit(f"{int(folds.isna().sum())} rows have no fold in {args.folds_file}")
    tv = None if args.test_folds is None else [type(folds.iloc[0])(v) for v in args.test_folds]
    return splits_from_column(folds.rename(args.fold_col), tv), None


def _encoder_state(args, cfg: TrainConfig, out: Path):
    if args.pretrain == "none":
        return None
    if args.pretrain_data is None:
        raise SystemExit("--pretrain requires --pretrain-data")
    if args.pretrain in ("dipole", "molecular"):
        df = pd.read_excel(args.pretrain_data)
        if {"HOMO", "LUMO", "GAP"} <= set(df.columns):  # drop rows whose gap is inconsistent or negative
            bad = (df["GAP"] <= 0) | ((df["GAP"] - (df["LUMO"] - df["HOMO"])).abs() > 0.01)
            if bad.any():
                log.info("dropping %d pretraining rows with inconsistent HOMO/LUMO/GAP", int(bad.sum()))
            df = df[~bad]
        if args.pretrain == "dipole":
            state = pretrain_dipole(df["SMILES"].tolist(), df["ELECTRIC MOMENT"].to_numpy(),
                                    args.model, epochs=args.pretrain_epochs, device=args.device)  # fmt: skip
        else:
            y = df[args.pretrain_targets].to_numpy()
            state = pretrain_molecular(df["SMILES"].tolist(), y, args.model, epochs=args.pretrain_epochs,
                                       device=args.device)  # fmt: skip
    else:
        oedb = load_oedb(args.pretrain_data, cations=tuple(args.oedb_cations))
        pcfg = TrainConfig(epochs=args.pretrain_epochs, patience=min(cfg.patience, 100),
                           batch_size=cfg.batch_size, device=cfg.device)  # fmt: skip
        state, scaling = pretrain_oedb(oedb, args.model, pcfg)
        (out / "oedb_scaling.json").write_text(json.dumps(scaling, indent=2))
    tag = (
        args.pretrain
        if args.pretrain != "molecular"
        else "mol_" + "_".join(t.replace(" ", "") for t in args.pretrain_targets)
    )
    torch.save(state, out / f"pretrained_{tag}_{args.model}.pt")
    return state


def cmd_train(args) -> None:
    extended = args.atom_features == "extended"
    if extended and args.pretrain != "none":
        raise SystemExit("--atom-features extended cannot reuse pretrained encoders (different input size)")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                      early_stop=args.early_stop, device=args.device, lr=args.lr)  # fmt: skip
    table = _load_table(args)
    splits, holdout = _splits(args, table)
    transform = args.transform or ("log10" if args.task == "ioncon" else "identity")
    if args.task == "oedb" and args.split not in ("group-solvent", "repeated"):
        raise SystemExit("--task oedb supports --split group-solvent or repeated")
    state = _encoder_state(args, cfg, args.out)
    make_model = partial(MixtureModel, args.model, table.n_slots, hidden1=args.hidden,
                         hidden2=args.head_hidden, scalar_dim=table.scalar_dim,
                         share_encoder=args.share_encoder, n_extra=table.n_extra,
                         pooling=args.pooling, node_dim=node_dim(extended), num_layers=args.layers,
                         dropout=args.dropout)  # fmt: skip
    cache = GraphCache(extended=extended)
    res = run_cv(table, splits, make_model, cfg, args.seeds, transform, state, holdout, cache,
                 train_frac=args.train_frac, add_back=args.add_back,
                 save_dir=args.out / "models" if args.save_models else None)  # fmt: skip
    for name, df in res.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
    fm = res["fold_metrics"]
    summary = {m: {"mean": float(fm[m].mean()), "std": float(fm[m].std())}
               for m in ["MAE", "RMSE", "R2", "Spearman", "MAE_log10", "median_fold_error", "bias_log10"]
               if m in fm}  # fmt: skip
    run_info = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "train_config": config_dict(cfg),
                "transform": transform, "summary": summary}  # fmt: skip
    if "holdout" in res:
        run_info["holdout"] = res["holdout"].iloc[0].to_dict()
    (args.out / "run.json").write_text(json.dumps(run_info, indent=2, default=str))
    log.info("summary: %s", json.dumps(summary))


def cmd_oedb_features(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                      device=args.device)  # fmt: skip
    cache = GraphCache()
    oedb = load_oedb(args.unified, cations=tuple(args.oedb_cations))
    log.info("training OEDB predictor on %d rows", len(oedb))
    pred = OEDBPredictor(args.model, args.n_models).fit(oedb, cfg, cache)
    pred.oof_metrics.to_csv(args.out / "oedb_predictor_heldout_metrics.csv", index=False)
    log.info("held-out accuracy on OEDB:\n%s",
             pred.oof_metrics.groupby("target")[["MAE", "R2"]].mean().round(3).to_string())  # fmt: skip
    ioncon = load_ioncon(args.unified)
    feats = pred.predict(ioncon, cfg, cache).assign(row_id=ioncon.row_id)
    feats.to_csv(args.out / "features_ioncon.csv", index=False)
    if args.lce is not None:
        lce = load_lce(args.lce)
        if args.lce_test is not None:
            lce = concat_tables(lce, load_lce(args.lce_test, prefix="lce_test"))
        lce_features(pred, lce, cfg, cache).to_csv(args.out / "features_lce.csv", index=False)
    log.info("wrote features to %s", args.out)


def cmd_raman(args) -> None:
    from .raman import analyse, composition_table, summarise

    args.out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        df = pd.read_csv(args.out / "raman_fractions.csv")
        pred = "CN_anion_pred" if "CN_anion_pred" in df else None
        summary = summarise(df, pred)
        summary.to_csv(args.out / "raman_summary.csv", index=False)
        table = summary[summary.group == "pooled"].pivot_table(index="anion", columns="comparison",
                                                               values="spearman")  # fmt: skip
        log.info(
            "agreement with the Raman coordinated-anion fraction (Spearman):\n%s", table.round(2).to_string()
        )
        return
    df = analyse(args.raman)
    md = pd.read_csv(args.md)
    link = md[["ID", "Coordination (Cation←Anion)", "Coordination (Cation ← Solvent)"]]
    link.columns = ["cmp_id", "CN_anion_MD", "CN_solvent_MD"]
    df = df.merge(link, on="cmp_id", how="left")
    df["solvent_in_oedb"] = df.solvent.isin(set(md["Solvent"]))
    log.info("fitted %d salt spectra (%d linked to MD; %d flagged for solvent overlap)", len(df),
             df.CN_anion_MD.notna().sum(), df.solvent_overlap.sum())  # fmt: skip
    pred_col = None
    if args.predict:
        if args.unified is None:
            raise SystemExit("--predict needs --unified")
        cfg = TrainConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                          device=args.device)  # fmt: skip
        cache = GraphCache()
        predictor = OEDBPredictor(args.model, args.n_models).fit(load_oedb(args.unified), cfg, cache)
        table, comp = composition_table(df, args.md)
        pred = predictor.predict(table, cfg, cache)
        comp = pd.concat(
            [
                comp,
                pred[["CN_anion", "CN_solvent"]].rename(
                    columns={"CN_anion": "CN_anion_pred", "CN_solvent": "CN_solvent_pred"}
                ),
            ],
            axis=1,
        )
        comp.to_csv(args.out / "predicted_coordination.csv", index=False)
        df = df.merge(comp[["anion", "solvent", "molality", "CN_anion_pred", "CN_solvent_pred"]],
                      on=["anion", "solvent", "molality"], how="left")  # fmt: skip
        pred_col = "CN_anion_pred"
    df.to_csv(args.out / "raman_fractions.csv", index=False)
    summary = summarise(df, pred_col)
    summary.to_csv(args.out / "raman_summary.csv", index=False)
    log.info("agreement with the Raman coordinated-anion fraction (Spearman):\n%s",
             summary.pivot_table(index="anion", columns=["comparison", "group"], values="spearman").round(2)
             .to_string())  # fmt: skip


PATH_ARGS = {"data", "scalar_db", "folds_file", "test_file", "extra_features", "pretrain_data", "out"}


def cmd_interpret(args) -> None:
    from types import SimpleNamespace

    from .data import MixtureDataset
    from .interpret import atom_attributions, descriptors, knn_r2, tsne
    from .train import TRANSFORMS

    info = json.loads((args.run / "run.json").read_text())
    ra = SimpleNamespace(**{k: (Path(v) if k in PATH_ARGS and v not in (None, "None") else v)
                            for k, v in info["args"].items()})  # fmt: skip
    out = args.out or args.run / "interpret"
    out.mkdir(parents=True, exist_ok=True)
    table = _load_table(ra)
    fwd, _ = TRANSFORMS[info["transform"]]
    extended = getattr(ra, "atom_features", "standard") == "extended"
    ds = MixtureDataset(table, fwd(table.y.astype(np.float64)), GraphCache(extended=extended))
    slot_names = ["solvent", "anion"] if ra.task == "ioncon" else [f"c{j + 1}" for j in range(table.n_slots)]

    def build() -> MixtureModel:
        return MixtureModel(ra.model, table.n_slots, hidden1=getattr(ra, "hidden", 64),
                            hidden2=getattr(ra, "head_hidden", 128), scalar_dim=table.scalar_dim,
                            share_encoder=ra.share_encoder, n_extra=table.n_extra,
                            pooling=getattr(ra, "pooling", "mean"), node_dim=node_dim(extended),
                            num_layers=getattr(ra, "layers", 4),
                            dropout=getattr(ra, "dropout", 0.2))  # fmt: skip

    from fnmatch import fnmatch

    device = TrainConfig(device=args.device).torch_device()
    files = sorted((args.run / "models").glob("seed*__*.pt"))
    if args.splits:
        files = [f for f in files if fnmatch(f.stem.split("__", 1)[1], args.splits)]
    if not files:
        raise SystemExit(f"no saved models in {args.run / 'models'}; train with --save-models")
    atoms, oof_desc, first_model = [], {}, None
    for f in files:
        seed, split = f.stem.split("__", 1)
        seed = int(seed[4:])
        if args.seeds and seed not in args.seeds:
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        model = build()
        model.load_state_dict(ck["state_dict"])
        first_model = first_model or (f.name, model)
        if not args.embeddings_only:
            atoms.append(
                atom_attributions(
                    model, ds, table, ck["test"], slot_names, seed, split, args.ig_steps, device
                )
            )
        for r, d in zip(ck["test"], descriptors(model, ds, ck["test"], device), strict=True):
            oof_desc.setdefault(int(r), d)
        log.info("processed %s", f.name)
    if first_model is None:
        raise SystemExit("no models matched --seeds/--splits")
    if atoms:
        a = pd.concat(atoms, ignore_index=True)
        a["role"] = np.where(a["smiles"].str.contains("Li"), "salt", "solvent")
        if ra.task == "ioncon":
            a["role"] = np.where(a["slot"] == "anion", "anion", "solvent")
        a.to_csv(out / "atom_attributions.csv", index=False)
        summary = a.groupby(["role", "group"]).agg(
            n=("ig", "size"),
            ig_mean=("ig", "mean"),
            ig_abs=("ig", lambda v: v.abs().mean()),
            attn_ratio=("attn_ratio", "mean"),
            entropy_ratio=("entropy_ratio", "mean"),
        )
        summary.round(4).to_csv(out / "attribution_summary.csv")

    # Embedding check: ONE trained model embeds every row, so all descriptors share one
    # coordinate system (descriptors from different fold models are not comparable).
    rows = np.arange(len(table))
    y = ds.y[rows, 0].numpy()
    trained = descriptors(first_model[1], ds, rows, device)
    torch.manual_seed(0)
    control = descriptors(build(), ds, rows, device)
    scores = {
        "n_rows": len(rows),
        "embedding_model": first_model[0],
        "knn_r2_trained": knn_r2(trained, y),
        "knn_r2_random_init": knn_r2(control, y),
    }
    (out / "embedding_scores.json").write_text(json.dumps(scores, indent=2))
    emb = pd.DataFrame({"row_id": table.row_id[rows], "y_transformed": y})
    emb[["tsne_trained_1", "tsne_trained_2"]] = tsne(trained)
    emb[["tsne_random_1", "tsne_random_2"]] = tsne(control)
    emb.merge(table.meta, on="row_id", how="left").to_csv(out / "embeddings.csv", index=False)
    log.info("embedding kNN R2: trained %.3f vs random-init %.3f", scores["knn_r2_trained"],
             scores["knn_r2_random_init"])  # fmt: skip


def cmd_compare(args) -> None:
    a = pd.read_csv(args.run_a / "fold_metrics.csv")
    b = pd.read_csv(args.run_b / "fold_metrics.csv")
    print(json.dumps(compare_runs(a, b, args.metric), indent=2))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    cmds = {"train": cmd_train, "oedb-features": cmd_oedb_features, "interpret": cmd_interpret,
            "raman": cmd_raman,
            "compare": cmd_compare}  # fmt: skip
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
