"""
CICIDS2017 NIDS — Multi-Variant Model Trainer
===============================================
Trains Supervised, Unsupervised & Hybrid models on:
  • reduced_pca.csv
  • reduced_kpca.csv
  • reduced_umap.csv
  • reduced_autoencoder.csv
  • feature_importance_variance.csv
  • raw / uncleaned baseline (for comparison)

Usage:
    python train_nids.py --data_dir "path/to/dim_reduction_results"

With raw data folder for comparison:
    python train_nids.py --data_dir "path/to/dim_reduction_results" --raw_dir "path/to/raw_csvs"

With simulated raw (noise added to PCA data):
    python train_nids.py --data_dir "path/to/dim_reduction_results" --simulate_raw

Output → saved_models/
    metrics.json, meta.json, model_*.pkl, scaler_*.pkl

Install:
    pip install pandas numpy scikit-learn flask flask-cors
"""

import os, glob, json, pickle, argparse, warnings, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── CSV variants we look for ───────────────────────────────────────────────
VARIANT_MAP = {
    "pca"       : ["reduced_pca.csv"],
    "kpca"      : ["reduced_kpca.csv"],
    "umap"      : ["reduced_umap.csv"],
    "autoencoder": ["reduced_autoencoder.csv"],
    "variance"  : ["feature_importance_variance.csv",
                   "reduced_feature_variance.csv"],
}

# ── PNG keys we expose ─────────────────────────────────────────────────────
PNG_KEYS = {
    "explained_variance" : "01_explained_variance.png",
    "pca_2d"             : "02_pca_2d.png",
    "kpca_2d"            : "02b_kpca_2d.png",
    "tsne_2d"            : "03_tsne_2d.png",
    "umap_2d"            : "04_umap_2d.png",
    "autoencoder_2d"     : "05_autoencoder_2d.png",
    "correlation_heatmap": "06_correlation_heatmap.png",
}

# ── Models ─────────────────────────────────────────────────────────────────
SUPERVISED = {
    "Decision Tree"      : lambda n: DecisionTreeClassifier(max_depth=10, random_state=SEED, class_weight="balanced"),
    "Random Forest"      : lambda n: RandomForestClassifier(n_estimators=80, max_depth=10, n_jobs=-1, random_state=SEED, class_weight="balanced"),
    "Logistic Regression": lambda n: LogisticRegression(max_iter=300, solver="saga", n_jobs=-1, random_state=SEED, class_weight="balanced"),
    "KNN"                : lambda n: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Gradient Boosting"  : lambda n: GradientBoostingClassifier(n_estimators=60, max_depth=4, random_state=SEED),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save(obj, key):
    with open(os.path.join(SAVE_DIR, f"{key}.pkl"), "wb") as f:
        pickle.dump(obj, f)

def _find(folder, names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.isfile(p):
            return p
    # fuzzy
    for f in glob.glob(os.path.join(folder, "*.csv")):
        for n in names:
            if os.path.basename(n).split(".")[0] in os.path.basename(f).lower():
                return f
    return None

def _prep(df, sample_n=20000):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("/", "_per_"))
    label_col = next((c for c in df.columns if "label" in c), None)
    if not label_col:
        return None, None, None, None, None

    df = df.dropna(subset=[label_col])

    if sample_n and len(df) > sample_n:
        df = (df.groupby(label_col, group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, int(sample_n * len(g) / len(df)))),
                    random_state=SEED))
                .reset_index(drop=True))

    drop = {"flow_id", "source_ip", "destination_ip", "timestamp"}
    feat = [c for c in df.select_dtypes(include=[np.number]).columns
            if c != label_col and c not in drop]
    X = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_raw = df[label_col].astype(str).str.strip()
    y_clean = y_raw.str.upper()
    y = np.where(y_clean == "BENIGN", 0, 1)
    dist = dict(y_raw.value_counts().head(10))
    return X, y, feat, label_col, dist


def _sup_met(name, cat, variant, dataset_type, model, X_te, y_te, elapsed):
    yp = model.predict(X_te)
    a = accuracy_score(y_te, yp)
    return dict(model=name, category=cat, variant=variant, dataset_type=dataset_type,
                accuracy=round(a,4),
                precision=round(precision_score(y_te,yp,zero_division=0,average="weighted"),4),
                recall=round(recall_score(y_te,yp,zero_division=0,average="weighted"),4),
                f1_score=round(f1_score(y_te,yp,zero_division=0,average="weighted"),4),
                error_rate=round(1-a,4), train_sec=round(elapsed,2))


def _uns_met(name, cat, variant, dataset_type, raw_preds, y_te, elapsed):
    yp = np.where(raw_preds == -1, 1, 0)
    a  = accuracy_score(y_te, yp)
    return dict(model=name, category=cat, variant=variant, dataset_type=dataset_type,
                accuracy=round(a,4),
                precision=round(precision_score(y_te,yp,zero_division=0),4),
                recall=round(recall_score(y_te,yp,zero_division=0),4),
                f1_score=round(f1_score(y_te,yp,zero_division=0),4),
                error_rate=round(1-a,4), train_sec=round(elapsed,2))


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE VARIANT
# ─────────────────────────────────────────────────────────────────────────────

def train_variant(X, y, tag, dataset_type, n_feat, results):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    _save(scaler, f"scaler_{tag}_{dataset_type}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.2, random_state=SEED, stratify=y)

    # ── Supervised ──
    for name, builder in SUPERVISED.items():
        t0 = time.time()
        m = builder(n_feat)
        m.fit(X_tr, y_tr)
        elapsed = time.time()-t0
        met = _sup_met(name, "Supervised", tag, dataset_type, m, X_te, y_te, elapsed)
        results.append(met)
        _save(m, f"{tag}__{dataset_type}__{name.replace(' ','_')}")
        print(f"      [{name:<22}] Acc={met['accuracy']*100:.1f}%  F1={met['f1_score']*100:.1f}%  ({elapsed:.1f}s)")

    # ── Unsupervised ──
    # Isolation Forest
    t0 = time.time()
    iso = IsolationForest(n_estimators=80, contamination=0.1, random_state=SEED, n_jobs=-1)
    iso.fit(X_tr)
    elapsed = time.time()-t0
    met = _uns_met("Isolation Forest", "Unsupervised", tag, dataset_type, iso.predict(X_te), y_te, elapsed)
    results.append(met)
    _save(iso, f"{tag}__{dataset_type}__Isolation_Forest")
    print(f"      [{'Isolation Forest':<22}] Acc={met['accuracy']*100:.1f}%  F1={met['f1_score']*100:.1f}%  ({elapsed:.1f}s)")

    # One-Class SVM (small sample)
    t0 = time.time()
    sub = min(4000, len(X_tr))
    ocs = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    ocs.fit(X_tr[:sub])
    elapsed = time.time()-t0
    met = _uns_met("One-Class SVM", "Unsupervised", tag, dataset_type, ocs.predict(X_te), y_te, elapsed)
    results.append(met)
    _save(ocs, f"{tag}__{dataset_type}__One-Class_SVM")
    print(f"      [{'One-Class SVM':<22}] Acc={met['accuracy']*100:.1f}%  F1={met['f1_score']*100:.1f}%  ({elapsed:.1f}s)")

    # ── Hybrid Pipelines ──
    n_pca = min(15, X_tr.shape[1])

    # PCA + Random Forest
    t0 = time.time()
    pipe = Pipeline([("pca", PCA(n_components=n_pca, random_state=SEED)),
                     ("clf", RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=-1, random_state=SEED, class_weight="balanced"))])
    pipe.fit(X_tr, y_tr)
    elapsed = time.time()-t0
    met = _sup_met("PCA + Random Forest", "Hybrid", tag, dataset_type, pipe, X_te, y_te, elapsed)
    results.append(met)
    _save(pipe, f"{tag}__{dataset_type}__PCA_RF")
    print(f"      [{'PCA + Random Forest':<22}] Acc={met['accuracy']*100:.1f}%  F1={met['f1_score']*100:.1f}%  ({elapsed:.1f}s)")

    # PCA + Decision Tree
    t0 = time.time()
    pipe2 = Pipeline([("pca", PCA(n_components=n_pca, random_state=SEED)),
                      ("clf", DecisionTreeClassifier(max_depth=8, random_state=SEED, class_weight="balanced"))])
    pipe2.fit(X_tr, y_tr)
    elapsed = time.time()-t0
    met = _sup_met("PCA + Decision Tree", "Hybrid", tag, dataset_type, pipe2, X_te, y_te, elapsed)
    results.append(met)
    _save(pipe2, f"{tag}__{dataset_type}__PCA_DT")
    print(f"      [{'PCA + Decision Tree':<22}] Acc={met['accuracy']*100:.1f}%  F1={met['f1_score']*100:.1f}%  ({elapsed:.1f}s)")

    # IsoForest + RF (anomaly flag as feature)
    t0 = time.time()
    iso2 = IsolationForest(n_estimators=60, contamination=0.1, random_state=SEED, n_jobs=-1)
    iso2.fit(X_tr)
    f_tr = (iso2.predict(X_tr)==-1).astype(int).reshape(-1,1)
    f_te = (iso2.predict(X_te)==-1).astype(int).reshape(-1,1)
    rf2  = RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=-1, random_state=SEED, class_weight="balanced")
    rf2.fit(np.hstack([X_tr,f_tr]), y_tr)
    elapsed = time.time()-t0
    met = _sup_met("IsoForest + RF", "Hybrid", tag, dataset_type, rf2,
                   np.hstack([X_te,f_te]), y_te, elapsed)
    results.append(met)
    _save((iso2,rf2), f"{tag}__{dataset_type}__IsoForest_RF")
    print(f"      [{'IsoForest + RF':<22}] Acc={met['accuracy']*100:.1f}%  F1={met['f1_score']*100:.1f}%  ({elapsed:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE RAW (add noise to first available variant)
# ─────────────────────────────────────────────────────────────────────────────

def make_raw(df_clean):
    df = df_clean.copy()
    num = df.select_dtypes(include=[np.number]).columns
    noise = np.random.normal(0, df[num].std().mean()*0.35, df[num].shape)
    df[num] = df[num] + noise
    for col in np.random.choice(list(num), size=min(8,len(num)), replace=False):
        idx = np.random.choice(len(df), size=int(len(df)*0.05), replace=False)
        df.loc[idx, col] = np.nan
    for col in np.random.choice(list(num), size=min(4,len(num)), replace=False):
        idx = np.random.choice(len(df), size=int(len(df)*0.02), replace=False)
        df.loc[idx, col] = df[col].max()*8
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT PNGs
# ─────────────────────────────────────────────────────────────────────────────

def collect_pngs(data_dir):
    png_map = {}
    for key, fname in PNG_KEYS.items():
        p = os.path.join(data_dir, fname)
        if os.path.isfile(p):
            png_map[key] = p
    # pick up any extra PNGs
    for f in glob.glob(os.path.join(data_dir, "*.png")):
        bname = os.path.basename(f).replace(".png","").lower().replace(" ","_")
        existing = {os.path.basename(v).replace(".png","") for v in png_map.values()}
        if os.path.basename(f).replace(".png","") not in existing:
            png_map[bname] = f
    return png_map


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir, raw_dir=None, simulate_raw=False, sample_n=20000):
    print("="*65)
    print("  CICIDS2017 NIDS — MULTI-VARIANT MODEL TRAINER")
    print("="*65)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Folder not found: {data_dir}")

    results = []
    meta    = {"variants": {}, "raw": None}
    first_df = None  # kept for simulate_raw

    # ── Train on each CSV variant ─────────────────────────────────────────
    for vname, fnames in VARIANT_MAP.items():
        path = _find(data_dir, fnames)
        if not path:
            print(f"\n  [SKIP] {vname} — file not found ({fnames})")
            continue

        print(f"\n{'─'*65}")
        print(f"  VARIANT: {vname.upper()}  →  {os.path.basename(path)}")
        print(f"{'─'*65}")

        df = pd.read_csv(path, low_memory=False)
        X, y, feat, lc, dist = _prep(df, sample_n)
        if X is None:
            print(f"  [SKIP] No label column in {vname}")
            continue

        print(f"  Shape: {X.shape}  |  Attack%: {y.mean()*100:.1f}%")
        meta["variants"][vname] = {
            "file": os.path.basename(path),
            "n_samples": len(X), "n_features": len(feat),
            "attack_pct": round(y.mean()*100, 2), "class_dist": dist
        }

        print(f"\n  [Cleaned variant: {vname}]")
        train_variant(X, y, vname, "cleaned", len(feat), results)

        if first_df is None:
            first_df = df  # save for simulate_raw

    if not results:
        raise RuntimeError("No variants were processed. Check your data_dir path.")

    # ── Train on RAW / Uncleaned baseline ─────────────────────────────────
    df_raw = None
    if raw_dir and os.path.isdir(raw_dir):
        print(f"\n{'─'*65}")
        print(f"  VARIANT: RAW  →  {raw_dir}")
        print(f"{'─'*65}")
        raw_files = glob.glob(os.path.join(raw_dir, "*.csv"))
        if raw_files:
            df_raw = pd.concat([pd.read_csv(f, low_memory=False) for f in raw_files], ignore_index=True)
    elif simulate_raw and first_df is not None:
        print(f"\n{'─'*65}")
        print(f"  VARIANT: RAW (simulated noise on PCA/first variant)")
        print(f"{'─'*65}")
        df_raw = make_raw(first_df)

    if df_raw is not None:
        X_r, y_r, feat_r, lc_r, dist_r = _prep(df_raw, sample_n)
        if X_r is not None:
            print(f"  Shape: {X_r.shape}  |  Attack%: {y_r.mean()*100:.1f}%")
            meta["raw"] = {"n_samples": len(X_r), "n_features": len(feat_r),
                           "attack_pct": round(y_r.mean()*100,2), "class_dist": dist_r}
            train_variant(X_r, y_r, "raw", "raw", len(feat_r), results)

    # ── PNGs ─────────────────────────────────────────────────────────────
    png_map = collect_pngs(data_dir)
    meta["png_map"] = png_map
    print(f"\n  Found {len(png_map)} PNG image(s): {list(png_map.keys())}")

    # ── Save ──────────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
        def convert(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)
        with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=convert)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  {'Model':<24} {'Category':<14} {'Variant':<12} {'DsType':<9} {'Acc':>7} {'F1':>7}")
    print("  "+"-"*65)
    for m in sorted(results, key=lambda x: (-x['accuracy'])):
        print(f"  {m['model']:<24} {m['category']:<14} {m['variant']:<12} {m['dataset_type']:<9} "
              f"{m['accuracy']*100:>6.1f}% {m['f1_score']*100:>6.1f}%")
    print("="*70)
    print(f"\n[+] {len(results)} results saved → {SAVE_DIR}/")
    print("[+] Run:  python app.py  →  http://localhost:5000")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True, help="dim_reduction_results folder.")
    p.add_argument("--raw_dir",     default=None,  help="Folder with raw CICIDS2017 CSVs.")
    p.add_argument("--simulate_raw",action="store_true", help="Simulate raw by adding noise.")
    p.add_argument("--sample",      type=int, default=20000, help="Rows per variant (default 20000).")
    args = p.parse_args()
    run(args.data_dir, args.raw_dir, args.simulate_raw,
        args.sample if args.sample > 0 else None)