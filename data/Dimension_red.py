"""
Techniques included:
  1. Variance Threshold     — removes low-variance features
  2. Correlation Filter     — removes highly correlated features
  3. PCA                    — linear projection, saves explained variance plot
  4. Incremental PCA        — memory-efficient PCA for large datasets
  5. Kernel PCA             — non-linear projection (RBF kernel)
  6. t-SNE                  — 2D non-linear visualization
  7. UMAP                   — fast non-linear reduction (optional, if installed)
  8. Autoencoder Embeddings — deep non-linear reduction

Outputs saved to:  dim_reduction_results/
  ├── reduced_pca.csv              — PCA-reduced feature matrix + label
  ├── reduced_kpca.csv             — Kernel PCA reduced
  ├── reduced_umap.csv             — UMAP reduced (if available)
  ├── reduced_autoencoder.csv      — Autoencoder bottleneck embeddings
  ├── feature_importance_variance.csv
  ├── 01_explained_variance.png
  ├── 02_pca_2d.png
  ├── 03_tsne_2d.png
  ├── 04_umap_2d.png               — (if UMAP available)
  ├── 05_autoencoder_2d.png
  └── 06_correlation_heatmap.png

Usage:
    python cicids2017_dim_reduction.py --csv "path/to/cicids2017_cleaned.csv"

Example (Windows):
    python cicids2017_dim_reduction.py --csv "CSV FILE.csv" --sample 30000 --pca_components 20

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
    pip install umap-learn   # optional but recommended
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dim_reduction_results")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOAD CLEANED DATA
# =============================================================================

def load_cleaned(csv_path: str, sample_n: int = 30_000):
    print("\n[1] Loading cleaned data ...")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"File not found: {csv_path}\n"
            "Run cicids2017_cleaning.py first."
        )

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"    Loaded shape : {df.shape}")

    # Detect label column
    label_cols = [c for c in df.columns if "label" in c.lower()]
    if not label_cols:
        raise ValueError("No label column found.")
    label_col = label_cols[0]
    print(f"    Label column : '{label_col}'")

    # Stratified sample for speed
    if sample_n and len(df) > sample_n:
        df = (df.groupby(label_col, group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, int(sample_n * len(g) / len(df)))),
                    random_state=SEED))
                .reset_index(drop=True))
        print(f"    Sampled to   : {len(df):,} rows (stratified)")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    class_names = list(le.classes_)

    # Feature matrix (numeric only, exclude label)
    drop_cols = {"flow_id", "source_ip", "destination_ip", "timestamp", label_col}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in drop_cols]
    X = df[feature_cols].copy()

    print(f"    Features     : {len(feature_cols)}")
    print(f"    Classes      : {class_names}")
    return X, y, df[label_col], feature_cols, label_col, class_names


# =============================================================================
# 2. SCALE
# =============================================================================

def scale_features(X: pd.DataFrame):
    print("\n[2] Scaling features (StandardScaler) ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"    Scaled shape : {X_scaled.shape}")
    return X_scaled, scaler


# =============================================================================
# 3. VARIANCE THRESHOLD FILTER
# =============================================================================

def variance_threshold_filter(X_scaled, feature_cols, threshold=0.01):
    print(f"\n[3] Variance Threshold Filter (threshold={threshold}) ...")
    vt = VarianceThreshold(threshold=threshold)
    X_vt = vt.fit_transform(X_scaled)
    selected = [feature_cols[i] for i in range(len(feature_cols)) if vt.get_support()[i]]
    removed = len(feature_cols) - len(selected)
    print(f"    Features before : {len(feature_cols)}")
    print(f"    Features removed: {removed}")
    print(f"    Features after  : {len(selected)}")

    # Save variance info
    var_df = pd.DataFrame({
        "feature": feature_cols,
        "variance": vt.variances_,
        "selected": vt.get_support()
    }).sort_values("variance", ascending=False)
    var_df.to_csv(os.path.join(OUT_DIR, "feature_importance_variance.csv"), index=False)
    print(f"    Saved: feature_importance_variance.csv")
    return X_vt, selected


# =============================================================================
# 4. CORRELATION FILTER
# =============================================================================

def correlation_filter(X_vt, selected_cols, threshold=0.95):
    print(f"\n[4] Correlation Filter (threshold={threshold}) ...")
    df_temp = pd.DataFrame(X_vt, columns=selected_cols)
    corr_matrix = df_temp.corr().abs()

    # Upper triangle mask
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df_filtered = df_temp.drop(columns=to_drop)
    remaining_cols = list(df_filtered.columns)

    print(f"    Highly correlated features removed : {len(to_drop)}")
    print(f"    Features remaining                 : {len(remaining_cols)}")

    # Plot correlation heatmap (top 30 features for readability)
    top_cols = remaining_cols[:30]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(df_filtered[top_cols].corr(), cmap="coolwarm", center=0,
                linewidths=0.3, ax=ax, annot=False)
    ax.set_title("Correlation Heatmap (after filter, top 30 features)", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "06_correlation_heatmap.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: 06_correlation_heatmap.png")

    return df_filtered.values, remaining_cols


# =============================================================================
# 5. PCA
# =============================================================================

def run_pca(X_filtered, y, label_series, n_components=None, variance_target=0.95):
    print(f"\n[5] PCA ...")

    # ── Full PCA to find optimal n_components ────────────────────────────────
    max_comp = min(X_filtered.shape[0], X_filtered.shape[1])
    pca_full = PCA(n_components=max_comp, random_state=SEED)
    pca_full.fit(X_filtered)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is None:
        n_components = int(np.searchsorted(cum_var, variance_target) + 1)
        print(f"    Auto n_components for {variance_target*100:.0f}% variance: {n_components}")
    else:
        print(f"    Using n_components = {n_components}")

    # ── Final PCA ─────────────────────────────────────────────────────────────
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X_filtered)
    actual_var = pca.explained_variance_ratio_.sum()
    print(f"    Variance explained : {actual_var*100:.2f}%")
    print(f"    Reduced shape      : {X_pca.shape}")

    # ── Save reduced CSV ──────────────────────────────────────────────────────
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    df_pca.insert(0, "label", label_series.values)
    p = os.path.join(OUT_DIR, "reduced_pca.csv")
    df_pca.to_csv(p, index=False)
    print(f"    Saved: reduced_pca.csv")

    # ── Plot 1: Explained variance ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(1, len(pca_full.explained_variance_ratio_[:50]) + 1),
            pca_full.explained_variance_ratio_[:50], color="#5c8de0", alpha=0.8)
    ax1.set_title("PCA — Individual Explained Variance (top 50)", fontweight="bold")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.grid(alpha=0.3)

    ax2.plot(range(1, len(cum_var) + 1), cum_var, color="#e05c5c", lw=2)
    ax2.axhline(variance_target, color="gray", linestyle="--",
                label=f"{variance_target*100:.0f}% threshold")
    ax2.axvline(n_components, color="green", linestyle="--",
                label=f"n={n_components} components")
    ax2.set_title("PCA — Cumulative Explained Variance", fontweight="bold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "01_explained_variance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: 01_explained_variance.png")

    # ── Plot 2: PCA 2D scatter ────────────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=SEED)
    X_pca2 = pca2.fit_transform(X_filtered)
    _scatter_plot(X_pca2, y, "02_pca_2d.png",
                  f"PCA 2D  ({pca2.explained_variance_ratio_.sum()*100:.1f}% variance)")

    return X_pca, pca, n_components


# =============================================================================
# 6. INCREMENTAL PCA  (memory efficient for large datasets)
# =============================================================================

def run_incremental_pca(X_filtered, n_components, batch_size=1000):
    print(f"\n[6] Incremental PCA (n={n_components}, batch={batch_size}) ...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_ipca = ipca.fit_transform(X_filtered)
    var = ipca.explained_variance_ratio_.sum()
    print(f"    Variance explained : {var*100:.2f}%")
    print(f"    Reduced shape      : {X_ipca.shape}")
    return X_ipca


# =============================================================================
# 7. KERNEL PCA
# =============================================================================

def run_kernel_pca(X_filtered, y, label_series, n_components=2, kernel="rbf"):
    print(f"\n[7] Kernel PCA (kernel={kernel}, n={n_components}) ...")
    kpca = KernelPCA(n_components=n_components, kernel=kernel,
                     random_state=SEED, n_jobs=-1)
    X_kpca = kpca.fit_transform(X_filtered)
    print(f"    Reduced shape : {X_kpca.shape}")

    # Save CSV
    cols = [f"KPC{i+1}" for i in range(n_components)]
    df_kpca = pd.DataFrame(X_kpca, columns=cols)
    df_kpca.insert(0, "label", label_series.values)
    df_kpca.to_csv(os.path.join(OUT_DIR, "reduced_kpca.csv"), index=False)
    print(f"    Saved: reduced_kpca.csv")

    if n_components >= 2:
        _scatter_plot(X_kpca[:, :2], y, "02b_kpca_2d.png", f"Kernel PCA 2D ({kernel})")

    return X_kpca


# =============================================================================
# 8. t-SNE  (2D visualization only)
# =============================================================================

def run_tsne(X_pca, y, perplexity=40):
    print(f"\n[8] t-SNE 2D (perplexity={perplexity}) ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
            random_state=SEED, init='pca', learning_rate="auto")
    X_tsne = tsne.fit_transform(X_pca)
    print(f"    Reduced shape : {X_tsne.shape}")
    _scatter_plot(X_tsne, y, "03_tsne_2d.png", "t-SNE 2D")
    return X_tsne


# =============================================================================
# 9. UMAP  (optional)
# =============================================================================

def run_umap(X_pca, y, label_series, n_components=2):
    print(f"\n[9] UMAP (n={n_components}) ...")
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=SEED, n_jobs=-1)
        X_umap = reducer.fit_transform(X_pca)
        print(f"    Reduced shape : {X_umap.shape}")

        # Save CSV
        cols = [f"UMAP{i+1}" for i in range(n_components)]
        df_umap = pd.DataFrame(X_umap, columns=cols)
        df_umap.insert(0, "label", label_series.values)
        df_umap.to_csv(os.path.join(OUT_DIR, "reduced_umap.csv"), index=False)
        print(f"    Saved: reduced_umap.csv")

        if n_components >= 2:
            _scatter_plot(X_umap[:, :2], y, "04_umap_2d.png", "UMAP 2D")

        return X_umap
    except ImportError:
        print("    [!] umap-learn not installed — skipping UMAP.")
        print("        Install with: pip install umap-learn")
        return None


# =============================================================================
# 10. AUTOENCODER EMBEDDINGS
# =============================================================================

def run_autoencoder_reduction(X_filtered, y, label_series,
                               bottleneck_dim=8, epochs=30, batch_size=512):
    print(f"\n[10] Autoencoder Reduction (bottleneck={bottleneck_dim}) ...")
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, callbacks
        tf.random.set_seed(SEED)
    except ImportError:
        print("    [!] TensorFlow not installed — skipping Autoencoder.")
        print("        Install with: pip install tensorflow")
        return None

    n_features = X_filtered.shape[1]

    # ── Encoder ───────────────────────────────────────────────────────────────
    inp = tf.keras.Input(shape=(n_features,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    bottleneck = layers.Dense(bottleneck_dim, activation="relu",
                              name="bottleneck")(x)

    # ── Decoder ───────────────────────────────────────────────────────────────
    x = layers.Dense(64, activation="relu")(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(n_features, activation="linear")(x)

    ae = Model(inp, out, name="AE_DimReduction")
    encoder = Model(inp, bottleneck, name="Encoder")
    ae.compile(optimizer="adam", loss="mse")

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)
    ]

    print(f"    Training (epochs={epochs}, batch={batch_size}) ...")
    history = ae.fit(X_filtered, X_filtered,
                     epochs=epochs, batch_size=batch_size,
                     validation_split=0.1, callbacks=cb, verbose=0)

    X_ae = encoder.predict(X_filtered, batch_size=batch_size, verbose=0)
    final_loss = history.history["val_loss"][-1]
    print(f"    Bottleneck shape : {X_ae.shape}")
    print(f"    Final val_loss   : {final_loss:.6f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    cols = [f"AE{i+1}" for i in range(bottleneck_dim)]
    df_ae = pd.DataFrame(X_ae, columns=cols)
    df_ae.insert(0, "label", label_series.values)
    df_ae.to_csv(os.path.join(OUT_DIR, "reduced_autoencoder.csv"), index=False)
    print(f"    Saved: reduced_autoencoder.csv")

    # ── 2D visualization via t-SNE on bottleneck ─────────────────────────────
    if bottleneck_dim > 2:
        print("    Running t-SNE on bottleneck for 2D plot ...")
        tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=800,
    random_state=SEED,
    init="pca",
    learning_rate="auto"
)
        X_ae_2d = tsne.fit_transform(X_ae)
    else:
        X_ae_2d = X_ae[:, :2]

    _scatter_plot(X_ae_2d, y, "05_autoencoder_2d.png",
                  f"Autoencoder Embeddings (bottleneck={bottleneck_dim}) — t-SNE 2D")

    return X_ae


# =============================================================================
# HELPER: Scatter plot
# =============================================================================

def _scatter_plot(X2d, y, filename, title):
    unique = np.unique(y)
    cmap = plt.cm.get_cmap("tab10", len(unique))
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, lbl in enumerate(unique):
        mask = y == lbl
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   s=5, alpha=0.5, color=cmap(i),
                   label=str(lbl), rasterized=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if len(unique) <= 15:
        ax.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.7)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, filename)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {filename}")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(original_shape, feature_cols, selected_after_vt,
                  remaining_after_corr, n_pca_components):
    print("\n" + "=" * 60)
    print("  DIMENSIONALITY REDUCTION SUMMARY")
    print("=" * 60)
    print(f"  Original features          : {original_shape[1]}")
    print(f"  After variance threshold   : {len(selected_after_vt)}")
    print(f"  After correlation filter   : {len(remaining_after_corr)}")
    print(f"  PCA components (95% var)   : {n_pca_components}")
    print(f"\n  Saved files in: {OUT_DIR}/")
    print("  +-- reduced_pca.csv")
    print("  +-- reduced_kpca.csv")
    print("  +-- reduced_umap.csv          (if umap-learn installed)")
    print("  +-- reduced_autoencoder.csv   (if tensorflow installed)")
    print("  +-- feature_importance_variance.csv")
    print("  +-- 01_explained_variance.png")
    print("  +-- 02_pca_2d.png")
    print("  +-- 02b_kpca_2d.png")
    print("  +-- 03_tsne_2d.png")
    print("  +-- 04_umap_2d.png            (if umap-learn installed)")
    print("  +-- 05_autoencoder_2d.png     (if tensorflow installed)")
    print("  +-- 06_correlation_heatmap.png")
    print("=" * 60)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(csv_path: str, sample_n: int = 30_000,
                 pca_components: int = None, bottleneck_dim: int = 8):

    print("=" * 60)
    print("  CICIDS2017 DIMENSIONALITY REDUCTION PIPELINE")
    print("=" * 60)

    # 1. Load
    X, y, label_series, feature_cols, label_col, class_names = \
        load_cleaned(csv_path, sample_n=sample_n)

    original_shape = X.shape

    # 2. Scale
    X_scaled, scaler = scale_features(X)

    # 3. Variance threshold
    X_vt, selected_after_vt = variance_threshold_filter(X_scaled, feature_cols)

    # 4. Correlation filter
    X_filtered, remaining_cols = correlation_filter(X_vt, selected_after_vt)

    # 5. PCA
    X_pca, pca_model, n_pca = run_pca(X_filtered, y, label_series,
                                       n_components=pca_components)

    # 6. Incremental PCA (same n as PCA)
    run_incremental_pca(X_filtered, n_components=n_pca)

    # 7. Kernel PCA (2D for visualization)
    run_kernel_pca(X_filtered, y, label_series, n_components=2)

    # 8. t-SNE (on PCA output for speed)
    idx = np.random.choice(len(X_pca), size=min(5000, len(X_pca)), replace=False)
    run_tsne(X_pca[idx], y[idx])

    # 9. UMAP
    run_umap(X_pca, y, label_series, n_components=2)

    # 10. Autoencoder
    run_autoencoder_reduction(X_filtered, y, label_series,
                               bottleneck_dim=bottleneck_dim)

    # Summary
    print_summary(original_shape, feature_cols,
                  selected_after_vt, remaining_cols, n_pca)

    print(f"\n[+] Pipeline complete. All outputs in: {OUT_DIR}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dimensionality reduction pipeline for cleaned CICIDS2017 data.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to cleaned CICIDS2017 CSV.")
    parser.add_argument("--sample", type=int, default=30_000,
                        help="Rows to sample (default 30000). Use 0 for full dataset.")
    parser.add_argument("--pca_components", type=int, default=None,
                        help="Number of PCA components (default: auto for 95%% variance).")
    parser.add_argument("--bottleneck", type=int, default=8,
                        help="Autoencoder bottleneck dimension (default: 8).")
    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        sample_n=args.sample if args.sample > 0 else None,
        pca_components=args.pca_components,
        bottleneck_dim=args.bottleneck
    )