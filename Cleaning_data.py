"""
python cleaning.py --data_dir "Input path" --output "where to store path.csv"
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path


# =============================================================================
# 1. LOAD DATA (MULTIPLE CSVs FROM FOLDER + SUBFOLDERS)
# =============================================================================

def load_cicids2017(data_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    print(f"[*] Found {len(csv_files)} CSV file(s):")

    dfs = []
    for f in csv_files:
        print(f"    Loading: {f}")
        try:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"    ❌ Skipping {f}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files could be loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n[+] Combined shape: {combined.shape}")
    return combined


# =============================================================================
# 2. NORMALIZE COLUMN NAMES
# =============================================================================

def normalize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_per_")
        .str.replace(".", "_")
    )
    return df


# =============================================================================
# 3. LABEL COLUMN
# =============================================================================

def get_label_column(df):
    for col in df.columns:
        if "label" in col:
            print(f"[+] Label column detected: {col}")
            return col
    raise ValueError("No label column found!")


# =============================================================================
# 4. CLEAN LABELS
# =============================================================================

def normalize_labels(df, label_col):
    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Remove " - attempted"
    df[label_col] = df[label_col].str.replace(" - attempted", "", regex=False)

    return df


# =============================================================================
# 5. DROP DUPLICATES
# =============================================================================

def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"[+] Removed {before - len(df)} duplicate rows")
    return df


# =============================================================================
# 6. DROP METADATA
# =============================================================================

def drop_metadata(df):
    meta_cols = ["flow_id", "source_ip", "destination_ip", "timestamp"]
    cols = [c for c in meta_cols if c in df.columns]
    df = df.drop(columns=cols, errors="ignore")
    print(f"[+] Dropped metadata columns: {cols}")
    return df


# =============================================================================
# 7. CONVERT TO NUMERIC
# =============================================================================

def convert_to_numeric(df, label_col):
    for col in df.columns:
        if col != label_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =============================================================================
# 8. HANDLE INFINITY
# =============================================================================

def handle_infinity(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# =============================================================================
# 9. HANDLE MISSING VALUES
# =============================================================================

def handle_missing(df, label_col):
    df = df.dropna(subset=[label_col])

    # Drop columns with >50% missing
    col_thresh = 0.5
    df = df.loc[:, df.isnull().mean() < col_thresh]

    # Drop rows with >50% missing
    row_thresh = 0.5
    df = df[df.isnull().mean(axis=1) < row_thresh]

    # Fill remaining NaN
    df.fillna(df.median(numeric_only=True), inplace=True)

    print("[+] Missing values handled")
    return df


# =============================================================================
# 10. DROP CONSTANT FEATURES
# =============================================================================

def drop_constant(df, label_col):
    const_cols = [c for c in df.columns if df[c].nunique() <= 1 and c != label_col]
    df.drop(columns=const_cols, inplace=True)
    print(f"[+] Dropped {len(const_cols)} constant columns")
    return df


# =============================================================================
# 11. CAP OUTLIERS
# =============================================================================

def cap_outliers(df, label_col):
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = [c for c in numeric_cols if c != label_col]

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    df[numeric_cols] = df[numeric_cols].clip(Q1 - 3*IQR, Q3 + 3*IQR, axis=1)
    print("[+] Outliers capped")
    return df


# =============================================================================
# 12. SUMMARY
# =============================================================================

def summary(df, label_col):
    print("\n========== FINAL DATASET ==========")
    print(f"Shape: {df.shape}")
    print("\nClass Distribution:")
    print(df[label_col].value_counts())


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def clean_pipeline(data_dir, output):

    df = load_cicids2017(data_dir)
    df = normalize_columns(df)
    label_col = get_label_column(df)
    df = normalize_labels(df, label_col)
    df = drop_duplicates(df)
    df = drop_metadata(df)
    df = convert_to_numeric(df, label_col)
    df = handle_infinity(df)
    df = handle_missing(df, label_col)
    df = drop_constant(df, label_col)
    df = cap_outliers(df, label_col)

    summary(df, label_col)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\n✅ Cleaned dataset saved at: {output}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="cicids2017_cleaned.csv")
    args = parser.parse_args()

    clean_pipeline(args.data_dir, args.output)