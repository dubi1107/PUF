import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main():

    # Only need to download once
    # selected_states = ["IL", "MT", "NM", "OK", "TX", "NC", "SC", "FL", "AL", "TN"]

    # output_dir = download_and_split_by_state_parquet(selected_states)

    # Quick example: Load Texas data
    # tx_df = pd.read_parquet(output_dir / "Rate_PUF_TX.parquet")
    # print("\nTexas preview:")
    # print(tx_df.head())
    # print(f"Shape: {tx_df.shape}")
    analyze_nans_by_state()


def download_and_split_by_state_parquet(
    states: list[str],
    output_dir: str = "data/rate_puf_py2026_by_state",
    force_download: bool = False,
):
    """
    Downloads Rate PUF PY2026 CSV and saves one Parquet file per state.
    """
    url = "https://data.healthcare.gov/datafile/py2026/Rate_PUF.csv"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_filepath = output_path / "Rate_PUF.csv"

    # Download if needed
    if not csv_filepath.exists() or force_download:
        print("Downloading full Rate PUF PY2026 CSV (large file ~270 MB)...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(csv_filepath, "wb") as f,
            tqdm(
                desc="Downloading",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        print("\nDownload complete.")
    else:
        print("CSV already exists — skipping download.")

    # Process in chunks and save per state
    print(f"Processing and splitting data for {len(states)} states...")
    state_files = {}
    chunks = pd.read_csv(csv_filepath, chunksize=200_000, low_memory=False)

    for chunk in tqdm(chunks, desc="Processing chunks"):
        for state in states:
            state_chunk = chunk[chunk["StateCode"] == state]
            if not state_chunk.empty:
                if state not in state_files:
                    state_files[state] = []
                state_files[state].append(state_chunk)

    # Save each state to Parquet
    for state, dfs in state_files.items():
        df_state = pd.concat(dfs, ignore_index=True)
        parquet_path = output_path / f"Rate_PUF_{state}.parquet"
        df_state.to_parquet(parquet_path, index=False, compression="zstd")

        print(
            f"  ✅ Saved {state}: {len(df_state):,} rows → {parquet_path.name} "
            f"({parquet_path.stat().st_size / (1024 * 1024):.1f} MB)"
        )

    print(f"\nAll done! Files saved to: {output_path.resolve()}")
    return output_path


def analyze_nans_by_state(
    data_dir: str = "data/rate_puf_py2026_by_state",
    output_report: str = "nan_report.txt",
):
    """
    Analyzes NaN values in all Rate_PUF_{STATE}.parquet files.
    """
    data_path = Path(data_dir)

    parquet_files = list(data_path.glob("Rate_PUF_*.parquet"))
    if not parquet_files:
        print("No Parquet files found in the directory!")
        return

    print(f"Found {len(parquet_files)} state files. Analyzing NaNs...\n")

    report_lines = []
    report_lines.append("=== Rate PUF PY2026 NaN Analysis ===\n")

    for parquet_file in tqdm(sorted(parquet_files), desc="Analyzing files"):
        state = parquet_file.stem.split("_")[-1]

        df = pd.read_parquet(parquet_file)
        total_rows = len(df)

        # Columns with any NaNs
        nan_counts = df.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]

        report_lines.append(f"\n{state} ({total_rows:,} rows)")
        report_lines.append("-" * 60)

        if nan_columns.empty:
            report_lines.append("   No NaN values found in any column.")
        else:
            report_lines.append(
                f"   Columns with NaNs: {len(nan_columns)} out of {len(df.columns)}"
            )
            for col, count in nan_columns.items():
                pct = (count / total_rows) * 100
                report_lines.append(f"   {col:35} : {count:10,} rows ({pct:6.2f}%)")

        # Optional: show overall NaN percentage
        total_nans = df.isna().sum().sum()
        overall_pct = (total_nans / (total_rows * len(df.columns))) * 100
        report_lines.append(f"   Overall missing cells: {overall_pct:.3f}%")

    # Save report
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ Analysis complete! Report saved to: {output_report}")

    # Also print summary of columns that have NaNs anywhere
    print("\nQuick cross-state summary (columns with NaNs in any state):")
    all_nan_cols = set()
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file, columns=None)  # just to get columns
        nan_cols = df.columns[df.isna().any()].tolist()
        all_nan_cols.update(nan_cols)

    if all_nan_cols:
        print(sorted(all_nan_cols))
    else:
        print("No NaNs found in any file!")


# ====================== USAGE ======================
if __name__ == "__main__":
    main()
