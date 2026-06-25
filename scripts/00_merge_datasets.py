import pandas as pd
from pathlib import Path


raw_dir = Path("data/raw")

output_xlsx = raw_dir / "combined_dataset.xlsx"
output_pkl = raw_dir / "combined_dataset.pkl"
output_parquet = raw_dir / "combined_dataset.parquet"

supported_extensions = [
    "*.xlsx",
    "*.xls",
    "*.csv",
]


def read_file(file_path: Path):
    suffix = file_path.suffix.lower()

    print(f"Reading: {file_path.name}")

    if suffix == ".csv":
        return pd.read_csv(file_path, low_memory=False)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, engine="openpyxl")

    return pd.DataFrame()


def main():
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_files = []

    for ext in supported_extensions:
        all_files.extend(raw_dir.glob(ext))

    all_files = [
        file for file in all_files
        if file.name not in [
            output_xlsx.name,
            output_pkl.name,
            output_parquet.name,
        ]
    ]

    if len(all_files) == 0:
        raise ValueError("No supported files found inside data/raw")

    print(f"Found {len(all_files)} files")

    all_dfs = []

    for file in all_files:
        try:
            df = read_file(file)

            if df.empty:
                continue

            df["SOURCE_FILE"] = file.name
            all_dfs.append(df)

            print(f"Rows loaded: {len(df)}")

        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    if len(all_dfs) == 0:
        raise ValueError("No valid datasets could be loaded")

    all_columns = sorted(
        set().union(*(df.columns for df in all_dfs))
    )

    aligned_dfs = [
        df.reindex(columns=all_columns)
        for df in all_dfs
    ]

    combined = pd.concat(
        aligned_dfs,
        ignore_index=True
    )

    combined = combined.drop_duplicates().reset_index(drop=True)

    combined.to_pickle(output_pkl)

    try:
        parquet_df = combined.copy()

        for col in parquet_df.columns:
            if parquet_df[col].dtype == "object":
                parquet_df[col] = parquet_df[col].astype(str)

        parquet_df.to_parquet(output_parquet, index=False)

    except Exception as e:
        print(f"Parquet save skipped: {e}")

    combined.to_excel(output_xlsx, index=False)

    print("\n===================================")
    print("FINAL COMBINED ROWS:", len(combined))
    print("OUTPUT PKL:", output_pkl)
    print("OUTPUT PARQUET:", output_parquet)
    print("OUTPUT EXCEL:", output_xlsx)
    print("===================================")


if __name__ == "__main__":
    main()