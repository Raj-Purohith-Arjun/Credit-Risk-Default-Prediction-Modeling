import pandas as pd
from pathlib import Path
from src.config import DATA_RAW


def load_raw(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = DATA_RAW / "lending_club.csv"
    if not path.exists():
        from data.download import download
        download()
    return pd.read_csv(path, low_memory=False)


def load_processed(path: Path = None) -> pd.DataFrame:
    from src.config import DATA_PROCESSED
    if path is None:
        path = DATA_PROCESSED / "features.parquet"
    return pd.read_parquet(path)
