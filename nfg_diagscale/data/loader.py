"""
Dataset loader for NASA HTTP and FIFA World Cup 1998 traces.

[P5] Guruge & Priyadarshana (2025), Front. Comput. Sci. 7:1509165
  - sect 4.1.1: NASA Kennedy Space Center HTTP logs, Jul-Aug 1995, 3,461,612 requests
  - sect 4.1.2: FIFA World Cup 1998 HTTP logs, Apr-Jul 1998, 1,352,804,107 requests
  - sect 4.2: "took 70% for training and 30% for evaluation while preserving the time order"
  - Preprocessing: "aggregate same-minute logs to calculate the HTTP request rate per minute"
"""
import os
import re
import gzip
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime


NASA_URLS = [
    "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
    "https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")

# [P5 sect 4.1.1] Apache Common Log Format timestamp pattern
LOG_TIMESTAMP_RE = re.compile(r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})')


def _parse_nasa_line(line):
    match = LOG_TIMESTAMP_RE.search(line)
    if match:
        try:
            return datetime.strptime(match.group(1), "%d/%b/%Y:%H:%M:%S")
        except ValueError:
            return None
    return None


def download_nasa(data_dir=None):
    """Download and parse NASA HTTP access logs into request-rate-per-minute."""
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "nasa_rps.csv")
    if os.path.exists(csv_path):
        print(f"[Data] NASA dataset already cached at {csv_path}")
        return pd.read_csv(csv_path, parse_dates=["ds"])

    timestamps = []
    for url in NASA_URLS:
        fname = os.path.basename(url)
        local_gz = os.path.join(data_dir, fname)

        if not os.path.exists(local_gz):
            print(f"[Data] Downloading {url} ...")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(local_gz, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[Data] Saved {local_gz}")

        print(f"[Data] Parsing {fname} ...")
        with gzip.open(local_gz, "rt", errors="replace") as f:
            for line in f:
                ts = _parse_nasa_line(line)
                if ts is not None:
                    timestamps.append(ts)

    print(f"[Data] Parsed {len(timestamps)} requests total")

    # [P5 sect 4.1.1] "aggregate same-minute logs to calculate the HTTP request rate per minute"
    ts_series = pd.Series(timestamps)
    ts_series = ts_series.dt.floor("min")
    counts = ts_series.value_counts().sort_index()

    df = pd.DataFrame({"ds": counts.index, "y": counts.values})
    df = df.sort_values("ds").reset_index(drop=True)

    # [P5 sect 4.2] "replace missing data with zero"
    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="min")
    df = df.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    df.columns = ["ds", "y"]

    df.to_csv(csv_path, index=False)
    print(f"[Data] NASA dataset saved to {csv_path}, shape={df.shape}")
    return df


def load_fifa_from_csv(csv_path):
    """
    Load FIFA World Cup 1998 dataset from a user-provided CSV.

    [P5 sect 4.1.2] The FIFA dataset is very large (1.35 billion requests).
    It must be pre-downloaded and pre-aggregated to request-rate-per-minute.
    Expected CSV format: columns 'ds' (datetime) and 'y' (request count per minute).
    """
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def generate_synthetic_fifa(n_days=30, seed=42):
    """
    Generate a synthetic FIFA-like trace with extreme spikes for testing
    when the real FIFA dataset is not available.

    [P5 sect 4.1.2] "demonstrates more significant anomalies than the NASA dataset"
    This captures that spike characteristic for development purposes.
    """
    rng = np.random.RandomState(seed)
    n_minutes = n_days * 24 * 60
    t = np.arange(n_minutes)

    # Diurnal base pattern
    base = 200 + 150 * np.sin(2 * np.pi * t / (24 * 60))
    # Weekly modulation
    base *= 1 + 0.3 * np.sin(2 * np.pi * t / (7 * 24 * 60))
    # Random noise
    noise = rng.normal(0, 20, n_minutes)
    # Extreme spikes (match day events)
    spikes = np.zeros(n_minutes)
    for _ in range(n_days // 3):
        spike_start = rng.randint(0, n_minutes - 120)
        spike_duration = rng.randint(60, 180)
        spike_magnitude = rng.uniform(3, 8)
        spikes[spike_start:spike_start + spike_duration] += base[spike_start] * spike_magnitude

    y = np.maximum(base + noise + spikes, 0).astype(int)

    start = pd.Timestamp("1998-04-30")
    ds = pd.date_range(start, periods=n_minutes, freq="min")
    df = pd.DataFrame({"ds": ds, "y": y})
    return df


def train_test_split(df, train_ratio=0.7):
    """
    [P5 sect 4.2] "took 70% for training and 30% for evaluation
    while preserving the time order"
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def load_dataset(config):
    name = config["dataset"]["name"]
    ratio = config["dataset"]["train_ratio"]

    if name == "nasa":
        df = download_nasa()
    elif name == "fifa_synthetic":
        df = generate_synthetic_fifa()
    elif name.endswith(".csv"):
        df = pd.read_csv(name, parse_dates=["ds"])
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'nasa', 'fifa_synthetic', or a CSV path.")

    agg = config["dataset"].get("aggregation_minutes", 1)
    if agg > 1:
        df = df.set_index("ds").resample(f"{agg}min").sum().reset_index()

    train_df, test_df = train_test_split(df, ratio)
    print(f"[Data] Dataset '{name}': train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df
