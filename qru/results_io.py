"""Read a frozen reference table or an independently recomputed CSV run."""
from pathlib import Path
import pandas as pd

def read_result(directory, name):
    return pd.read_csv(Path(directory)/name, float_precision="round_trip")
