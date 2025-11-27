from typing import Optional
from pathlib import Path
import pandas as pd
from loguru import logger

from loan_mlops.utils.paths import APPROVAL_DATA_FILE, DEFAULT_DATA_FILE


def load_approval_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the loan approval dataset from CSV.

    Parameters
    ----------
    path : Path | None
        Optional explicit path. If None, uses APPROVAL_DATA_FILE.

    Returns
    -------
    pd.DataFrame
        Loaded approval dataset.
    """
    csv_path = path or APPROVAL_DATA_FILE
    logger.info(f"Loading approval data from {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Approval dataset not found at {csv_path}. "
            "Place the loan approval CSV there or pass path explicitly."
        )
    df = pd.read_csv(csv_path)
    logger.info(f"Approval data loaded with shape {df.shape}")
    return df

def load_default_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the loan default dataset from CSV.

    Parameters
    ----------
    path : Path | None
        Optional explicit path. If None, uses DEFAULT_DATA_FILE.

    Returns
    -------
    pd.DataFrame
        Loaded default dataset.
    """
    csv_path = path or DEFAULT_DATA_FILE
    logger.info(f"Loading default data from {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Default dataset not found at {csv_path}. "
            "Place the loan default CSV there or pass path explicitly."
        )
    df = pd.read_csv(csv_path)
    logger.info(f"Default data loaded with shape {df.shape}")
    return df