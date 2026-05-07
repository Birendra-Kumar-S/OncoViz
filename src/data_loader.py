"""
Data loading and preprocessing for OncoViz.

Handles reading the VA Lung Cancer (CSV) and Prostate Cancer (XLS) datasets,
cleaning column types, creating derived columns, and providing metadata
for the UI components.

Design: Raw files stay untouched in data/raw/. All transformations happen
in-memory at load time and are cached by Streamlit so they only run once
per session.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Dataset registry — central metadata for each dataset
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "VA Lung Cancer": {
        "file": "VA_Lung_Cancer.csv",
        "description": (
            "Veterans' Administration lung cancer clinical trial. "
            "137 patients randomized to standard or test chemotherapy, "
            "with survival times across 4 cell types."
        ),
        "time_col": "t",
        "event_col": "event",          # derived binary column
        "treatment_col": "therapy",
        "source": "Kalbfleisch & Prentice (1980)",
    },
    "Prostate Cancer": {
        "file": "Prostate_Cancer.xls",
        "description": (
            "Prostate cancer clinical trial comparing estrogen dosages "
            "vs placebo. 502 patients with comprehensive clinical "
            "covariates including tumor size, Gleason score, and "
            "bone metastasis status."
        ),
        "time_col": "dtime",
        "event_col": "event",          # derived binary column
        "treatment_col": "rx",
        "source": "Byar & Green (1980)",
    },
}

# ---------------------------------------------------------------------------
# Human-readable variable descriptions (for tooltips / data dictionary)
# ---------------------------------------------------------------------------
VARIABLE_DESCRIPTIONS = {
    # --- VA Lung Cancer ---
    "therapy":   "Treatment arm (standard or test chemotherapy)",
    "cell":      "Tumor cell type (Squamous, Small, Adeno, Large)",
    "t":         "Survival time (days from randomization)",
    "dead":      "Original status (dead / censored)",
    "event":     "Event indicator (1 = died, 0 = censored)",
    "kps":       "Karnofsky performance score (0–100; higher = better)",
    "diagtime":  "Months from diagnosis to randomization",
    "age":       "Patient age at randomization (years)",
    "prior":     "Prior therapy received (yes / no)",

    # --- Prostate Cancer ---
    "patno":     "Patient ID number",
    "stage":     "Disease stage (3 or 4)",
    "rx":        "Treatment (placebo, 0.2 mg / 1.0 mg / 5.0 mg estrogen)",
    "dtime":     "Survival time (months from randomization)",
    "status":    "Original status (alive or specific cause of death)",
    "wt":        "Weight in pounds",
    "pf":        "Performance status (activity level)",
    "hx":        "History of cardiovascular disease (1 = yes, 0 = no)",
    "sbp":       "Systolic blood pressure (cm Hg)",
    "dbp":       "Diastolic blood pressure (cm Hg)",
    "ekg":       "Electrocardiogram reading",
    "hg":        "Serum hemoglobin level (g/100 ml)",
    "sz":        "Tumor size (cm²)",
    "sg":        "Combined index of tumor stage and histologic grade",
    "ap":        "Serum acid phosphatase (King-Armstrong units)",
    "bm":        "Bone metastases (1 = yes, 0 = no)",
    "sdate":     "Date of study entry (encoded)",
}


# ===================================================================
# Loading & preprocessing
# ===================================================================

@st.cache_data
def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset by registry name, apply preprocessing, and return
    a clean DataFrame ready for analysis.

    Parameters
    ----------
    name : str
        Key in DATASET_REGISTRY ("VA Lung Cancer" or "Prostate Cancer").

    Returns
    -------
    pd.DataFrame
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    meta = DATASET_REGISTRY[name]
    filepath = DATA_DIR / meta["file"]

    if name == "VA Lung Cancer":
        df = _load_lung_cancer(filepath)
    else:
        df = _load_prostate_cancer(filepath)

    return df


def _load_lung_cancer(filepath: Path) -> pd.DataFrame:
    """Load and preprocess the VA Lung Cancer dataset."""
    df = pd.read_csv(filepath)

    # --- Binary event column: 1 = dead, 0 = censored ---
    df["event"] = (df["dead"] == "dead").astype(int)

    # --- Encode prior therapy: 1 = yes, 0 = no ---
    df["prior_binary"] = (df["prior"] == "yes").astype(int)

    # --- Ensure categorical types for UI selectors ---
    df["therapy"] = pd.Categorical(df["therapy"])
    df["cell"] = pd.Categorical(
        df["cell"],
        categories=["Squamous", "Small", "Large", "Adeno"],
        ordered=False,
    )
    df["prior"] = pd.Categorical(df["prior"])
    df["dead"] = pd.Categorical(df["dead"])

    return df


def _load_prostate_cancer(filepath: Path) -> pd.DataFrame:
    """Load and preprocess the Prostate Cancer dataset."""
    df = pd.read_excel(filepath, engine="xlrd")

    # --- Binary event column: 1 = dead (any cause), 0 = alive ---
    df["event"] = (df["status"] != "alive").astype(int)

    # --- Cancer-specific death flag (for cause-specific analysis) ---
    df["cancer_death"] = (df["status"] == "dead - prostatic ca").astype(int)

    # --- Simplify status into broader categories ---
    status_map = {
        "alive":                          "Alive",
        "dead - prostatic ca":            "Prostatic Cancer",
        "dead - heart or vascular":       "Cardiovascular",
        "dead - cerebrovascular":         "Cerebrovascular",
        "dead - pulmonary embolus":       "Pulmonary Embolus",
        "dead - other ca":                "Other Cancer",
        "dead - respiratory disease":     "Respiratory",
        "dead - other specific non-ca":   "Other Non-Cancer",
        "dead - unspecified non-ca":      "Unspecified Non-Cancer",
        "dead - unknown cause":           "Unknown",
    }
    df["death_category"] = df["status"].map(status_map)
    df["death_category"] = pd.Categorical(df["death_category"])

    # --- Handle acid phosphatase outlier sentinel values ---
    # Values near 9999 or 999.875 are likely missing/sentinel
    ap_upper = df["ap"].quantile(0.99)
    df["ap_clean"] = df["ap"].where(df["ap"] <= ap_upper * 3, np.nan)

    # --- Ensure categorical types ---
    df["rx"] = pd.Categorical(
        df["rx"],
        categories=["placebo", "0.2 mg estrogen", "1.0 mg estrogen", "5.0 mg estrogen"],
        ordered=True,
    )
    df["stage"] = pd.Categorical(df["stage"])
    df["pf"] = pd.Categorical(
        df["pf"],
        categories=[
            "normal activity",
            "in bed < 50% daytime",
            "in bed > 50% daytime",
            "confined to bed",
        ],
        ordered=True,
    )
    df["ekg"] = pd.Categorical(df["ekg"])
    df["hx"] = df["hx"].astype(int)
    df["bm"] = df["bm"].astype(int)

    # --- Drop sdate (encoded date, not useful for analysis) ---
    df = df.drop(columns=["sdate"])

    return df


# ===================================================================
# Column metadata — drives the UI selectors and smart suggestions
# ===================================================================

def get_column_metadata(df: pd.DataFrame) -> dict:
    """
    Classify every column in a DataFrame by its analytical type.

    Returns
    -------
    dict with keys:
        "numerical"   : list of continuous numeric columns
        "categorical" : list of categorical / string columns
        "binary"      : list of columns with exactly 2 unique non-null values
        "all"         : list of all column names
        "types"       : dict mapping column name → type string
    """
    numerical = []
    categorical = []
    binary = []
    col_types = {}

    for col in df.columns:
        n_unique = df[col].dropna().nunique()

        if n_unique <= 2 and n_unique > 0:
            binary.append(col)
            col_types[col] = "binary"
        elif pd.api.types.is_numeric_dtype(df[col]) and n_unique > 10:
            numerical.append(col)
            col_types[col] = "numerical"
        elif pd.api.types.is_numeric_dtype(df[col]) and n_unique <= 10:
            # Low-cardinality numeric → treat as categorical
            categorical.append(col)
            col_types[col] = "categorical"
        else:
            categorical.append(col)
            col_types[col] = "categorical"

    return {
        "numerical": sorted(numerical),
        "categorical": sorted(categorical),
        "binary": sorted(binary),
        "all": list(df.columns),
        "types": col_types,
    }


def get_variable_description(col: str) -> str:
    """Return a human-readable description for a column, or a fallback."""
    return VARIABLE_DESCRIPTIONS.get(col, col.replace("_", " ").title())


def get_dataset_names() -> list:
    """Return the list of available dataset names."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(name: str) -> dict:
    """Return the registry metadata for a given dataset."""
    return DATASET_REGISTRY[name]
