# OncoViz — Interactive Clinical Trial Data Explorer

> A Streamlit dashboard combining interactive statistical analysis with automated figure generation for cancer clinical trial datasets.

**Course:** BMED 8813 — Biomedical Data Visualization, Georgia Institute of Technology  
**Author:** Birendra

---

## Overview

OncoViz provides researchers with a unified tool for exploring cancer clinical trial data. It merges two core workflows — **statistical hypothesis testing** and **automated publication-quality figure generation** — into a single interactive web application. Users select variables, run tests with smart recommendations, and generate customizable charts, all without writing code.

The dashboard ships with two curated clinical trial datasets and is designed around accessibility: a colorblind-safe palette (no pure red or green), downloadable figures (PNG, JPEG, SVG), and exportable data tables (CSV).

### Datasets

| Dataset | Patients | Variables | Description | Source |
|---------|----------|-----------|-------------|--------|
| VA Lung Cancer | 137 | 8 | Veterans' Administration lung cancer trial — patients randomized to standard vs. test chemotherapy, with survival times across 4 tumor cell types | Kalbfleisch & Prentice (1980) |
| Prostate Cancer | 502 | 18 | Prostate cancer trial comparing estrogen dosages vs. placebo, with comprehensive clinical covariates including tumor size, Gleason score, and bone metastasis status | Byar & Green (1980) |

---

## Features

### Statistical Analysis Module

- **Group Comparison** — Compares a continuous variable across groups. Supports Independent t-test, Mann-Whitney U (2 groups), One-way ANOVA, and Kruskal-Wallis (3+ groups). Includes automatic normality checking (Shapiro-Wilk) to recommend the appropriate test, effect sizes (Cohen's d with 95% CI, eta-squared, epsilon-squared), and post-hoc pairwise comparisons with multiple testing correction (Bonferroni, Benjamini-Hochberg FDR, Holm). Results are visualized with violin + box plots and a forest plot of pairwise effect sizes.

- **Correlation Analysis** — Computes Pearson, Spearman, or Kendall correlation between two numerical variables. Auto-recommends the method based on normality and monotonicity checks. Displays r, R², 95% confidence interval (Fisher z-transformation), and strength/direction interpretation. Includes scatter plots with OLS (Pearson) or LOWESS (Spearman/Kendall) trendlines and a full correlation matrix heatmap with significance threshold filtering.

- **Survival Analysis** — Kaplan-Meier survival estimation with optional grouping. Features confidence interval bands, censoring marks (cross-thin tick marks), median survival reference lines with vertical drops per group, 95% CI for median survival, a number-at-risk table at evenly spaced time points, and log-rank test for group comparison.

### Automated Figure Maker Module

- **Distribution Plot** — Histogram with optional KDE overlay. Supports splitting by a categorical variable with side-by-side or overlay modes, adjustable bin count, and selectable color palettes.

- **Categorical Plot** — Unified bar/proportion chart. Supports optional grouping with side-by-side or stacked bar modes, counts or percentage display, vertical/horizontal orientation, and value labels. Includes frequency and contingency tables.

- **Clinical Feature Heatmap** — Z-scored heatmap of standardized clinical measurements across all patients. Supports variable selection, patient sorting by group, and multiple color scales (PuOr, BrBG, Viridis, Plasma, Inferno, Cividis).

### Global Features

- **Figure Export** — Every plot in the dashboard (statistical and figure maker) has a download button offering PNG, JPEG, and SVG formats via kaleido, with an HTML fallback when kaleido is not installed.
- **Table Export** — Every data table has a CSV download button.
- **Colorblind-Safe Design** — All palettes exclude pure red and green. Diverging heatmap scales use PuOr (purple-orange) instead of RdBu.
- **Smart Recommendations** — Statistical tests, correlation methods, and post-hoc corrections are auto-suggested based on variable types and data properties.
- **Session State Persistence** — Test results and selections persist across UI interactions without requiring re-runs.

---

## Project Structure

```
oncoviz/
├── app.py                      # Main Streamlit entry point (global CSS, routing)
├── requirements.txt            # Python dependencies (pip)
├── environment.yml             # Conda environment specification
├── Makefile                    # Common commands (install, run, test)
├── LICENSE
├── .gitignore
│
├── .streamlit/
│   └── config.toml             # Streamlit theme (colors, font) and server config
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading, preprocessing, column metadata
│   ├── theme.py                # Chart color palette and Plotly layout defaults
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py          # Sidebar: dataset selector, module switcher, preview
│   │   ├── stats_panel.py      # Statistical analysis UI (3 tabs)
│   │   └── viz_panel.py        # Automated figure maker UI (3 tabs)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── statistics.py       # All statistical test functions and TestResult class
│       └── helpers.py          # Formatting, column classification, export utilities
│
├── config/
│   ├── __init__.py
│   └── settings.py             # App-wide constants (palettes, alpha, dimensions)
│
├── data/
│   ├── raw/                    # Original datasets (tracked in git)
│   │   ├── VA_Lung_Cancer.csv
│   │   └── Prostate_Cancer.xls
│   └── processed/              # Derived data (gitignored)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py     # Dataset loading and metadata tests
│   └── test_statistics.py      # Statistical test function tests
│
└── docs/
    ├── architecture.md         # Design decisions and architecture notes
    └── modules/
        ├── Statistical_Analysis_Module.txt
        └── Automated_Figure_Maker_Module.txt
```

---

## Documentation

Detailed technical documentation for each module is available in the `docs/modules/` directory:

- **[Statistical Analysis Module](docs/modules/Statistical_Analysis_Module.txt)** — Covers all statistical tests (t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis), effect sizes (Cohen's d, eta-squared, epsilon-squared), post-hoc pairwise comparisons with multiple testing correction, correlation methods (Pearson, Spearman, Kendall) with Fisher z-transformation CI, Kaplan-Meier survival analysis with log-rank test, and all visualizations.

- **[Automated Figure Maker Module](docs/modules/Automated_Figure_Maker_Module.txt)** — Covers the customization system, the full colorblind-safe palette catalog (12 palettes), distribution plots, categorical plots, clinical feature heatmaps with z-score standardization, the figure export system, and UI/UX design details.

---

## Installation

### Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or pip

### Option A — Conda (recommended)

```bash
# Clone the repository
git clone https://github.gatech.edu/BMED-8813-BDV-Team13/OncoViz.git
cd oncoviz

# Create and activate the conda environment
conda env create -f environment.yml
conda activate OncoViz

# Launch the dashboard
streamlit run app.py
```

### Option B — pip with virtual environment

```bash
# Clone the repository
git clone https://github.gatech.edu/BMED-8813-BDV-Team13/OncoViz.git
cd oncoviz

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

### Option C — Makefile shortcuts

```bash
make install    # pip install -r requirements.txt
make run        # streamlit run app.py
make test       # pytest tests/
```

### Verifying the installation

After running `streamlit run app.py`, the dashboard opens at `http://localhost:8501`. You should see the OncoViz header with the VA Lung Cancer dataset loaded by default. If figure downloads produce HTML files instead of images, install kaleido:

```bash
pip install -U kaleido
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `plotly` | Interactive charts |
| `scipy` | Statistical tests (t-test, ANOVA, Mann-Whitney, etc.) |
| `statsmodels` | Multiple testing correction, LOWESS |
| `lifelines` | Kaplan-Meier survival analysis, log-rank test |
| `pandas` / `numpy` | Data manipulation |
| `xlrd` | Reading .xls files (Prostate Cancer dataset) |
| `kaleido` | Static image export for Plotly figures (PNG, JPEG, SVG) |

---

## Usage

1. **Select a dataset** from the sidebar — VA Lung Cancer or Prostate Cancer.
2. **Choose a module** — Statistical Analysis or Automated Figure Maker.
3. **Explore the data** — expand "Dataset Overview" for summary statistics, raw data, and a data dictionary. Each table has a CSV download button.

### Statistical Analysis

- **Group Comparison** — Pick a grouping variable and a measurement variable. OncoViz recommends the appropriate test. Click "Run Test" to see results with effect sizes and interpretation. For 3+ groups, post-hoc pairwise comparisons are available with a forest plot.
- **Correlation** — Select two numerical variables. OncoViz recommends Pearson, Spearman, or Kendall based on normality. Click "Compute Correlation" for r, R², CI, and a scatter plot. Optionally view the full correlation matrix.
- **Survival Analysis** — Optionally group by a categorical variable. Click "Run Survival Analysis" for Kaplan-Meier curves with confidence bands, censoring marks, median lines, at-risk table, and log-rank test.

### Automated Figure Maker

- **Distribution** — Choose a numerical variable, optionally split by a category. Adjust bins, overlay mode, and color palette.
- **Categorical Plot** — Pick a categorical variable, optionally group by another. Toggle between counts/percentages and side-by-side/stacked layouts.
- **Heatmap** — Select clinical variables to display as a z-scored patient-feature heatmap. Optionally sort patients by a grouping variable.

All plots have a format selector (PNG/JPEG/SVG) and a "Download Figure" button. All tables have a "Download CSV" button.

---

## Testing

Run the test suite with:

```bash
pytest tests/ -v
```

Or with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit (wide layout, custom CSS pill-button tabs, red sliders) |
| Visualization | Plotly (interactive charts, hover tooltips, export via kaleido) |
| Statistics | SciPy, Statsmodels, Lifelines |
| Data | Pandas, NumPy, xlrd |
| Styling | Colorblind-safe palette, Streamlit theming via `.streamlit/config.toml` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
