# Architecture Notes

## Design Decisions

- **Streamlit** chosen for rapid prototyping and interactivity without frontend JS
- **Plotly** as the sole visualization library — interactive charts with static export via **kaleido** (PNG, JPEG, SVG)
- **Lifelines** for survival analysis (Kaplan-Meier, log-rank) — standard in biomedical Python
- **Modular src/ layout** separates UI components from business logic for testability
- **Colorblind-safe palette** — no pure red or green anywhere; diverging scales use PuOr instead of RdBu

## Data Flow

```
User selects dataset → data_loader.py → DataFrame in session state
User picks variables → sidebar.py → stats_panel / viz_panel
Stats panel → utils/statistics.py → results rendered in UI with download buttons
Viz panel → customization sidebar → interactive Plotly figure + export button
```

## Module Boundaries

| Layer | Files | Responsibility |
|-------|-------|----------------|
| Entry point | `app.py` | Page config, global CSS (pill tabs, red sliders), routing |
| Sidebar | `src/components/sidebar.py` | Dataset selector, module switcher, data preview |
| Statistical Analysis | `src/components/stats_panel.py` | 3 analysis tabs (Group Comparison, Correlation, Survival) |
| Figure Maker | `src/components/viz_panel.py` | 3 plot tabs (Distribution, Categorical, Heatmap) |
| Statistics engine | `src/utils/statistics.py` | All test functions, effect sizes, recommendation engines |
| Helpers | `src/utils/helpers.py` | Formatting, figure download, CSV export |
| Theme | `src/theme.py` | Color palette, Plotly layout defaults |
| Data | `src/data_loader.py` | Loading, preprocessing, column classification |
| Config | `config/settings.py` | App-wide constants (palettes, alpha, dimensions) |
