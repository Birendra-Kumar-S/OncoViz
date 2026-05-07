"""
Optional preprocessing script.

Run standalone to clean raw data and save to data/processed/.
Usage: python scripts/preprocess.py

Currently, all preprocessing is performed in-memory at load time
by src/data_loader.py (cached via @st.cache_data). This script
can be extended if heavier offline preprocessing is needed.
"""
