from pathlib import Path

# data loading configuration
DATA_DIR = (Path(__file__).parent.parent.parent / 'data').resolve()
RAW_DIR = DATA_DIR / 'raw'
STAGE_DIR = DATA_DIR / 'stage'
ANALYSIS_DIR = DATA_DIR / 'analysis'
