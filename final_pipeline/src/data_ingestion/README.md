# Data Ingestion Module

Ingestion and source-format conversion utilities.

## What this module does

- Contains scripts used to convert source-specific files into cleaned JSON/text structures.
- Includes utilities to split multi-article files and normalise JSON path fields.

## Why this module exists

Raw corpora are heterogeneous (PDF, HTML, multi-article JSON). These scripts standardise formats before preprocessing.

## Files

```
data_ingestion/
├── data_parsing/
│   ├── process_era2_sources.py
│   ├── split_datamation.py
│   └── standardize_json.py
└── utils/
```

## Current status

This folder contains a mix of active and legacy scripts.

- `split_datamation.py` and `standardize_json.py` target the current `data/...` layout.
- `process_era2_sources.py` contains older path/import assumptions (`Sources/...`) and may require local fixes before use in this repository state.

## How to use safely

1. Review script paths before execution.
2. Run on a copy or branch first.
3. Re-run preprocessing after any ingestion changes.

Typical commands (from `final_pipeline/src/data_ingestion`):

```bash
python data_parsing/split_datamation.py
python data_parsing/standardize_json.py
```

## Note

For routine experiments, ingestion is usually not required as cleaned and processed data already exists in this repository.
