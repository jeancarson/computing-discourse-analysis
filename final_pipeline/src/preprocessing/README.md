# Preprocessing Module

Text cleaning and processed-corpus generation used before topic modelling.

## What this module does

- Loads cleaned source JSONs via `DataLoader`.
- Applies normalisation and filtering via `TextPreprocessor`.
- Writes processed plain-text files and index metadata to `data/processed/Era1` and `data/processed/Era2`.

## Why this module exists

The topic modelling stage expects consistent text input. This module standardises text format and removes high-frequency non-informative terms.

## Files

```
preprocessing/
├── data_loader.py
├── text_preprocessor.py
├── text_utils.py
├── config.py
├── config.json
└── stopwords.json
```

## Cleaning steps (config-driven)

The current config supports:

- dehyphenation
- whitespace normalisation
- special-character removal
- stopword removal (NLTK defaults + custom domain-specific list via `stopwords.json`)
- lowercasing
- minimum document length filtering

## How to run

From `final_pipeline/src/preprocessing`:

```bash
python text_preprocessor.py
```

This processes both eras when source data is available.

## Outputs

Per era:

- processed text files in `data/processed/EraX/`
- `EraX_index.json` mapping each processed file back to source metadata

## Important note

Semantic chunking for long documents is handled later in `topic_modelling/utils/chunker.py`, not in this module.
