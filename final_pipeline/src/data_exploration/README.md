# Data Exploration Module

Exploratory scripts for inspecting corpus characteristics and informing stopword decisions.

## What this module does

- Inspects token and document word counts.
- Identifies candidate domain-specific stopwords via word frequency analysis.
- Generates word clouds for visual inspection of term distribution.

## Why this module exists

Exploration supports preprocessing decisions - in particular, building and refining the custom domain-specific stopword list — before re-running modelling.

## Files

```
data_exploration/
├── analyze_word_counts.py
├── analyze_stopwords.py
├── wordcloud_generator.py
└── stopwords.txt
```

## Usage status

These scripts are **supporting utilities**, not part of the core notebook pipeline.

Some scripts in this folder still reference older import/path layouts and may require local path fixes before execution in this repository state.

## Typical use

Run from `final_pipeline/src/data_exploration` after verifying paths and imports in the target script:

```bash
python analyze_word_counts.py
python analyze_stopwords.py --era 1
python wordcloud_generator.py
```

## Output expectation

- Console summaries for term frequency and document coverage.
- Optional CSV and image outputs depending on script options.

## Note

Use this module for diagnostics and iteration, not as a required step in the default end-to-end run.
