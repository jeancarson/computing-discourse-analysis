# Topic Modelling Module

BERTopic workflow used by the project to produce topic assignments and topic descriptors.

## What this module does

- Loads processed texts for Era 1 and Era 2.
- Applies semantic chunking to long documents; embeddings are averaged across chunks before topic assignment.
- Trains BERTopic using:
  - sentence-transformer embeddings (`Alibaba-NLP/gte-base-en-v1.5`),
  - UMAP dimensionality reduction,
  - HDBSCAN clustering,
  - class-based TF-IDF for topic term extraction.
- Exports topic metadata, assignments, probabilities, and visualisations.

## Why this module exists

It provides a reproducible way to generate topic structures for both eras with a shared configuration format and era-specific parameter overrides.

## Files

```
topic_modelling/
├── core/topic_modeler.py
├── utils/chunker.py
├── utils/data_utils.py
├── run_bertopic.py
├── config.py
└── config.json
```

## How to run

From `final_pipeline/src`:

```bash
python topic_modelling/run_bertopic.py --era 1
python topic_modelling/run_bertopic.py --era 2
python topic_modelling/run_bertopic.py --era 2 --years 2023 2025
```

## Programmatic use

```python
from config import load_config
from topic_modelling.core.topic_modeler import TopicModeler

config = load_config(era=1)
modeler = TopicModeler(config)
topics, probs = modeler.fit_transform(texts, embeddings=embeddings, metadata=chunked_metadata)
```

## Outputs

By default, script outputs are written under `final_pipeline/outputs/era1` or `final_pipeline/outputs/era2`:

- `EraX_topic_info.csv`
- `EraX_topic_keywords.csv`
- `EraX_topic_assignments.csv`
- `EraX_topic_probabilities.csv` (if probabilities are enabled)
- topic visualisation HTML files

## Configuration

This module reads `final_pipeline/src/config.json` via `final_pipeline/src/config.py`.

- Shared settings are under `shared`.
- Era-specific parameter overrides are under `era1` and `era2`.

## Notes

- The main notebook may override some parameters in a cell for experiments.
- Stopword removal is handled during preprocessing. BERTopic's vectorizer is configured with `stop_words=None` in the current implementation.
