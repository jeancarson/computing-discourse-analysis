# Cross-Era Analysis Module

Computes topic similarity between Era 1 and Era 2 from pipeline outputs.

## What this module does

- Loads cached chunk embeddings for each era.
- Loads processed metadata and document-topic assignments.
- Builds topic centroids by computing the mean embedding vector across all documents assigned to each topic.
- Computes cosine similarity within and across eras.
- Exports heatmaps and a similarity table.

## Why this module exists

It provides a consistent quantitative comparison layer after topic modelling, identifying thematic correspondences and divergences between the two eras.

## Files

```
cross_era_analysis/
├── similarity_analyzer.py
├── visualizations.py
└── run_similarity_analysis.py
```

## Inputs required

- `final_pipeline/outputs/embeddings_cache/*.npy`
- `final_pipeline/outputs/era1/Era1_document_topics.csv`
- `final_pipeline/outputs/era2/Era2_document_topics.csv`
- `final_pipeline/outputs/era1/Era1_topic_keywords.csv`
- `final_pipeline/outputs/era2/Era2_topic_keywords.csv`

## How to run

From `final_pipeline/src`:

```bash
python cross_era_analysis/run_similarity_analysis.py
python cross_era_analysis/run_similarity_analysis.py --top-n 10
```

## Outputs

Under `final_pipeline/outputs/cross_era_analysis/`:

- `era1_within_similarity.html`
- `era2_within_similarity.html`
- `cross_era_similarity.html`
- `topic_similarity.csv`

## Interpreting similarity scores

Scores are classified as follows:

- **> 0.7** — high similarity: topics discuss closely related concepts across eras
- **0.5–0.7** — moderate similarity: related but evolved concepts with partial overlap
- **< 0.5** — low similarity: era-specific topics with minimal cross-era correspondence

Similarity is centroid-based and reflects semantic proximity, not causal or historical equivalence.
