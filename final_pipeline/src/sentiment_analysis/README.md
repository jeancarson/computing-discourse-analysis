# Sentiment Analysis Module

Topic-aware sentiment analysis used after topic modelling.

## What this module does

- Runs document-level sentiment classification using a fine-tuned transformer model.
- Handles long documents via sentence-based chunking and probability averaging.
- Aggregates document sentiment to topic level and topic-by-time level.
- Exports CSV outputs and HTML visualisations.

## Why this module exists

Topic modelling identifies what is discussed; sentiment analysis adds how those topics are emotionally framed — revealing whether themes are discussed with optimism, concern, neutrality, or negativity across each era.

## Model and scoring

- Model: `lxyuan/distilbert-base-multilingual-cased-sentiments-student` — a fine-tuned DistilBERT model trained on formally written multilingual content, selected for its match to the register of the sources under analysis (academic papers, industry reports, oral histories)
- 3-class probabilities: positive, neutral, negative
- Continuous compound score used for aggregation:

  `compound_score = P(positive) - P(negative)`

  This produces a normalised scale from −1 (strongly negative) to +1 (strongly positive).

## Long-document handling

- Documents over 400 words are split at sentence boundaries.
- Sentiment is analysed per chunk independently.
- Class probabilities are averaged across chunks before deriving the final label and compound score.

## Files

```
sentiment_analysis/
├── sentiment_analyzer.py
├── aggregator.py
├── utils.py
├── visualizations.py
├── run_sentiment_analysis.py
└── __init__.py
```

## How to run

From `final_pipeline/src`:

```bash
python sentiment_analysis/run_sentiment_analysis.py --era 1
python sentiment_analysis/run_sentiment_analysis.py --era 2 --device 0
```

## Programmatic use

```python
from sentiment_analysis import SentimentAnalyzer, TopicSentimentAggregator

analyzer = SentimentAnalyzer(device=0)
sentiments = analyzer.analyze_batch(texts)

aggregator = TopicSentimentAggregator()
topic_sentiments = aggregator.aggregate_by_topic(document_topics_df, sentiments, doc_ids)
topic_sentiments_temporal = aggregator.aggregate_by_topic_temporal(
    document_topics_df,
    sentiments,
    doc_ids,
    publication_dates,
    time_period='year'
)
```

## Outputs

Typical per-era outputs under `final_pipeline/outputs/era1` or `final_pipeline/outputs/era2`:

- `EraX_document_sentiments.csv`
- `EraX_topic_sentiments.csv`
- `EraX_topic_sentiments_temporal.csv`
- `EraX_sentiment_summary.json` (script path)
- `EraX_sentiment_temporal.html`
- `EraX_sentiment_heatmap.html`
- `EraX_sentiment_distribution.html`
