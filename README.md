# Computing Discourse Analysis

Computational discourse analysis comparing professional and public narratives across two transformative eras in computing:

- **Era 1:** The introduction of high-level programming languages and compilers (1955–1970)
- **Era 2:** The emergence of AI-assisted coding tools (2020–present)

## Project Aim

The project asks: _what are the similarities and differences in how the computing profession responded to these two disruptions?_ It uses topic modelling, sentiment analysis, and cross-era embedding similarity to surface thematic structures and emotional valences across ~200 documents per era, drawn from academic, industry, and public sources.

## Methods

1. **Preprocessing**
   - Text normalisation, stopword removal (NLTK defaults + custom domain-specific list).
   - Processed files and an index are written to `data/processed/Era1` and `data/processed/Era2`.

2. **Topic Modelling**
   - BERTopic with sentence-transformer embeddings (`Alibaba-NLP/gte-base-en-v1.5`).
   - UMAP + HDBSCAN with era-specific parameters from `final_pipeline/src/config.json`.
   - Long documents are chunked and embeddings averaged before topic assignment.

3. **Sentiment Analysis**
   - Transformer model: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`.
   - Per-document label and class probabilities are produced.
   - Continuous compound score computed as `P(positive) - P(negative)`.
   - Topic-level and temporal aggregations are generated.

4. **Cross-Era Similarity**
   - Cosine similarity computed between topic embedding centroids across eras.
   - Within-era and cross-era similarity heatmaps are exported.

## Key Results

- **Era 1 topics identified:** 15 | **Era 2 topics identified:** 14
- Era 1 sentiment is predominantly negative (era mean: −0.09); Era 2 is more balanced (era mean: −0.02)
- In both eras, the most negatively valenced topics concern practitioner identity; the most positive concern measurable technical achievement
- Topic 1.09 (Tower of Babel / language proliferation) maintains cross-era cosine similarity above 0.50 with 13 of 14 Era 2 topics - the single most structurally persistent theme across the two corpora

Full topic tables, sentiment distributions, and similarity matrices are in:

- `final_pipeline/outputs/`
- `final_pipeline/outputs/era1/`
- `final_pipeline/outputs/era2/`
- `final_pipeline/outputs/cross_era_analysis/`

## Topic Summary Tables

### Era 1 Topics

| Topic ID | Title                                                      | Doc Count | Keywords                                                    | Mean Sentiment |
| -------- | ---------------------------------------------------------- | --------: | ----------------------------------------------------------- | -------------: |
| 1.00     | Mythology of Early Computing                               |        41 | thing, would, people, think, one                            |         -0.102 |
| 1.01     | Programming Systems and the Emerging Discipline            |        19 | program, function, system, computer, form                   |         -0.138 |
| 1.02     | COBOL and Committee-Driven Language Design                 |        17 | cobol, committee, language, computer, problem               |         -0.066 |
| 1.03     | Symbolic Computation and Mathematical Programming          |        13 | program, letter, solution, zero, may                        |         -0.018 |
| 1.04     | Historical Reflection and Programming Paradigms            |        11 | logic, system, language, programming, computer              |         -0.130 |
| 1.05     | Institution-Building and CS Departments                    |        10 | cj, th, one, year, computer                                 |         -0.120 |
| 1.06     | Early Pioneer Journeys and Professional Identity Crisis    |         9 | gb, jb, program, yes, dijkstra                              |         -0.099 |
| 1.07     | Language Comparison, Specification, and Standards          |         9 | unit, format, algol, program, data                          |         -0.024 |
| 1.08     | The Priesthood, FORTRAN Culture, and Programming Economics |         9 | computer, fortran, programming, system, program             |         -0.085 |
| 1.09     | The Tower of Babel and Language Proliferation              |         9 | language, ieee, programming, tower, babel                   |         -0.181 |
| 1.10     | Major High-Level Languages and the Promise of Automation   |         8 | cobol, language, noise, statement, source                   |          0.043 |
| 1.11     | Origins of Computer Science                                |         8 | newell, norberg, people, thing, computer                    |         -0.227 |
| 1.12     | Universal Languages and Commercial Computing               |         7 | uncol, sale, computer, system, order                        |         -0.009 |
| 1.13     | Language Implementation and Code Optimisation              |         7 | loop, code, set, iv, statement                              |         -0.157 |
| 1.14     | When Technology Became Language                            |         7 | computer, user, programming, language, programming language |         -0.027 |

### Era 2 Topics

| Topic ID | Title                                              | Doc Count | Keywords                                           | Mean Sentiment |
| -------- | -------------------------------------------------- | --------: | -------------------------------------------------- | -------------: |
| 2.00     | The AI-Augmented Development Ecosystem             |        42 | model, one, code, tool, say                        |         -0.026 |
| 2.01     | Code Security, Vulnerability, and AI Quality       |        18 | code, test, codex, suggestion, copilot             |          0.001 |
| 2.02     | AI as Pair Programming Partner                     |        15 | assistant, code, activity, student, developer      |          0.036 |
| 2.03     | ChatGPT, Prompt Engineering, and Empirical Studies |        12 | chatgpt, participant, code, copilot, study         |         -0.021 |
| 2.04     | Empirical Productivity and Developer Experience    |        12 | code, experiment, developer, copilot, tool         |         -0.022 |
| 2.05     | Vibe Coding, AI Agents, and the New Workflow       |        11 | code, jules, description, method, vibe coding      |         -0.049 |
| 2.06     | Designing AI-Assisted Coding Experiences           |        10 | participant, canvas, design, chatgpt canvas, study |         -0.082 |
| 2.07     | AI in Programming Education and Pedagogy           |        10 | student, assistant, education, learning, ai        |          0.013 |
| 2.08     | Vibe Coding Culture and Startup Transformation     |         8 | vibe, vibe coding, coding, agent, startup          |          0.039 |
| 2.09     | Developer Perceptions and User Experience          |         7 | copilot, user, participant, chatgpt, developer     |         -0.183 |
| 2.10     | GenAI Adoption, Industry Metrics, and Job Market   |         7 | copilot, code, genai, http, software               |         -0.030 |
| 2.11     | Technical Benchmarking of AI Code Generation       |         6 | code, prompt, fig, int, array                      |          0.058 |
| 2.12     | Ethics, Governance, and Professional Standards     |         6 | code, test, tool, ethical, certification           |         -0.036 |
| 2.13     | AI in Military and National Security Contexts      |         3 | air, harasimowicz, air force, oligo, force         |          0.097 |

## Data Sources

The raw corpus (original PDFs/HTML and cleaned full text) is not included in this repository, since a large share of it is copyrighted third-party material (academic papers, magazine articles, industry publications) that can't be redistributed in bulk. `data/processed/Era{1,2}/Era{1,2}_index.json` retains a per-document manifest (source publication, publication date, original filepath) without any document body text, so the composition of each corpus remains fully auditable.

**Era 1 corpus (165 documents, 1957–1970):**

| Source | Docs |
|---|---:|
| Communications of the ACM (CACM) | 69 |
| Datamation | 49 |
| Charles Babbage Institute (oral histories) | 14 |
| Communications of the ACM (misc./duplicate tagging) | 9 |
| Computer History Museum | 2 |
| NASA/JPL Technical Memorandum 33-566 | 2 |
| IEEE Annals of the History of Computing, IEEE Transactions on Computers, Technology and Culture, and other individually-cited academic/archival sources | 20 |

**Era 2 corpus (181 documents, 2015–present):**

| Source | Docs |
|---|---:|
| InfoWorld | 68 |
| Wired | 51 |
| Academic papers (arXiv, ACM/IEEE venues, journals) | 32 |
| Job-market coverage (InfoWorld/Computerworld) | 12 |
| Technical documentation (vendor whitepapers, docs) | 11 |
| ACM Queue | 5 |
| Industry analyst reports | 1 |
| Blogs/misc. (GitHub, CACM) | 1 |

Individual source citations (title, publisher, date) for every document are recoverable from the index JSON files; get in touch if you need the full text of a specific source for verification.

## Repository Structure

```
.
├── data/
│   └── processed/        # per-document source manifests only (no full text)
├── final_pipeline/
│   ├── src/
│   │   ├── data_ingestion/
│   │   ├── preprocessing/
│   │   ├── data_exploration/
│   │   ├── topic_modelling/
│   │   ├── sentiment_analysis/
│   │   ├── cross_era_analysis/
│   │   └── parameter_tuning/
│   └── outputs/
├── requirements.txt
└── README.md
```

## How to Run

### Preferred: notebook pipeline

Run `final_pipeline/src/main.ipynb` in order.

### Script entry points

From `final_pipeline/src/`:

```bash
python topic_modelling/run_bertopic.py --era 1
python sentiment_analysis/run_sentiment_analysis.py --era 1
python cross_era_analysis/run_similarity_analysis.py
```

## Further Resources

- Module-level technical details are documented in each subdirectory README.
- Manual topic review notes (labels, themes, sentiment, cross-topic comparisons) are in `analysis/manual_topic_review.csv`.
- Some ingestion/exploration utilities are legacy and not required for the main notebook pipeline.
