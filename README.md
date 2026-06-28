# EARC Pipeline

**Evidence-Aware Context Compression for Token-Efficient RAG**

A 13-stage RAG pipeline that retrieves, scores, selects, and compresses evidence sentences before generation.

---

## Repository Structure

```
earc-pipeline/
├── retrieval/                  ← Module 1 (Anagha) — Stages 1–3
│   ├── __init__.py
│   ├── retrieval_config.py     ← all constants and closed linguistic sets
│   ├── sentence_object.py      ← shared SentenceObject dataclass
│   ├── loader.py               ← load_corpus_artifacts()
│   ├── query_analyser.py       ← Stage 1: QueryAnalyzer
│   ├── bm25_retriever.py       ← Stage 2a: BM25 keyword retrieval
│   ├── dense_retriever.py      ← Stage 2b: FAISS dense retrieval
│   ├── hybrid_retriever.py     ← Stage 2c: RRF fusion
│   ├── segmenter.py            ← Stage 3: sentence segmentation
│   └── retrieval_pipeline.py  ← RetrievalLayer (top-level interface)
│
├── scoring/                    ← Module 2 — Stages 4–6
│   ├── query_first_embedder.py
│   ├── multi_signal_scorer.py
│   ├── redundancy_remover.py
│   └── scoring_pipeline.py
│
├── selection/                  ← Module 3 — Stages 7–9
│   ├── graph_builder.py
│   ├── budget_allocator.py
│   ├── evidence_guard.py
│   └── selection_pipeline.py
│
├── generation/                 ← Module 4 — Stages 10–12
│   ├── prompt_builder.py
│   ├── llm_generator.py
│   ├── citation_manager.py
│   └── generation_pipeline.py
│
├── evaluation/                 ← Stage 13: evaluation metrics
├── data/                       ← corpus artifacts (not committed)
├── tests/
│   ├── test_query_analyser.py
│   └── test_segmenter.py
├── notebooks/                  ← Colab notebooks for development
├── pipeline.py                 ← end-to-end EARCPipeline entry point
├── config.py                   ← top-level config re-export
└── requirements.txt
```

---

## Module 1: Retrieval Layer

**Owner:** Anagha  
**Stages:** 1 (Query Analysis), 2 (Hybrid Retrieval), 3 (Sentence Segmentation)  
**Output:** `List[SentenceObject]` handed in-RAM to Module 2

### Stage 1 — Query Analysis & Classification

`QueryAnalyzer.analyze(query)` returns:

| Field | Type | Description |
|---|---|---|
| `query` | str | original query (propagated to all modules) |
| `query_type` | str | `'factoid'` / `'multi_hop'` / `'descriptive'` |
| `keywords` | List[str] | lemmatised content words for BM25 |
| `entities` | List[str] | named entity strings for entity guard |
| `has_negation` | bool | negation/exclusion signal for Module 2/3 |

Classification priority: `multi_hop` > `factoid` > `descriptive`

multi_hop signals: 2+ substantive entities, 2+ finite verbs + WH-word, 2+ WH-words, comparison tokens, compound conjunction.

### Stage 2 — Adaptive Hybrid Retrieval

BM25 (sparse) + FAISS IndexFlatIP (dense) fused by Reciprocal Rank Fusion.

Retrieval depths scale with `query_type` via `K_BY_TYPE` in `retrieval_config.py`.

### Stage 3 — Sentence Segmentation

Batch spaCy pipeline (`nlp.pipe()`) over all retrieved chunks.
Filters: fragment detection, length bounds [5, 150] tokens.
Each sentence becomes a `SentenceObject` with `embedding=None`.

---

## SentenceObject

The shared data model flowing through the entire pipeline.

```python
@dataclass
class SentenceObject:
    # Module 1 sets these:
    sentence_id           : str            # dataset:doc_id:chunk_id:position
    text                  : str
    doc_id, dataset, title: str
    position, retrieval_rank, chunk_id : int
    year                  : Optional[int]
    bm25_score, faiss_score, retrieval_score : float
    contains_query_entity : bool
    token_count           : int

    # Module 2 fills these:
    embedding             : Optional[np.ndarray]  # None until Module 2
    semantic_score, evidence_score, ...           : float

    # Module 3 sets this:
    force_include         : bool
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run Tests

```bash
pytest tests/ -v
```

## Debug Export (Colab only)

To share Module 1 output with teammates without running the full pipeline:

```python
# In notebooks/module1_dev.ipynb
import pickle, re
from pathlib import Path

OUTPUT_DIR = Path('/content/drive/MyDrive/RAG_Project/earc_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def export_for_module2(query):
    sentences, query_info = retrieval_layer.retrieve(query)
    slug = re.sub(r'[^a-z0-9]+', '_', query.lower())[:40]
    out_path = OUTPUT_DIR / f'm1_{slug}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({'query': query, 'query_info': query_info, 'sentences': sentences}, f)
    print(f'Saved {len(sentences)} SentenceObjects → {out_path.name}')
    return out_path
```
