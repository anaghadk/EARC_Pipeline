"""
config.py
──────────
Top-level config re-export.

Each module has its own *_config.py for module-specific parameters.
This file re-exports the shared constants that cross module boundaries
(e.g. K_BY_TYPE used by both Module 1 and Module 2's scorer).

Import from here in pipeline.py and evaluation scripts.
Import from retrieval.retrieval_config inside the retrieval module itself.
"""

from retrieval.retrieval_config import (
    AUX_DEPS,
    BM25_PATH,
    CHUNKS_DIR,
    CLAUSE_DEPS,
    COMPARISON_TOKENS,
    CONTENT_POS,
    DESCRIPTIVE_WH,
    DETERMINERS,
    EMBED_DIM,
    EMBED_MODEL,
    FACTOID_WH,
    FAISS_PATH,
    FRAGMENT_START_CHARS,
    K_BY_TYPE,
    MAX_SENT_TOKENS,
    METADATA_DIR,
    MIN_SENT_TOKENS,
    NEGATION_TOKENS,
    RRF_K,
    SPACY_BATCH_SIZE,
    SUBSTANTIVE_ENT_TYPES,
)

__all__ = [
    'BM25_PATH', 'CHUNKS_DIR', 'EMBED_DIM', 'EMBED_MODEL',
    'FAISS_PATH', 'K_BY_TYPE', 'MAX_SENT_TOKENS', 'METADATA_DIR',
    'MIN_SENT_TOKENS', 'RRF_K', 'SPACY_BATCH_SIZE',
    'SUBSTANTIVE_ENT_TYPES', 'CONTENT_POS', 'FACTOID_WH',
    'DESCRIPTIVE_WH', 'AUX_DEPS', 'CLAUSE_DEPS', 'COMPARISON_TOKENS',
    'NEGATION_TOKENS', 'FRAGMENT_START_CHARS', 'DETERMINERS',
]
