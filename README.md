# Premier League Knowledge Graph Pipeline

A complete knowledge graph pipeline for Premier League football data, built as a final project for the ESILV Web Mining course. Covers the full lifecycle: data acquisition, KB construction, entity alignment, Wikidata expansion, SWRL reasoning, knowledge graph embeddings, and RAG-based question answering.

## Project Structure

```
├── src/
│   ├── crawl/          # Web crawler and data cleaning
│   ├── ie/             # Named Entity Recognition (spaCy)
│   ├── kg/             # KG construction, alignment, Wikidata expansion
│   ├── reason/         # SWRL reasoning (OWLReady2)
│   ├── kge/            # Knowledge Graph Embeddings (PyKEEN)
│   └── rag/            # RAG pipeline (NL→SPARQL) + Streamlit UI
├── data/
│   ├── raw/            # Raw crawled HTML
│   └── processed/      # Cleaned JSON data
├── kg_artifacts/       # Ontology, alignment, expanded KB, stats
├── kge_data/           # Train/valid/test splits for KGE
├── models/             # Trained KGE model checkpoints
├── reports/            # Final report + t-SNE visualization
├── docs/               # Lab materials (TD2, TD3)
└── .env.example        # API key template
```

## Quick Start

### Prerequisites
- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An Anthropic API key (for RAG module only)

### Setup

```bash
# Create virtual environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up API key (only needed for RAG steps 11-12)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Pipeline Steps

The pipeline runs in order. Each step produces output files consumed by later steps.

> **Shortcut**: The repo includes pre-generated `kg_artifacts/` and `kge_data/`, so you can skip steps 1-9 and jump directly to step 10 (KGE training) or step 11 (RAG).

| Step | Command | Time | Output |
|------|---------|------|--------|
| 1. Crawl Wikipedia | `python -m src.crawl.crawler` | ~2 min | `data/raw/`, `data/processed/teams.json` |
| 2. Clean data | `python -m src.crawl.cleaner` | ~10 sec | `data/processed/teams_clean.json` |
| 3. NER | `python -m src.ie.ner` | ~30 sec | `data/processed/ner_results.json` |
| 4. Build ontology | `python -m src.kg.ontology` | ~5 sec | `kg_artifacts/ontology.ttl` |
| 5. Build RDF graph | `python -m src.kg.build_rdf` | ~10 sec | `kg_artifacts/football_kg.nt` |
| 6. Entity alignment | `python -m src.kg.alignment` | ~3 min | `kg_artifacts/alignment.ttl` |
| 7. Wikidata expansion | `python -m src.kg.expand` | **~30 min** | `kg_artifacts/expanded.nt` (68k triples) |
| 8. SWRL reasoning | `python -m src.reason.swrl_rules` | ~15 sec | Console output |
| 9. KGE data prep | `python -m src.kge.prepare` | ~20 sec | `kge_data/train.txt`, `valid.txt`, `test.txt` |
| 10. KGE training | `python -m src.kge.train_eval` | **~5 min** | `models/`, `reports/tsne_embeddings.png` |
| 11. RAG evaluation | `python -m src.rag.rag_pipeline --run-all` | ~2 min | Console output (needs API key) |
| 12. Streamlit demo | `streamlit run src/rag/app.py` | — | Web UI at `localhost:8501` (needs API key) |

### Running Everything from Scratch

```bash
python -m src.crawl.crawler
python -m src.crawl.cleaner
python -m src.ie.ner
python -m src.kg.ontology
python -m src.kg.build_rdf
python -m src.kg.alignment
python -m src.kg.expand          # ~30 min (Wikidata queries)
python -m src.reason.swrl_rules
python -m src.kge.prepare
python -m src.kge.train_eval     # ~5 min (CPU)
python -m src.rag.rag_pipeline --run-all   # needs ANTHROPIC_API_KEY
```

## Key Results

| Metric | Value |
|--------|-------|
| KB size (after expansion) | 68,723 triples |
| Unique entities | 23,952 |
| KGE best model | ComplEx (MRR: 0.053, Hits@10: 0.105) |
| RAG queries answered | 2/5 (limited by KB coverage) |

See `reports/final_report.md` for the full analysis.

## Data Sources

- **Wikipedia** (English) — Premier League team and player pages
- **Wikidata** SPARQL endpoint — KB expansion via batch queries

## Tech Stack

- **Crawling**: requests, BeautifulSoup4
- **NER**: spaCy (en_core_web_sm)
- **RDF/SPARQL**: rdflib, SPARQLWrapper
- **Reasoning**: OWLReady2
- **KGE**: PyKEEN (TransE, ComplEx)
- **RAG**: Anthropic Claude API
- **UI**: Streamlit

## License

MIT
