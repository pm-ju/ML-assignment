# My Assignment

Multimodal fashion and context retrieval using CLIP embeddings, garment regions, color profiles, scene scoring, and constraint reranking.

## Overview

This project builds a fashion search system for natural language queries like:

```text
A red tie and a white shirt in a formal setting.
```

The implementation improves over a plain CLIP search by separating the query into garment, color, scene, and style constraints. Images are indexed with global embeddings, garment-region embeddings, dominant color profiles, and scene scores.

## Project Structure

```text
ML-assigment/
|-- data/
|   |-- build_metadata.py
|   `-- prepare_dataset.py
|-- indexer/
|   |-- build_index.py
|   |-- color_extractor.py
|   |-- embedding_extractor.py
|   |-- region_extractor.py
|   `-- vector_store.py
|-- retriever/
|   |-- constraint_reranker.py
|   |-- query_parser.py
|   `-- retriever.py
|-- eval/
|   |-- evaluate.py
|   |-- llm_judge_results.md
|   |-- llm_judge_prompt.md
|   `-- queries.json
|-- output/
|-- assignment_report.tex
|-- main.py
|-- visualize.py
|-- pyproject.toml
`-- README.md
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
```

Download the fashion dataset and put it into as `test` folder in `data` directory

If Torch or TorchVision is already installed globally and causes import errors, create a fresh virtual environment and install this project there. The code is intended to run fully locally after dependencies and model weights are available.

## Dataset

Place your images and optional Fashionpedia-style annotations inside this project:

```text
data/test/
data/val_annotations.json
```

## Build Index

```bash
python -m indexer.build_index --annotations data\val_annotations.json --images data\test --output output\index.pkl --limit 3000
```

If annotations are not available, the indexer falls back to full, upper, and lower image regions:

```bash
python -m indexer.build_index --images data\test --output output\index.pkl --limit 1000
```

## Run Retrieval

```bash
python main.py
```

## Visualize Results

```bash
python visualize.py
```

## Evaluate

```bash
python -m eval.evaluate --index output\index.pkl --queries eval\queries.json --k 10 --ks 1,3,5,10 --threshold 0.75
```

The evaluator reports average constraint score, proxy Recall@1/3/5/10, Hit@K, and latency per query.

For comparison with another saved result JSON:

```bash
python -m eval.evaluate --index output\index.pkl --queries eval\queries.json --baseline path\to\result.json --k 10 --ks 1,3,5,10 --threshold 0.75
```

## LLM-as-Judge

Generate the visualization first:

```bash
python visualize.py
```

Then use `eval/llm_judge_prompt.md` with a multimodal LLM and upload `output/visualization.png`. This gives an additional human-like relevance check for garment, color, binding, context, and overall quality.

The current LLM judge output is recorded in `eval/llm_judge_results.md`.

## Method

The indexer stores:

- Global image embeddings
- Region embeddings from Fashionpedia annotations when available
- Dominant color profiles for each garment region
- Scene scores for office, street, park, home, indoor, outdoor, city, and runway

The retriever:

- Parses color-garment bindings such as `red tie` and `white shirt`
- Searches global, region, and scene vectors
- Reranks candidates with bound garment-color checks
- Adds scene and style constraints
- Returns top-k image paths and score breakdowns

## Difference From Baseline

The baseline-style approach depends mostly on global CLIP similarity and rough upper/lower crops. This version makes the difficult parts explicit: color belongs to a specific garment, scene is scored separately, and evaluation can compare constraint satisfaction.
