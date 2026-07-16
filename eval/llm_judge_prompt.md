# LLM-as-Judge Evaluation Prompt

Use this prompt with a multimodal LLM after generating `output/visualization.png`.

## Judge Task

You are judging an image retrieval system for fashion and context search. For each query, inspect the top returned images and score how well the results satisfy the prompt.

## Inputs

- Query text
- Top-k retrieved image grid
- Optional result JSON with image IDs and scores

## Scoring Rubric

Use a 1 to 5 score.

- 5: The top results clearly satisfy all important constraints.
- 4: Most top results satisfy the query, with minor misses.
- 3: Some relevant results appear, but important constraints are inconsistent.
- 2: Results are weakly related but miss major constraints.
- 1: Results are mostly unrelated.

## Judge Dimensions

Evaluate these separately:

- Garment correctness
- Color correctness
- Color-to-garment binding
- Scene or context correctness
- Overall top-k usefulness

## Output Format

Return a table with:

| Query | Garment | Color | Binding | Context | Overall | Notes |
|---|---:|---:|---:|---:|---:|---|

## Important Note

Do not reward an image only because it has the right color somewhere. If the query says "red tie and white shirt", the red should belong to the tie and the white should belong to the shirt.
