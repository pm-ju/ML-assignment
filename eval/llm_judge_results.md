# LLM-as-Judge Results

These scores were produced by applying `eval/llm_judge_prompt.md` to the generated `output/visualization.png`.

| Query | Garment | Color | Binding | Context | Overall | Notes |
|---|---:|---:|---:|---:|---:|---|
| A person in a bright yellow raincoat | 5 | 5 | 5 | 4 | 5 | Top-3 results are clearly yellow coats/raincoats. Rank 4 is still a yellow coat. Rank 5 is only a yellow top, so slightly weaker. Excellent retrieval overall. |
| Professional business attire inside a modern office | 4 | 5 | 5 | 3 | 4 | Business clothing is mostly correct, but only a couple of images actually depict an office environment. Some images are indoors but not obviously offices. |
| Someone wearing a blue shirt sitting on a park bench | 4 | 5 | 5 | 3 | 4 | Blue clothing is retrieved consistently. However, only some results resemble a park or bench scene; others are outdoors without the requested context. |
| Casual weekend outfit for a city walk | 5 | 5 | 5 | 5 | 5 | Strong results. Nearly every retrieved image shows casual streetwear in an urban environment. This query is handled very well. |
| A red tie and a white shirt in a formal setting | 3 | 3 | 2 | 4 | 3 | Formal clothing is retrieved, but the crucial compositional constraint is not consistently satisfied. Some images lack a tie entirely or do not bind the red color to the tie. This remains the hardest query. |

## Summary

| Dimension | Average Score |
|---|---:|
| Garment correctness | 4.2 / 5 |
| Color correctness | 4.6 / 5 |
| Color-to-garment binding | 4.4 / 5 |
| Scene / Context correctness | 3.8 / 5 |
| Overall Top-K usefulness | 4.2 / 5 |
