# Public benchmark suite

Seven official datasets, one pointer head. Each file keeps its own allowed answers and its own test split. Training mixes the files. The head scores whatever options are written in the row, so choice, yes/no, and score share `runs/public/joint/checkpoint.pt`.

| Task | Question | Train | Laptop train | Score split |
|---|---|---:|---:|---|
| `trec6` | choice, 6 question types | 5,452 | 5,452 | TREC test, 500 |
| `emotion` | choice, 6 emotions | 15,956 | 15,956 | test, 1,986 |
| `sst2` | yes/no | 67,349 | 8,000 | SST-2 validation, 872 |
| `sst5` | score, 5 levels | 8,544 | 8,544 | test, 2,210 |
| `phrasebank` | score, negative to positive | 1,129 | 1,129 | test, 1,129 |
| `amazon_reviews_en` | score, 1 to 5 stars | 200,000 | 8,000 | test, 5,000 |
| `massive_intent_en` | choice, 60 intents | 11,514 | 11,514 | English test, 2,974 |

Option order is fixed inside each file. Do not sort it. Context for the joint model is 1,024 tokens, which is enough for the 60 MASSIVE intents.

```bash
PYTHONPATH=src .venv/bin/python -m decision_model.datasets.public
PYTHONPATH=src .venv/bin/python -m decision_model.train_suite --dry-run
PYTHONPATH=src .venv/bin/python -m decision_model.train_suite
PYTHONPATH=src .venv/bin/python -m decision_model.train_suite --profile full
```

## Compute

Measured on this Mac, Apple MPS, frozen Qwen3-0.6B in float32: about 0.33 seconds per step at batch 4 on short text. A joint batch is padded out to the longest row in the batch, so the planner uses 0.8 seconds per step.

| Profile | Mixture | About |
|---|---|---|
| `laptop` (default) | 58,595 rows, 4 epochs, batch 4 | 13 hours |
| `full` | 309,944 official training rows, 1 epoch, batch 4 | 17 hours |

One process, one checkpoint. Peak memory is the 0.6B model plus a batch of 4 sequences at 1,024 tokens. If that runs out of memory, the batch has to drop to 2 and the time roughly doubles.

After training, the script prints accuracy on each official eval file separately. `laptop` is the development mixture. `full` is the run that uses every official training row.
