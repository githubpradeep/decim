# Decision model

One state, a map of typed questions, and a probability for every answer you allowed. The model does not write a sentence. `usage.output_tokens` is always 0.

The backbone is a frozen `Qwen/Qwen3-0.6B-Base`. A pointer head scores the options. Architecture, the token row, the causal mask, and training are in [docs/decision-model.md](docs/decision-model.md).

## Setup

Python 3.11 or newer.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
```

The first run downloads `Qwen/Qwen3-0.6B-Base`. After it is cached, prefix commands with `HF_HUB_OFFLINE=1`. Checkpoints live in `runs/` and are not in git. Public training JSONL under `data/public/` is not in git either.

## Live demos

One server loads one checkpoint.

Snake, the head trained to play:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m decision_model.serve \
  --checkpoint runs/snake/checkpoint.pt
```

Open http://127.0.0.1:8009/ . The board is written into the state. The options are the words up, down, left, and right. The page steps the top direction.

Intent reel, the joint public head:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m decision_model.serve \
  --checkpoint runs/public/joint/checkpoint.pt
```

Open http://127.0.0.1:8009/intents . `/`, `/snake`, and `/demo` are the snake page on whichever checkpoint the server loaded.

`POST /v1/systemone` takes the same JSON as `decide`. The server binds to `127.0.0.1:8009`. A schema error is HTTP 422.

## One request

```bash
PYTHONPATH=src .venv/bin/python -m decision_model.decide \
  --checkpoint runs/public/joint/checkpoint.pt \
  data/support/demo/01-extra-capture-on-order-7781.json
```

Omit the path to read JSON from stdin.

## Train the joint head

Prepare the seven public tasks, then train. `laptop` is the default mixture. `full` uses every official training row.

```bash
PYTHONPATH=src .venv/bin/python -m decision_model.datasets.public
PYTHONPATH=src .venv/bin/python -m decision_model.train_suite --dry-run
PYTHONPATH=src .venv/bin/python -m decision_model.train_suite
PYTHONPATH=src .venv/bin/python -m decision_model.train_suite --profile full
```

The checkpoint is `runs/public/joint/checkpoint.pt`. Task files, row counts, and the time estimate are in [data/public/README.md](data/public/README.md). Option order inside each file is fixed. Do not sort it.

## Train the snake head

This continues the joint head. It does not replace `runs/public/joint/checkpoint.pt`.

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m decision_model.snake_train
```

The checkpoint is `runs/snake/checkpoint.pt`. On this Mac, keep the batch at 4.

## Support desk

A separate small head, not the public model. Files and the exact command are in [data/support/README.md](data/support/README.md).

```bash
PYTHONPATH=src .venv/bin/python -m decision_model.demo \
  --checkpoint runs/support/checkpoint.pt
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```
