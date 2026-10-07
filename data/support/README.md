# Support desk

Train and demo set for inbox routing. Each line in the JSONL files is one ticket plus three questions.

| File | Rows | Labels |
|---|---|---|
| `train.jsonl` | 80 | yes |
| `eval.jsonl` | 24 | yes, different situations from train |
| `demo/*.json` | 10 | no. `demo/expected.json` is the call sheet for the person presenting |

Questions, in this option order:

- `queue`: billing, shipping, access, product
- `escalate`: supervisor now, or not
- `priority`: Can wait, This week, Today, Right now

The label follows the ask. Tickets mention other teams on purpose.

```bash
PYTHONPATH=src .venv/bin/python -m decision_model.datasets.support
PYTHONPATH=src .venv/bin/python -m decision_model.train \
  --model qwen \
  --data data/support/train.jsonl \
  --eval data/support/eval.jsonl \
  --epochs 30 \
  --lr 1e-4 \
  --out runs/support/checkpoint.pt
PYTHONPATH=src .venv/bin/python -m decision_model.demo \
  --checkpoint runs/support/checkpoint.pt
```

The checkpoint from that command is already at `runs/support/checkpoint.pt`. Queue was correct on all ten demo tickets. Escalate missed the supervisor cases, and Right now often comes back as Today.

Regenerate the files from `src/decision_model/datasets/support.py` with the first command. Keep the queue order when you send a request. Sorting the options changes the model input.
