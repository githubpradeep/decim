# Decision model

This repository is a Jev-style decision model. A request carries one piece of evidence, called the state, and one or more typed questions. The model returns a probability for every answer you allowed. It does not write a sentence.

The production backbone is a frozen pretrained causal language model. The default is `Qwen/Qwen3-0.6B-Base`. A small pointer head, trained here, reads that model's hidden states and scores the options. A scratch transformer exists only so the unit tests can run without a download.

The walkthrough below follows one ticket from the JSON request to the JSON answer, then covers the dataset, the training loop, and the checkpoint.

## Repository map

| Path | Role |
|---|---|
| `src/decision_model/schema.py` | Validate the request. Turn choice, yes/no, and score into option lists. |
| `src/decision_model/render.py` | Turn one question into a token row. Record which positions the head reads. |
| `src/decision_model/pretrained.py` | Load Qwen, Gemma, or LFM. Attach delimiter tokens. Freeze the backbone. |
| `src/decision_model/model.py` | Pointer head, scratch backbone, batching, temperature fit. |
| `src/decision_model/confidence.py` | Turn a probability vector into a choice, a yes/no value, or a score. |
| `src/decision_model/data.py` | Synthetic support tickets and JSONL loading. |
| `src/decision_model/train.py` | Train the pointer head and save a checkpoint. |
| `src/decision_model/decide.py` | Score one request. |
| `src/decision_model/serve.py` | `POST /v1/systemone`. |
| `tests/test_pointer.py` | Schema, confidence arithmetic, mask-free isolation, and a tiny causal LM. |

## End-to-end path

1. `parse_request` checks the JSON and builds one `Question` per entry.
2. `render_question` builds a separate token row for each question. The row contains the state and that question only.
3. The causal backbone returns a hidden vector at every token. Qwen's weights are frozen, and those vectors are detached.
4. The pointer head scores the hidden state at the decide token against the hidden state at each option's closing token.
5. Softmax, inside that question only, turns the scores into probabilities.
6. `format_answer` turns the probabilities into a choice name, a yes probability, or an expected level. This step is ordinary arithmetic.

Question ids such as `team` and `angry` never enter the model. They are the keys of the answer object.

## Request schema

`state` is required. It may be a string, a number, a boolean, a list, or an object. Objects and lists are flattened to labeled lines (`key: value`) before tokenization. `questions` is a non-empty object. Each question needs `type` and a non-empty `instructions` string. `instructions` may also be an object or a list; it is flattened the same way.

Three types are accepted. Swapping the shape of `criteria` is an error.

| Type | `criteria` | Options the model sees | Label while training |
|---|---|---|---|
| `choice` | Object of 2 to 255 names. A value is a description string or `null`. | `name: description`, or just `name` when the description is null. | The option name. Stored as its index in the object order. |
| `noul` | Omitted, or `{ "true": "...", "false": "..." }`. | `no`, then `yes`. A description is appended as `no: ...` or `yes: ...`. | `true` or `false`. Stored as 1 or 0. |
| `score` | Array of 2 to 10 level strings, low to high. | The level strings, in that order. Indexes are `0 .. n-1`. | The integer level index. A boolean is rejected, because `True` would otherwise be read as `1`. |

A decision request omits `label`. A training row includes it. `parse_request(..., labeled=True)` requires a label on every question.

Example decision request:

```json
{
  "state": "Customer is furious about invoice and needs help immediately.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, invoices, refunds",
        "shipping": "Packages, delays, tracking",
        "access": "Login, password, account access"
      }
    },
    "angry": {
      "type": "noul",
      "instructions": "Is the customer angry?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["Can wait", "Soon", "Immediate"]
    }
  }
}
```

The same object with `"label": "billing"`, `"label": true`, and `"label": 2` is one training row.

## How one question becomes a token row

Each question is encoded alone. Sibling questions are not in the row, so the team question cannot see the words "is the customer angry," and the anger question cannot see the team list. Isolation is by construction of the row. It does not depend on a special mask between questions.

On Qwen the five delimiter tokens already exist as single vocabulary entries:

| Role | Token | What it marks |
|---|---|---|
| State | `<\|fim_prefix\|>` | Start of the evidence. |
| Question | `<\|fim_middle\|>` | Start of the instruction. |
| Option open | `<\|box_start\|>` | Start of one allowed answer. |
| Option close | `<\|box_end\|>` | End of that answer. The pointer reads this position. |
| Decide | `<\|fim_suffix\|>` | End of the row. The pointer reads this position. |

If a backbone does not already have those five tokens as single ids, the loader adds `<state>`, `<q>`, `<opt>`, `</opt>`, and `<decide>`, resizes the embedding, and fills the new rows with the mean of the existing rows. Those new rows are then frozen with the rest of the backbone.

Caller text that contains a delimiter string is rewritten before tokenization (`<` is split off) so a user cannot invent an option boundary.

The row layout, built in `render_question`, is:

```text
<|fim_prefix|>  <state tokens>
<|fim_middle|>  <instruction tokens>
<|box_start|>   <option 1 tokens>  <|box_end|>
<|box_start|>   <option 2 tokens>  <|box_end|>
...
<|fim_suffix|>
```

`opt_index` stores the position of each `<|box_end|>`. `decide_index` stores the position of `<|fim_suffix|>`. The code checks those positions after packing and raises if a marker landed somewhere else.

For the team question on the sample ticket, the text inside the markers is:

```text
<|fim_prefix|> Customer is furious about invoice and needs help immediately.
<|fim_middle|> Which team should handle this ticket?
<|box_start|> billing: Payments, invoices, refunds <|box_end|>
<|box_start|> shipping: Packages, delays, tracking <|box_end|>
<|box_start|> access: Login, password, account access <|box_end|>
<|fim_suffix|>
```

Qwen's tokenizer then splits the words into its own subwords. The five markers stay single tokens. The anger row uses the same state and the options `no` then `yes`. The urgency row uses the level strings `Can wait`, `Soon`, and `Immediate` as the three options.

Context for a pretrained backbone is 512 tokens (`max_len`). The question suffix is kept whole. If it cannot fit in `max_len - 1` tokens, the request fails. Otherwise the state prefix is cut from the end, keeping the earliest state tokens, and the response sets `"truncated": true`. The reported `input_tokens` count the state prefix once and then add every question suffix. The forward pass still re-reads the state on every row.

## Attention mask

The mask is the ordinary causal mask. It was not changed to separate options.

For Qwen, `HFCausalBackbone.forward` builds a padding mask of shape `[batch, sequence]`: `1` where the id is a real token, `0` where it is pad. An all-pad row, which does not occur for a real question, would have its first position unmasked so the softmax stays defined. That tensor is passed as `attention_mask`. Qwen combines it with its own causal mask. A token may read itself and tokens to its left. It may not read tokens to its right.

The scratch test backbone writes the same rule explicitly: an upper-triangular boolean mask blocks future positions, and pad positions are a key-padding mask.

Inside one question row the visibility is:

| Token that is reading | State | Instruction | Earlier option | Its own text | Later option |
|---|---|---|---|---|---|
| State word | Earlier state words | Hidden | Hidden | Hidden | Hidden |
| Instruction word | Visible | Earlier instruction words | Hidden | Hidden | Hidden |
| First option | Visible | Visible | None before it | Visible | Hidden |
| Second option | Visible | Visible | Visible | Visible | Hidden |
| Decide token | Visible | Visible | Visible | Visible | Visible |

Because the model can only look left, `<|box_start|>` has not seen that option's words. `<|box_end|>` has. `<|fim_suffix|>` is last, so it has seen every option. The pointer head reads only those closing positions and the decide position.

There is no extra block between options. The second option can read the first. The decide token reads the options in list order, at different positions. That is why choice order changes the probabilities.

Questions do not need a mask to hide from each other. Each question is a separate row.

## Choice order

The allowed answers of a choice are a sequence. "Billing" in slot 1 and "billing" in slot 3 are different inputs: later options can see earlier ones, and each slot has its own position.

The synthetic set lists the teams in one fixed order on every row: billing, then shipping, then access. The label `billing` is always index 0. The head can learn the slot as well as the words.

Measured on the trained Qwen head, 6 Oct 2026. The state was `Customer is furious about invoice and needs help immediately.` The three descriptions were unchanged. Only the list order changed.

| List order | Top answer | P(billing) | P(shipping) | P(access) |
|---|---|---:|---:|---:|
| billing, shipping, access | billing | 0.986 | 0.014 | 0.000 |
| access, shipping, billing | access | 0.017 | 0.117 | 0.867 |
| shipping, billing, access | billing | 0.500 | 0.500 | 0.001 |

A score list is a scale. `Can wait`, `Soon`, `Immediate` is the order that defines indexes 0, 1, and 2. Moving Immediate to the front would renumber the scale. Yes/no is fixed as `no`, then `yes`. Order invariance is a property of a choice list: the probability of billing should stay put when billing moves.

That requires a stricter mask than the one in this repository. Each option would be allowed to read the state, the instruction, and its own text, and the other options would stay hidden. Every option would reuse the same position numbers, and the decide token would sit at one fixed position after the longest option. Training would shuffle the choice list so the label index moves with the text. None of that is implemented. The current mask leaves options visible to later tokens, and the synthetic generator never shuffles the team list.

## Pointer head

`PointerHead` is two bias-enabled linear maps, `q` and `k`, from the backbone hidden size to a pointer width. The width is 256 on a pretrained backbone and 32 on the scratch net. The scale is `1 / sqrt(pointer width)`, so 256 uses `1/16`.

For one question, with `h_decide` the hidden vector at `<|fim_suffix|>` and `h_opts` the hidden vectors at each `<|box_end|>`:

```text
logit_k = (W_k h_opt_k) · (W_q h_decide) / sqrt(d_p)
```

In code, `W_k` is `self.k` and `W_q` is `self.q`. A batched question uses the same product, broadcast over the batch. Padded option slots, used only to rectangularize a batch, are filled with the dtype's minimum after the scores are computed so they receive no softmax mass.

During training, and whenever `temperature == 1`, the logits are used as they are. At eval time, if a temperature has been fitted, the logits are divided by that temperature before the softmax. A positive temperature does not change which option wins. It only changes how peaked the distribution is.

Softmax is taken over the options of that question. The resulting probabilities sum to 1.

Qwen3-0.6B-Base, the default backbone, is a pretrained causal language model: 28 layers, grouped-query attention with 16 query heads and 8 key/value heads, context length 32,768, about 0.6 billion parameters. `AutoModel.from_pretrained` loads the layer stack. The next-word head is not loaded. The forward pass asks for `last_hidden_state` and sets `use_cache` off. Weights are loaded in float32. The device is CUDA if present, otherwise Apple MPS, otherwise CPU.

Every backbone parameter has `requires_grad = False`. In `PointerDecisionModel.forward`, a frozen backbone is run under `torch.no_grad()` and the hidden states are detached. The backward pass therefore stops at the pointer head. A training step is a full backbone forward and a small backward through `q` and `k`.

The scratch `CausalBackbone` is a 2-layer Transformer encoder used by the tests: token embedding, learned positions, a causal upper-triangular mask, and a pad mask. Its weights are trained from random initialization on the synthetic vocabulary. It is selected with `--model scratch`.

Other `--model` names:

| Flag | Hugging Face id |
|---|---|
| `qwen`, `qwen3`, `qwen-0.6b` | `Qwen/Qwen3-0.6B-Base` |
| `qwen-instruct` | `Qwen/Qwen3-0.6B` |
| `gemma`, `gemma-270m` | `google/gemma-3-270m` |
| `lfm`, `lfm2` | `LiquidAI/LFM2-350M` |
| any other string | Used as a Hugging Face id |

Gemma weights are gated. Accept the license on the model page and run `hf auth login` before `--model gemma`. An `OSError` from the hub is re-raised with that instruction.

## Dataset

With no `--data` file, `make_rows` builds synthetic support tickets. Two templates are chosen at random:

```text
Customer is {tone} about {cue} and needs help {urgency}.
Ticket mentions {cue}. Tone is {tone}. Timing is {urgency}.
```

The cue word selects the team label. The tone word selects the anger label. The urgency word selects the level index.

| Words planted in the state | Label |
|---|---|
| invoice, charged, refund, payment | `team = billing` |
| package, delayed, delivery, tracking | `team = shipping` |
| password, login, locked, username | `team = access` |
| furious, angry, unacceptable | `angry = true` |
| please, thanks, wondering | `angry = false` |
| later, whenever, eventually | `urgency = 0` (Can wait) |
| soon, today, waiting | `urgency = 1` (Soon) |
| asap, immediately, outage | `urgency = 2` (Immediate) |

Choice descriptions are constant:

```text
billing:  Payments, invoices, refunds
shipping: Packages, delays, tracking
access:   Login, password, account access
```

Anger is always the instruction "Is the customer angry?" Urgency levels are always "Can wait", "Soon", "Immediate". The team list order is always billing, shipping, access.

The default pretrained run draws 120 rows with seed 0. The first 80 percent, 96 rows, are the training set. The last 24 rows are the held-out set. Each row contains three labeled questions, so training sees 288 questions and evaluation sees 72. The scratch default draws 240 rows instead.

A custom file is JSONL: one JSON object per line, same shape as a request, with a `label` on every question. Empty lines are skipped. The same 80/20 prefix split is used. If the file is too small to hold out a tail, the training prefix is reused for evaluation. Pass `--eval` to keep a prepared held-out file intact instead of cutting the training file.

## Support desk dataset

`data/support/` is the set to train and demo. It is a commerce inbox, not the keyword generator. Each ticket is a `channel`, `customer`, `subject`, and `body`. Three questions are attached to every ticket, with the same wording and the same option order:

| Question | Type | Answers |
|---|---|---|
| `queue` | choice | `billing`, `shipping`, `access`, `product` |
| `escalate` | yes/no | A supervisor is required now |
| `priority` | score | Can wait, This week, Today, Right now |

The label is the ask, not the vocabulary. A shipping ticket can contain "refund" and "invoice" because the customer is telling you not to refund them. An access ticket can mention an invoice they still intend to pay. Train, eval, and demo are different situations: 80 training tickets, 24 held-out tickets, and 10 unlabeled demo tickets.

Escalate is yes for a legal threat, an account compromise, a failure that already came back after an earlier attempt, or a site-wide outage. A courier at the door is priority Right now and escalate no.

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

Thirty epochs at learning rate `1e-4` is the run that fits this file. The default eight epochs at `3e-4` underfit it. On that longer run the training-set top label was choice 0.963, yes/no 0.863, score 0.900. Held-out, on situations that do not appear in training:

| Question | Held-out accuracy | Chance |
|---|---:|---:|
| Queue, 4 options | 0.708 | 0.250 |
| Escalate | 0.875 | 0.500 |
| Priority, 4 levels | 0.750 | 0.250 |

The fitted temperature was 2.771. On the ten demo tickets the queue call was right every time, including tickets that mention a delivery while asking for a refund, and tickets that mention a carrier while asking for a page fix. Escalate missed the three supervisor cases. Priority often stops at Today when the call sheet says Right now.

The demo command prints each live ticket, the probabilities, and whether the call matches `data/support/demo/expected.json`. That file is for the person running the demo. It is not sent to the model. One ticket on its own goes through `decision_model.decide` or `POST /v1/systemone`.

Send the four queues in the order above. This model does not hide options from each other, so a client that sorts the criteria changes the input.

The scratch vocabulary is the whitespace words of the training text, plus seven reserved ids (`pad`, `unk`, `state`, `q`, `opt`, `opt_end`, `decide`). Unknown words at decision time become `unk`. The pretrained path does not build this vocabulary. It uses the backbone tokenizer.

## Training

`python -m decision_model.train` defaults to Qwen. The pretrained and scratch settings differ:

| Setting | Pretrained (`--model qwen`) | Scratch (`--model scratch`) |
|---|---|---|
| Epochs | 8 | 12 |
| Synthetic rows | 120 | 240 |
| Batch size | 4 | 32 |
| Learning rate | 0.0003 | 0.002 |
| Pointer width | 256 | 32 |
| Max tokens | 512 | 256 |
| Device | CUDA, else MPS, else CPU | CPU |
| What is updated | `q` and `k` only | The whole scratch net and the head |

Shared optimizer settings: AdamW, weight decay `0.01`, gradient clip `1.0`, seed `0` for both PyTorch and Python's `random`. Each epoch shuffles the question rows. Questions from the same ticket can land in different batches. The loss is cross-entropy on the gold index. Padded option logits are the dtype minimum, so they do not compete.

Cross-entropy pushes probability onto the labeled index: index of the choice name, `1` if the yes/no label is true, or the score level index. The label string itself is not an input token beyond the option text that was already in the row.

On a pretrained run the mean training loss is printed each epoch. The 8-epoch Qwen run on 6 Oct 2026, learning rate 0.0003, produced:

| Epoch | Mean cross-entropy |
|---:|---:|
| 1 | 1.7478 |
| 2 | 1.1113 |
| 3 | 0.7307 |
| 4 | 0.8104 |
| 5 | 0.9028 |
| 6 | 0.7715 |
| 7 | 0.5600 |
| 8 | 0.5902 |

Held-out top-label accuracy on the 24 tickets from that run:

| Question | Accuracy | Chance if the model guessed |
|---|---:|---:|
| Choice, 3 teams | 0.375 | 0.333 |
| Yes/no, angry | 0.917 | 0.500 |
| Score, 3 levels | 0.583 | 0.333 |
| All questions | 0.625 | |

Choice on this short run is close to chance once the ticket is new. Yes/no is the question the head learned. The sample ticket, which matches the training template and the fixed option order, was answered as billing, angry 0.877, urgency 1.944.

## Temperature

After the epochs, `fit_temperature` sets the backbone to eval, forces temperature 1, and records logits on the held-out questions. It searches 40 temperatures on a geometric grid from 0.05 to 5.0 and keeps the value with the lowest cross-entropy. Training itself always saw temperature 1, because `PointerHead` skips the division while the module is in train mode.

The Qwen run selected 5.0, the top of the grid. The raw scores were sharper than the held-out error rate. Dividing by 5 flattens the distribution. It does not change the winning option. A threshold should be chosen again after any change to the data, the option text, or the option order.

## How an answer is built

Probabilities are renormalized, then rounded to 6 decimal places in the JSON.

**Choice.** The winner is the option with the highest probability. Ties take the earliest option. Confidence for `K > 1` options is

```text
(p_max - 1/K) / (1 - 1/K)
```

One option would have confidence 1. The schema requires at least two. On the sample, billing had probability 0.986, so

```text
(0.986 - 1/3) / (1 - 1/3) = 0.979
```

Confidence describes how concentrated the distribution is. It is not a measured chance that the winner is correct.

**Yes/no.** The returned `noul` field is the probability of the second option, yes. There is no confidence field. A value near 0.5 means the model splits `no` and `yes`. It does not mean the anger is medium. The sample returned 0.877.

**Score.** Level `i` has weight `i`, starting at 0. The score is

```text
sum_i  i * p_i
```

The sample probabilities were about 0.028, 0.001, and 0.971, so

```text
0*(0.028) + 1*(0.001) + 2*(0.971) = 1.944
```

That is a position on the three level names, close to Immediate. It is not a count of hours or a percent. Two different spreads can share one score, so `probabilities` and `legend` are part of the answer. Score confidence is

```text
max(0, 1 - E|level - mode| / D)
```

`mode` is the level with the highest probability. `D` is the mean absolute distance of a uniform distribution from the middle of the levels. For three levels the middle is 1 and `D` is 2/3. All mass on one level gives confidence 1. A flat distribution gives 0.

`usage.output_tokens` is 0. Nothing is generated. `latency_ms` is the wall time of parsing, rendering, the forward pass, and formatting.

## Checkpoint and how to run

A pretrained checkpoint stores the pointer head, the temperature, the Hugging Face model id, and the `TrainConfig`. It does not store Qwen. Loading downloads or reuses the cached backbone, rebuilds the head, and copies the saved matrices in. A scratch checkpoint stores the full network and the word vocabulary.

```bash
# Support-desk demo: frozen Qwen, pointer head, prepared train and eval files.
PYTHONPATH=src .venv/bin/python -m decision_model.train \
  --model qwen \
  --data data/support/train.jsonl \
  --eval data/support/eval.jsonl \
  --epochs 30 \
  --lr 1e-4 \
  --out runs/support/checkpoint.pt

# Score the ten live tickets against the expected calls.
PYTHONPATH=src .venv/bin/python -m decision_model.demo \
  --checkpoint runs/support/checkpoint.pt

# One request.
PYTHONPATH=src .venv/bin/python -m decision_model.decide \
  --checkpoint runs/support/checkpoint.pt \
  data/support/demo/01-extra-capture-on-order-7781.json

# Local server, POST /v1/systemone
PYTHONPATH=src .venv/bin/python -m decision_model.serve \
  --checkpoint runs/support/checkpoint.pt --port 8009
```

`--data tickets.jsonl` trains on your rows. `--eval heldout.jsonl` uses that file as the held-out set instead of cutting 20 percent off `--data`. `--epochs`, `--rows`, and `--seed` override the defaults. `--out` sets the checkpoint path. The default path is `runs/<model name>/checkpoint.pt`. Omitting `--data` still trains on the synthetic keyword tickets.

`POST /v1/systemone` reads a JSON body and returns the `decide` object. A schema error is HTTP 422 with `{"error": "..."}`. Invalid JSON is HTTP 400. The server binds to `127.0.0.1`.

Install from `pyproject.toml`. The runtime dependencies are PyTorch, NumPy, and Transformers. Python 3.11 or newer is required. Tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_pointer
```

The tests cover the confidence formulas, swapped `criteria` shapes, question isolation, unforgeable decide tokens, a causal Qwen2-shaped tiny backbone, and a scratch-net training run. They do not download Qwen.

## What this repository leaves out

- The attention mask does not hide options from each other, and choice lists are not shuffled in the synthetic data. Choice probabilities depend on list order.
- Qwen is frozen. There is no LoRA and no update of the backbone.
- Reinforcement learning for calibrated decisions is not implemented. Calibration here is one temperature chosen on the held-out split.
- The synthetic tickets remain the smoke test used when `--data` is omitted. Choice accuracy on that 24-ticket holdout was 0.375. The support-desk files in `data/support/` are the practical training set.
