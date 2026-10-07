"""Continue the joint pointer head on snake boards plus a slice of the public tasks.

    python -m decision_model.snake_train
"""

from __future__ import annotations

import random
from pathlib import Path

from decision_model.config import TrainConfig
from decision_model.pretrained import resolve_model_id
from decision_model.snake_data import public_sample, snake_rows
from decision_model.train import accuracy, save_checkpoint, train_model


def main() -> None:
    random.seed(0)
    public = [
        row
        for row in public_sample(Path("data/public"), per_task=80, seed=1)
        if len(row["state"]) <= 400
        and not (
            isinstance(next(iter(row["questions"].values())).get("criteria"), dict)
            and len(next(iter(row["questions"].values()))["criteria"]) > 8
        )
    ]
    train = snake_rows(2400, seed=0) + public
    random.shuffle(train)
    held = snake_rows(240, seed=1)
    model_id = resolve_model_id("qwen")
    config = TrainConfig(
        epochs=3,
        seed=0,
        model="qwen",
        model_id=model_id,
        name=model_id,
        proj_size=256,
        max_len=512,
        batch_size=4,
        lr=1e-4,
    )
    out = Path("runs/snake/checkpoint.pt")
    print(f"train rows {len(train)}  snake eval {len(held)}", flush=True)
    model, encoder = train_model(
        train,
        held,
        config,
        checkpoint_path=out,
        init_checkpoint=Path("runs/public/joint/checkpoint.pt"),
    )
    scores = accuracy(model, held, encoder, config.max_len, config.batch_size)
    save_checkpoint(out, model, encoder, config)
    print(f"snake held-out choice {scores.get('choice', 0):.3f}  noul {scores.get('noul', 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
