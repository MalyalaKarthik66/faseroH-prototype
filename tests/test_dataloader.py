import pandas as pd

from data.dataloader import build_dataloaders


def test_build_dataloaders_shapes_and_splits(tmp_path):
    csv_path = tmp_path / "tiny_dataset.csv"
    df = pd.DataFrame(
        [
            {"expression": "x", "taylor": "x", "split": "train"},
            {"expression": "x + 1", "taylor": "x + 1", "split": "train"},
            {"expression": "sin(x)", "taylor": "x", "split": "val"},
            {"expression": "exp(x)", "taylor": "x + 1", "split": "test"},
        ]
    )
    df.to_csv(csv_path, index=False)

    loaders, _, _ = build_dataloaders(
        csv_path=str(csv_path),
        batch_size=2,
        max_vocab_size=64,
        max_input_len=20,
        max_target_len=20,
        num_buckets=2,
    )

    assert set(loaders.keys()) == {"train", "val", "test"}

    batch = next(iter(loaders["train"]))
    assert batch["src_ids"].ndim == 2
    assert batch["tgt_ids"].ndim == 2
    assert batch["src_len"].ndim == 1
    assert batch["tgt_len"].ndim == 1
    assert len(batch["expression"]) == batch["src_ids"].shape[0]
    assert len(batch["target"]) == batch["tgt_ids"].shape[0]
