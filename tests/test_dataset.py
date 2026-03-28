from data.generate_taylor_dataset import build_dataset


def test_dataset_generation_small():
    df = build_dataset(size=50, max_order=4, seed=0)
    assert not df.empty
    assert "expression" in df.columns
    assert "taylor" in df.columns
