# RLLA SFT data preparation

`prepare_rlla_sft.py` deterministically samples 400 rows from the RLLA RL
training parquet, converts the original `system + user` prompt into
`system + user + assistant`, and uses `extra_info.output` as the supervised
assistant target. The original RLLA parquet files are never modified.

Default inputs and outputs:

```text
toolcall_rl/rlla_train_rl/data/train.parquet
toolcall_rl/rlla_train_rl/data/test.parquet
               |
               v
toolcall_rl/rlla_train_rl/data/sft_400/train.parquet  # 400 sampled rows
toolcall_rl/rlla_train_rl/data/sft_400/test.parquet   # all 80 validation rows
toolcall_rl/rlla_train_rl/data/sft_400/manifest.json  # seed and source positions
```

Run from `corrgrpo-src` with:

```bash
python \
  toolcall_rl/rlla_train_rl/rlla_data_process/prepare_rlla_sft.py
```

Rebuild deterministically with a different seed:

```bash
python \
  toolcall_rl/rlla_train_rl/rlla_data_process/prepare_rlla_sft.py --seed 123 --overwrite
```
