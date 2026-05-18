# WM-811K raw files

Source: Kaggle — https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Place `LSWMD.pkl` (~400 MB) in this directory. Download manually after accepting Kaggle's dataset terms, or use the Kaggle CLI:

```bash
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw/wm811k/ --unzip
```

The pickle contains a single DataFrame with columns:

- `waferMap` — 2-D numpy arrays of variable shape
- `dieSize`, `lotName`, `waferIndex`
- `trianTestLabel` — Training / Test split annotation
- `failureType` — defect-pattern label (`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`, `Near-full`, `none`); most rows are empty (unlabeled)
