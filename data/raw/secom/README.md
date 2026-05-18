# SECOM raw files

Source: UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/179/secom

Two files expected in this directory:

- `secom.data` — 1567 rows × 590 sensor measurements (space-separated, NaN as `NaN`)
- `secom_labels.data` — 1567 rows × 2 columns: pass/fail label (`-1` = pass, `1` = fail) and a timestamp

Download command (also documented in the top-level README):

```bash
curl -L -o data/raw/secom/secom.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data
curl -L -o data/raw/secom/secom_labels.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data
```
