# Who has the ball? — team hustlehard905

More Than A Hack 2027, Phase 1

One label per frame of the tactical video (`H` home in possession, `A` away in possession,
`D` dead ball, `X` not graded), built only from the provided automatic data. CPU only, ~90 s.

## Run

```bash
pip install -r requirements.txt
# data/ laid out exactly as delivered: data/tactical/..., data/broadcast/...
# (or point --data at the dataset folder wherever it is)
python pipeline.py --data data --out hustlehard905.csv
python <challenge-repo>/evaluation/validate_submission.py hustlehard905.csv
```

`MTAH_DATA_DIR` can replace `--data`. `--no_broadcast` runs on the tactical feed alone.
The output is deterministic: the same inputs and pinned versions give the same file, byte for byte
(MD5 `a37aea05bf0ff9b3ab3026b011564686`).

## Files

| file | what |
|---|---|
| `pipeline.py` | the whole pipeline, end to end |
| `dead_ball.py` | dead-ball features, algorithmic pseudo-labels, classifier, stoppage segmentation |
| `possession_tracking.py` | tracking loader, ball control, hysteresis, gap model, two-feed sync |
| `TECHNICAL.md` | the technical document |

The match data is not in this repository (see `.gitignore`).
