# Who has the ball? — More Than A Hack 2027, Phase 1

One label per frame of the tactical video (`H` home in possession, `A` away in possession,
`D` dead ball, `X` not graded), built only from the provided automatic data. CPU only, ~90 s.

## Run

```bash
pip install -r requirements.txt
# data/ laid out exactly as delivered: data/tactical/..., data/broadcast/...
python src/pipeline.py --data data --out submission/<team_name>.csv
python <challenge-repo>/evaluation/validate_submission.py submission/<team_name>.csv
```

`MTAH_DATA_DIR` can replace `--data`. `--no_broadcast` runs on the tactical feed alone.
The output is deterministic: the same inputs and pinned versions give the same file, byte for byte.

## Files

| file | what |
|---|---|
| `src/pipeline.py` | the whole pipeline, end to end |
| `src/dead_ball.py` | dead-ball features, algorithmic pseudo-labels, classifier, stoppage segmentation |
| `src/possession_tracking.py` | tracking loader, ball control, hysteresis, gap model, two-feed sync |
| `TECHNICAL.md` | the technical document |

The match data is not in this repository (see `.gitignore`).
