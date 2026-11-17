# Technical document — Who has the ball?

## 1. Approach

Possession is decided in two layers: **is the ball alive?** then **which team owns it while it is?**
Both are computed from the provided automatic tracking only, by rules with a football meaning plus
two small gradient-boosting models trained on *algorithmically generated* labels. No human label of
this game is used anywhere, and the public ground truth was used only to confirm that the pipeline
is wired correctly (labels, frame alignment, halves) — never to choose a parameter or a design.

Pipeline (`src/pipeline.py`):

1. **Load** the tactical ATD (positions in metres on a 105.3 × 68.0 pitch) and the graded halves
   published in DATA.md.
2. **Sync the broadcast feed** (optional input). Ball paths are matched in 30-s windows to find one
   constant offset per half (the broadcast drops half-time), refined to the exact frame; every
   broadcast frame is then marked live / not live by comparing player positions. Where both cameras
   see the ball but disagree on its ground position by > 3 m, the ball is **in the air** (each
   homography projects a high ball onto the grass at a different spot), so nobody is controlling it.
3. **Dead ball.**
   - *Pseudo-labels by fixed rules*: certainly dead = ball > 1 m outside the pitch; both teams lined
     up in their own halves for ≥ 5 s (kick-off); the first second of every Smart Tagging set
     piece (automatic tags). Certainly alive = ball inside the pitch, on a player, and within 4 s on
     another player ≥ 5 m away (being played), at least 10 s from any dead anchor.
   - *Classifier*: gradient boosting on **player-movement features only** (mean/median/90th-percentile
     speed, share of walking and sprinting players, collective acceleration, team spread, number of
     tracked players; 1/3/8-s windows plus the previous and next 3 s). Ball features were removed on
     purpose: every "alive" anchor has a visible ball, so a model with ball features learns
     "ball missing = dead" (it called 99 % of ball-missing frames dead).
   - *Segmentation*: 1-s smoothing, hysteresis (enter 0.6 / leave 0.4), dead stretches ≥ 2 s,
     live stretches ≥ 3 s; the ball outside the pitch always counts as dead.
4. **Ball control** on live frames: a team controls the ball when one of its players is within
   1.5 m and the other team is at least 0.4 m further away; ball gaps ≤ 1 s are interpolated;
   airborne frames never count.
5. **Possession**, restarted inside every live stretch (a dead ball ends a possession by
   definition): the first team to control the ball after the restart owns it from the restart;
   after that, possession changes only when the other team controls the ball for 0.8 s
   (near-continuously, gaps ≤ 0.4 s), back-dated to the start of that control — so a pass in
   flight belongs to the passer and a deflection is not a change.
6. **Live frames without a usable ball** (~35 % of live frames): a gap model predicts the team
   from the players' shape (depth, length, width, spread, speed of each team relative to its
   direction of play) plus who had the ball before and after the gap and how long ago / ahead. It
   is trained on *simulated* gaps — the ball is hidden around frames where possession is known —
   which also gives an honest held-out accuracy (85 % vs 67 % for carrying the last team forward).
   Its output is smoothed over 10 s inside each gap.
7. **Labels**: dead → `D`, live → `H`/`A`, outside the two halves → `X`.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | everything |
| broadcast ATD | sync + airborne-ball detection only |
| Smart Tagging | set-piece start times as dead-ball anchors; phase tags as a coarse internal check |
| homography | not needed (tracking is already in pitch coordinates); used to confirm projection direction |
| videos | not used |

Tried and dropped: copying missing ball positions from the broadcast into the tactical feed (both
cameras lose the ball at the same moments; the borrowed positions are often airborne balls);
positive-unlabelled learning for the dead ball (the anchors are not separable enough from movement
alone, so the correction factor was unstable); a dead-ball model with ball features (shortcut above).

## 3. Training data

No hand-labelled data, from this or any other game. Both models are trained on labels produced by
the fixed rules above, computed on this game's automatic data:
- dead-ball model: ~8 k "dead" and ~46 k "alive" frames from the pseudo-label rules;
- gap model: 250 k frames where possession is known from ball control, with the ball hidden.

## 4. How parameters were chosen

From football definitions and data-internal checks, never from the public ground truth:
1.5 m control radius and 0.8 s control duration (what "controlled" means: a deflection or a single
touch is shorter; sweeping it against the *automatic* Smart Tagging phases peaked at 0.8–1.0 s);
3 m airborne threshold (same sweep); walking-pace and minimum-duration rules for stoppages.
Plausibility checks: 128 stoppages, median 9.7 s, 35 % of graded time dead; 92 % of the Smart
Tagging set pieces fall inside a detected stoppage; 276 possessions averaging 14.9 s.

## 5. Possession rules implemented

- Loose / contested ball: stays with the last team in control.
- Deflections, single touches, aerial duels without control: no change (< 0.8 s of control, or
  airborne).
- Restarts: a new possession starts at the restart and belongs to the first team to control the
  ball; dead-ball ownership is not modelled (`D`).
- Ball in flight: belongs to the passer until someone controls it.

## 6. Limitations

- Stoppages inside the pitch (fouls, offsides) are detected from player movement only; quick
  restarts and play-on decisions are the main source of error, as are stoppage boundaries (±1–2 s).
- Short possessions (< 0.8 s of control) are merged into their neighbours by design.
- The automatic tracking mislabels some tracks' teams (e.g. a goalkeeper slot in the wrong goal).
- Next steps: ball-stationarity cues for free kicks, restart detection from ball kinematics,
  per-player track repair, and video-based ball detection for the 41 % of frames without a ball.

## 7. Reproduction

See README.md. ~90 s on a laptop CPU, deterministic output with the pinned versions.
