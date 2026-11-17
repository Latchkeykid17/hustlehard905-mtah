"""Dead-ball detection from tracking kinematics, trained on algorithmic pseudo-labels (no human labels)."""
import numpy as np, pandas as pd

L_PITCH, W_PITCH = 105.3, 68.0


def roll(v, w, fn="mean", center=True):
    s = pd.Series(v)
    r = s.rolling(max(1, int(w)), center=center, min_periods=1)
    return getattr(r, fn)().to_numpy()


def kinematic_features(X, Y, S, bx, by, fps):
    """Per-frame features describing how the players and ball move. Deliberately excludes where the
    ball is on the pitch, so the model can't just learn 'ball near the touchline = dead'."""
    n = len(bx)
    with np.errstate(all="ignore"):
        sp_mean = np.nanmean(S, 1)
        sp_med = np.nanmedian(S, 1)
        sp_p90 = np.nanpercentile(S, 90, axis=1)
        slow = np.nanmean(S < 1.0, 1)
        fast = np.nanmean(S > 4.0, 1)
        nplay = np.sum(~np.isnan(X), 1)
        # collective acceleration (change of mean speed)
        acc = np.gradient(roll(sp_mean, fps / 2)) * fps
        # ball kinematics
        bvis = (~np.isnan(bx)).astype(float)
        bsp = np.hypot(np.gradient(bx), np.gradient(by)) * fps
        bsp = np.where(np.isfinite(bsp), np.minimum(bsp, 40), np.nan)
        d = np.hypot(X - bx[:, None], Y - by[:, None])
        near = np.nanmin(np.where(np.isnan(d), np.inf, d), 1)
        near = np.where(np.isfinite(near), np.minimum(near, 40), np.nan)
        crowd = np.sum(d < 10, 1).astype(float)
        crowd[np.isnan(bx)] = np.nan
        # team shape
        spread_x = np.nanstd(X, 1)
        spread_y = np.nanstd(Y, 1)
    F = {}
    for name, v in dict(sp_mean=sp_mean, sp_med=sp_med, sp_p90=sp_p90, slow=slow, fast=fast, acc=acc,
                        bvis=bvis, bsp=bsp, near=near, crowd=crowd, spread_x=spread_x, spread_y=spread_y,
                        nplay=nplay.astype(float)).items():
        for w_s in (1, 3, 8):
            w = int(w_s * fps)
            F[f"{name}_m{w_s}"] = roll(v, w, "mean")
        F[f"{name}_past3"] = roll(v, 3 * fps, "mean", center=False)               # last 3 s
        F[f"{name}_next3"] = roll(v[::-1], 3 * fps, "mean", center=False)[::-1]   # next 3 s
    for name in ("sp_mean", "bsp"):
        F[f"{name}_sd3"] = roll(F[f"{name}_m1"], 3 * fps, "std")
    return pd.DataFrame(F).astype(np.float32)


def ball_out(bx, by, margin=0.5):
    return (bx < -margin) | (bx > L_PITCH + margin) | (by < -margin) | (by > W_PITCH + margin)


def runs_of(mask):
    m = np.asarray(mask, bool)
    d = np.diff(np.r_[0, m.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def pseudo_labels(X, Y, bx, by, team_cols, graded, set_piece_starts, fps, formation):
    """1 = certainly dead, 0 = certainly alive, -1 = unknown. Produced by fixed rules, no human labels.

    dead : ball clearly outside the pitch; teams lined up in their halves for a kick-off (>= 5 s);
           the first second of every Smart Tagging set piece (automatic tags).
    alive: the ball is inside the pitch, a player is on it, and within 4 s it is on another player
           >= 5 m away (it is being played), away from any dead anchor.
    """
    n = len(bx)
    lab = np.full(n, -1, np.int8)
    out = ball_out(bx, by, margin=1.0)
    dead = out.copy()
    rl = pd.Series(formation).groupby((~pd.Series(formation)).cumsum()).transform("sum").to_numpy()
    dead |= formation & (rl >= 5 * fps)
    for s in set_piece_starts:
        a = int(round(s * fps))
        dead[max(0, a):min(n, a + int(fps))] = True

    inside = (bx > 3) & (bx < L_PITCH - 3) & (by > 3) & (by < W_PITCH - 3)
    d = np.hypot(X - bx[:, None], Y - by[:, None])
    dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), 1)
    on_player = inside & (dmin < 1.0)
    idx = np.flatnonzero(on_player)
    alive = np.zeros(n, bool)
    # next on-player frame >= 5 m away within 4 s
    j = 0
    for k, i in enumerate(idx):
        # look ahead in idx
        m = k + 1
        while m < len(idx) and idx[m] - i <= 4 * fps:
            if np.hypot(bx[idx[m]] - bx[i], by[idx[m]] - by[i]) >= 5:
                alive[i:idx[m] + 1] = True
                break
            m += 1
    near_dead = roll(dead.astype(float), 10 * fps, "max") > 0
    lab[alive & ~near_dead] = 0
    lab[dead] = 1
    lab[~graded] = -1
    return lab


def fit_dead_model(F, lab, fps, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    idx = np.flatnonzero(lab >= 0)
    y = lab[idx]
    minute = idx // int(60 * fps)
    hold = minute % 5 == 0
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                l2_regularization=1.0, class_weight="balanced",
                                                random_state=seed)
    clf = mk().fit(F.iloc[idx[~hold]], y[~hold])
    p = clf.predict_proba(F.iloc[idx[hold]])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[hold], p)
    clf = mk().fit(F.iloc[idx], y)
    return clf, auc


def fit_dead_pu(F, dead_anchor, graded, fps, seed=0, n_unl=150000):
    """Positive-unlabelled learning (Elkan & Noto 2008): classifier g separates certain-dead anchors
    from every other graded frame; c = mean g on held-out anchors estimates P(anchor | dead), and
    P(dead | x) = g(x) / c. No negative labels and no assumed dead-ball share are needed."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(dead_anchor & graded)
    unl = np.flatnonzero(~dead_anchor & graded)
    if len(unl) > n_unl:
        unl = rng.choice(unl, n_unl, replace=False)
    idx = np.r_[pos, unl]
    y = np.r_[np.ones(len(pos)), np.zeros(len(unl))]
    hold = (idx // int(60 * fps)) % 5 == 0
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                l2_regularization=1.0, random_state=seed)
    g = mk().fit(F.iloc[idx[~hold]], y[~hold])
    c = float(np.mean(g.predict_proba(F.iloc[idx[hold & (y == 1)]])[:, 1]))
    g = mk().fit(F.iloc[idx], y)
    return g, c


def segment_dead(p, bx, by, graded, fps, enter=0.6, exit_=0.4, min_dead_s=2.0, min_alive_s=3.0, smooth_s=1.0):
    """Turn P(dead) into stoppages: smoothed, hysteresis thresholds, minimum durations, and the ball
    being outside the pitch always counts as dead."""
    n = len(p)
    ps = roll(np.nan_to_num(p, nan=0.0), smooth_s * fps, "mean")
    ps = np.where(ball_out(bx, by, 0.5), 1.0, ps)
    dead = np.zeros(n, bool)
    state = False
    for i in range(n):
        if not graded[i]:
            state = False
            continue
        if state and ps[i] < exit_:
            state = False
        elif not state and ps[i] > enter:
            state = True
        dead[i] = state
    # remove short alive gaps, then short dead blips
    for a, b in runs_of(~dead & graded):
        if b - a < min_alive_s * fps and a > 0 and b < n and dead[a - 1] and dead[b]:
            dead[a:b] = True
    for a, b in runs_of(dead):
        if b - a < min_dead_s * fps:
            dead[a:b] = False
    return dead & graded
