#!/usr/bin/env python3
"""
Per-frame team possession from Metrica / FIFA-EPTS tracking data. CPU only.

Inputs  (provided by the competition):
  <name>_atd.xml   metadata: teams, which track belongs to which team, goalkeepers, frame rate
  <name>_atd.txt   one line per video frame: frame:p1x,p1y,p1s;...;pNx,pNy,pNs:bx,by,bs
                   positions normalised 0..1 on a 105 x 68 m pitch
  (optional) the other camera feed of the same match (broadcast/tactical) -> synced automatically
  (optional) extra ball detections from yolo_ball_fill.py (uses the provided homography)
  (optional) smart_tagging.xml       analyst phases, used to measure accuracy

Method:
  1. Ball control: in each frame, the team with a player within RADIUS of the ball
     (and clearly closer than the other team) controls it. Short ball gaps are interpolated.
  2. Hysteresis: possession only changes after the other team controls the ball for a
     near-continuous MIN_SWITCH seconds; the change is back-dated to the start of that run,
     so the ball in flight belongs to the passer (last-touch rule).
  3. Frames with no trustworthy ball (~40% here: ball not tracked, or nobody near it for
     2 s): a gradient-boosting model predicts possession from the players' shape (depth,
     spread, speed of each team, direction of play) plus who had the ball before and after
     the gap and how long ago/ahead. It is trained on simulated gaps: the ball is hidden
     around frames where possession is known, so its accuracy can be measured honestly.
  4. Two feeds: synced frame-exactly by matching ball paths. Where both cameras see the ball but
     disagree on its ground position by > 3 m the ball is in the air -> nobody controls it.
     Live broadcast frames take the (better) tactical label; replays are matched to the earlier moment
     they show; close-ups/crowd shots take the real-time tactical label (the match keeps going).
  5. Match periods: kick-offs (players line up in their halves, stand still, then accelerate), half-time
     and full-time are detected; frames outside play get team_or_none = "none" (team_id is still filled).

Usage:
  # one feed
  python possession_tracking.py --xml BAR-ATH_tactical_atd.xml --txt BAR-ATH_tactical_atd.txt \
      --out tactical_possession.csv [--tags BAR-ATH_tactical_smart_tagging.xml]
  # both feeds (recommended): auto-sync, airborne-ball detection, broadcast gets synced labels
  python possession_tracking.py --xml BAR-ATH_tactical_atd.xml --txt BAR-ATH_tactical_atd.txt \
      --out tactical_possession.csv --other_xml BAR-ATH_broadcast_atd.xml \
      --other_txt BAR-ATH_broadcast_atd.txt --other_out broadcast_possession.csv \
      [--tags ...] [--extra_ball yolo_tactical/extra_ball.csv]
"""
import argparse
import io
import json
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

PITCH_L, PITCH_W = 105.0, 68.0


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- loading
def load_meta(xml_path):
    r = ET.parse(xml_path).getroot()
    teams = {t.get("id"): t.findtext("Name") for t in r.iter("Team")}
    rows = []
    for p in r.iter("Player"):
        pos = p.find(".//ProviderParameter[Name='position_type']")
        rows.append(dict(pid=p.get("id"), team=teams[p.get("teamId")],
                         gk=pos is not None and pos.findtext("Value") == "Goalkeeper"))
    fps = float(r.findtext(".//FrameRate") or 25)
    ld = r.find(".//FieldSize")
    size = (float(ld.findtext("Width")), float(ld.findtext("Height"))) if ld is not None else (PITCH_L, PITCH_W)
    return pd.DataFrame(rows), list(teams.values()), fps, size


def load_tracking(txt_path, n_players, size):
    s = open(txt_path).read().translate(str.maketrans(":;", ",,"))
    a = pd.read_csv(io.StringIO(s), header=None, dtype=np.float32, na_values=["NaN"]).to_numpy()
    del s
    frame = a[:, 0].astype(np.int64)
    P = a[:, 1:1 + 3 * n_players].reshape(len(a), n_players, 3)
    B = a[:, 1 + 3 * n_players:4 + 3 * n_players]
    L, W = size
    X, Y, S = P[..., 0] * L, P[..., 1] * W, P[..., 2]
    bx, by = B[:, 0] * L, B[:, 1] * W
    return frame, X, Y, S, bx, by


# ----------------------------------------------------------------------------- helpers
def interp_short_gaps(v, max_gap):
    s = pd.Series(v)
    isna = s.isna()
    run_len = isna.groupby((~isna).cumsum()).transform("sum")
    return s.interpolate(limit_area="inside").where(~isna | (run_len <= max_gap)).to_numpy()


def run_lengths(mask):
    """For each element, length of the run of equal mask values it belongs to."""
    m = np.asarray(mask)
    if len(m) == 0:
        return m.astype(int)
    starts = np.r_[0, np.flatnonzero(m[1:] != m[:-1]) + 1]
    lens = np.diff(np.r_[starts, len(m)])
    return np.repeat(lens, lens)


def team_min(d, cols):
    sub = d[:, cols]
    out = np.fmin.reduce(sub, axis=1) if sub.shape[1] else np.full(len(d), np.nan)
    return np.where(np.isnan(out), np.inf, out)


# ----------------------------------------------------------------------------- step 1+2
def ball_control(X, Y, bx, by, team_cols, radius, margin):
    d = np.hypot(X - bx[:, None], Y - by[:, None])
    d0, d1 = team_min(d, team_cols[0]), team_min(d, team_cols[1])
    best = np.where(d0 <= d1, 0, 1)
    dbest, dother = np.minimum(d0, d1), np.maximum(d0, d1)
    raw = np.where((dbest < radius) & (dother - dbest >= margin), best, -1)
    near = np.where(np.isfinite(dbest), np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1), -1)
    return raw, dbest, near


def hysteresis(raw, K, G):
    lab = np.full(len(raw), -1, int)
    cur, cand, cnt, cand_start, seg_start, cand_last = -1, -1, 0, 0, 0, -10 ** 9
    for i, r in enumerate(raw):
        if r < 0:
            continue
        if r == cur:
            cand, cnt = -1, 0
            continue
        if r != cand or i - cand_last > G:
            cand, cnt, cand_start = r, 0, i
        cand_last = i
        cnt += 1
        if cnt >= K:
            lab[seg_start:cand_start] = cur
            cur, seg_start, cand, cnt = r, cand_start, -1, 0
    lab[seg_start:] = cur
    return lab


# ----------------------------------------------------------------------------- step 3
def shape_features(X, Y, S, bx, by, team_cols, attack_sign):
    """Per-frame features from both teams' positions (no ball needed).
    attack_sign[t] = +1 if team t attacks towards x=105 in this frame, else -1."""
    feats = {}
    for t in (0, 1):
        x, y, s = X[:, team_cols[t]], Y[:, team_cols[t]], S[:, team_cols[t]]
        sgn = np.asarray(attack_sign[t])[:, None]
        xa = np.where(sgn > 0, x, PITCH_L - x)  # x in team t's attacking direction
        with np.errstate(all="ignore"):
            feats[f"n{t}"] = np.sum(~np.isnan(x), 1)
            feats[f"depth{t}"] = np.nanmean(xa, 1)
            feats[f"front{t}"] = np.nanmax(xa, 1)
            feats[f"back{t}"] = np.nanmin(xa, 1)
            feats[f"len{t}"] = np.nanmax(xa, 1) - np.nanmin(xa, 1)
            feats[f"wid{t}"] = np.nanmax(y, 1) - np.nanmin(y, 1)
            feats[f"sdx{t}"] = np.nanstd(xa, 1)
            feats[f"sdy{t}"] = np.nanstd(y, 1)
            feats[f"spd{t}"] = np.nanmean(s, 1)
            feats[f"spdmax{t}"] = np.nanmax(s, 1)
            feats[f"ymean{t}"] = np.nanmean(y, 1)
    f = pd.DataFrame(feats)
    for k in ["depth", "front", "back", "len", "wid", "sdx", "sdy", "spd", "spdmax"]:
        f[f"{k}_diff"] = f[f"{k}0"] - f[f"{k}1"]
    return f


# ----------------------------------------------------------------------------- two-feed sync


def player_match_cost(Xb, Yb, Xt, Yt, team_b, team_t, b_idx, t_idx):
    """Median distance (m) from each broadcast player to the nearest same-team tactical player."""
    out = np.full(len(b_idx), np.nan)
    teams = np.unique(team_b)
    for k in range(0, len(b_idx), 4000):
        bi, ti = b_idx[k:k + 4000], t_idx[k:k + 4000]
        ds = []
        for tm in teams:
            cb, ct = np.flatnonzero(team_b == tm), np.flatnonzero(team_t == tm)
            dx = Xb[bi][:, cb][:, :, None] - Xt[ti][:, ct][:, None, :]
            dy = Yb[bi][:, cb][:, :, None] - Yt[ti][:, ct][:, None, :]
            d = np.hypot(dx, dy)
            ds.append(np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2))  # (frames, players_b)
        d = np.concatenate(ds, 1)
        d = np.where(np.isinf(d), np.nan, d)
        d[np.isnan(np.c_[Xb[bi][:, np.isin(team_b, teams)]])] = np.nan
        with np.errstate(all="ignore"):
            out[k:k + 4000] = np.nanmedian(d, 1)
    return out


def find_offset(bxb, byb, bxt, byt, approx, search=12, sl=None):
    nb, nt = len(bxb), len(bxt)
    b_all = np.arange(nb) if sl is None else np.arange(sl.start, min(sl.stop, nb))
    best = None
    for off in range(approx - search, approx + search + 1):
        t = b_all + off
        ok = (t >= 0) & (t < nt)
        d = np.hypot(bxb[b_all[ok]] - bxt[t[ok]], byb[b_all[ok]] - byt[t[ok]])
        e = np.nanmedian(d)
        if best is None or e < best[1]:
            best = (off, e)
    return best


def sync_feeds(Xb, Yb, bxb, byb, team_b, Xt, Yt, bxt, byt, team_t, offsets, fps=25, live_thresh_m=2.0,
               smooth_s=2.0):
    """For each broadcast frame: which offset (segment) it belongs to, the tactical frame, and whether
    it is live play (players match the tactical feed) or a replay / graphics / crowd shot."""
    nb, nt = len(bxb), len(bxt)
    b = np.arange(nb)
    costs = []
    for off in offsets:
        t = np.clip(b + off, 0, nt - 1)
        c = player_match_cost(Xb, Yb, Xt, Yt, team_b, team_t, b, t)
        c[(b + off < 0) | (b + off >= nt)] = np.nan
        costs.append(pd.Series(c).rolling(int(smooth_s * fps), center=True, min_periods=1).median().to_numpy())
    C = np.stack(costs, 1)
    Cf = np.where(np.isnan(C), np.inf, C)
    seg = np.argmin(Cf, 1)
    best = Cf[b, seg]
    # segments are consecutive in time: choose cut points that best fit all frames
    # (a replay that happens to match the other offset can't flip the segment)
    good = (best < live_thresh_m)
    K = len(offsets)
    seg_s = np.zeros(nb, int)
    start = 0
    for k in range(K - 1):
        a = (good & (seg == k)).astype(int)
        c = (good & (seg > k)).astype(int)
        # score(cut) = #frames before cut fitting k + #frames after cut fitting later segments
        sc = np.r_[0, np.cumsum(a)] + (c.sum() - np.r_[0, np.cumsum(c)])
        sc[:start + 1] = -1
        cut = int(np.argmax(sc))
        seg_s[cut:] = k + 1
        start = cut
    off = np.asarray(offsets)[seg_s]
    t = b + off
    cost = C[b, seg_s]
    n_players = np.sum(~np.isnan(Xb), 1)
    live = (cost < live_thresh_m) & (t >= 0) & (t < nt)
    # frames with too few players to judge: inherit from neighbours (short) else not live
    unknown = np.isnan(cost) | (n_players < 3)
    live_s = pd.Series(np.where(unknown, np.nan, live.astype(float))).ffill(limit=int(3 * fps)).bfill(limit=int(3 * fps))
    live = live_s.fillna(0).to_numpy() > 0.5
    return pd.DataFrame({"b_frame_idx": b, "t_frame_idx": np.where(live, t, -1), "segment": seg_s,
                         "live": live, "match_cost_m": np.round(cost, 2)})


def detect_offsets(bxb, byb, bxt, byt, fps=25, ds=5, win_s=30, min_share=0.1):
    """Find the constant time offsets between two feeds of the same match by matching ball paths in
    30-s windows (the broadcast drops half-time, so there is usually one offset per half)."""
    T = np.stack([bxt[::ds], byt[::ds]], 1)
    B = np.stack([bxb[::ds], byb[::ds]], 1)
    W = int(win_s * fps / ds)
    found = []
    for w0 in range(0, len(B) - W, W):
        q = B[w0:w0 + W]
        qi = np.flatnonzero(~np.isnan(q[:, 0]))
        if len(qi) < W * 0.4:
            continue
        best = (np.inf, None)
        offs = np.arange(-w0, len(T) - W - w0)
        for c in range(0, len(offs), 4000):
            o = offs[c:c + 4000] + w0
            tx, ty = T[o[:, None] + qi[None, :], 0], T[o[:, None] + qi[None, :], 1]
            d = np.hypot(tx - q[qi, 0], ty - q[qi, 1])
            cnt = np.sum(~np.isnan(d), 1)
            md = np.where(cnt >= 0.6 * len(qi), np.nanmedian(d, 1), np.inf)
            k = int(np.argmin(md))
            if md[k] < best[0]:
                best = (md[k], (o[k] - w0) * ds)
        if best[0] < 2.0:
            found.append((w0 * ds, best[1]))
    if not found:
        return []
    f = pd.DataFrame(found, columns=["b", "off"])
    f["key"] = (f.off / fps).round()
    groups = f.groupby("key").agg(n=("off", "size"), first=("b", "min"), off=("off", "median"))
    groups = groups[groups.n >= max(3, min_share * len(f))].sort_values("first")
    offsets = []
    for off in groups.off.astype(int):
        offsets.append(find_offset(bxb, byb, bxt, byt, off, search=12)[0])  # frame-exact
    return offsets


def airborne_mask(bx1, by1, bx2, by2, thresh_m=3.0, dilate=5):
    """Two cameras see the ball but disagree on where it touches the ground -> it is in the air."""
    both = ~np.isnan(bx1) & ~np.isnan(bx2)
    dd = np.where(both, np.hypot(bx1 - bx2, by1 - by2), 0)
    return pd.Series(dd > thresh_m).rolling(dilate, center=True, min_periods=1).max().to_numpy() > 0


def plausible_extra_ball(bx, by, ex, ey, fps, max_speed=35.0, slack=3.0):
    """Keep extra ball positions (e.g. from YOLO) only where the tracking has no ball, the point is on
    the pitch, and it is reachable from the tracked ball before and after (ball speed <= 35 m/s)."""
    n = len(bx)
    idx = np.arange(n)
    known = ~np.isnan(bx)
    prv = pd.Series(np.where(known, idx, np.nan)).ffill().to_numpy()
    nxt = pd.Series(np.where(known, idx, np.nan)).bfill().to_numpy()
    cand = ~known & ~np.isnan(ex) & (ex > -2) & (ex < PITCH_L + 2) & (ey > -2) & (ey < PITCH_W + 2)
    ok = cand.copy()
    for ref in (prv, nxt):
        r = np.nan_to_num(ref).astype(int)
        d = np.hypot(ex - bx[r], ey - by[r])
        dt = np.abs(idx - ref) / fps
        ok &= np.isnan(ref) | (d <= max_speed * dt + slack)
    return np.where(ok, ex, bx), np.where(ok, ey, by), ok


def load_extra_ball(path, n, frame0):
    """CSV with columns frame,ball_x_m,ball_y_m (frame numbers as in the tracking file)."""
    e = pd.read_csv(path)
    ex, ey = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    i = e.frame.to_numpy() - frame0
    k = (i >= 0) & (i < n)
    ex[i[k]], ey[i[k]] = e.ball_x_m.to_numpy()[k], e.ball_y_m.to_numpy()[k]
    return ex, ey


def run_match(tac_xml, tac_txt, bro_xml, bro_txt, extra_ball_tac=None, extra_ball_bro=None,
              air_thresh_m=3.0, verbose=True, **kw):
    """Both feeds together: sync them, flag airborne balls from the two views, optionally add extra
    ball detections (YOLO), estimate possession on the tactical feed and give live broadcast frames
    the synced tactical label (replays / close-ups keep the broadcast's own estimate)."""
    t0 = time.time()
    mt, teams, fps, size = load_meta(tac_xml)
    mb, teams_b, fps_b, size_b = load_meta(bro_xml)
    assert teams == teams_b, "the two feeds list the teams differently"
    ft, Xt, Yt, St, bxt, byt = load_tracking(tac_txt, len(mt), size)
    fb, Xb, Yb, Sb, bxb, byb = load_tracking(bro_txt, len(mb), size_b)
    nt, nb = len(ft), len(fb)
    tcT = [np.flatnonzero(mt.team.to_numpy() == t) for t in teams]
    tcB = [np.flatnonzero(mb.team.to_numpy() == t) for t in teams]
    if verbose:
        log(f"loaded both feeds in {time.time() - t0:.1f}s; syncing...")

    offsets = detect_offsets(bxb, byb, bxt, byt, fps)
    if not offsets:
        raise RuntimeError("could not sync the feeds (no matching ball paths)")
    S = sync_feeds(Xb, Yb, bxb, byb, mb.team.to_numpy(), Xt, Yt, bxt, byt, mt.team.to_numpy(), offsets, fps)
    live = S.live.to_numpy()
    bi = np.flatnonzero(live)
    ti = S.t_frame_idx.to_numpy()[bi]
    if verbose:
        log(f"sync: broadcast = tactical - {[o for o in offsets]} frames "
            f"({[round(o / fps, 2) for o in offsets]} s); {live.mean():.0%} of broadcast frames are live play, "
            f"the rest replays/close-ups")

    # broadcast ball on the tactical clock -> airborne mask
    bxb_t = np.full(nt, np.nan, np.float32); byb_t = np.full(nt, np.nan, np.float32)
    bxb_t[ti], byb_t[ti] = bxb[bi], byb[bi]
    air_t = airborne_mask(bxt, byt, bxb_t, byb_t, air_thresh_m)
    air_b = np.zeros(nb, bool); air_b[bi] = air_t[ti]

    extra = {}
    if extra_ball_tac:
        ex, ey = load_extra_ball(extra_ball_tac, nt, ft[0])
        bxt, byt, used = plausible_extra_ball(bxt, byt, ex, ey, fps)
        extra["extra_ball_frames_tactical"] = int(used.sum())
    if extra_ball_bro:
        ex, ey = load_extra_ball(extra_ball_bro, nb, fb[0])
        bxb, byb, used = plausible_extra_ball(bxb, byb, ex, ey, fps)
        extra["extra_ball_frames_broadcast"] = int(used.sum())

    period_t, halves = detect_periods(Xt, tcT, bxt, byt, fps, S=St)
    if len(halves) < 2:
        period_t = None
    elif verbose:
        log("halves (tactical clock): " + ", ".join(f"{a / fps / 60:.1f}-{b / fps / 60:.1f} min" for a, b in halves))
    outT, infoT = estimate_arrays(ft, Xt, Yt, St, bxt, byt, teams, tcT, fps, air_mask=air_t, period=period_t,
                                  verbose=verbose, name="tactical: ", **kw)
    outT["ball_in_air"] = air_t.astype(int)
    labT = outT.team_id.to_numpy()
    confT = outT.confidence.to_numpy()

    # every broadcast frame has a real-time position on the tactical clock (the match keeps going
    # during close-ups and replays); replays additionally show an earlier moment
    seg = S.segment.to_numpy()
    t_real = np.clip(np.arange(nb) + np.asarray(offsets)[seg], 0, nt - 1)
    t_shown, _ = map_replays(Xb, Yb, mb.team.to_numpy(), Xt, Yt, mt.team.to_numpy(), live, t_real, fps)
    replay = (~live) & (t_shown >= 0)
    view = np.where(live, "live", np.where(replay, "replay", "other"))
    t_label = np.where(live, S.t_frame_idx.to_numpy(), np.where(replay, t_shown, t_real))
    t_label = np.clip(t_label, 0, nt - 1)
    labB = labT[t_label]
    if verbose:
        log(f"broadcast views: live {np.mean(view == 'live'):.0%}, replays matched to the moment shown "
            f"{np.mean(view == 'replay'):.1%}, close-ups/other (real-time label) {np.mean(view == 'other'):.0%}")
    outB = pd.DataFrame({
        "frame": fb, "time_s": np.round((fb - fb[0]) / fps, 2),
        "team_id": labB, "team": np.array(teams)[labB],
        "confidence": confT[t_label],
        "view": view,
        "tactical_frame": ft[t_label],
        "team_realtime": np.array(teams)[labT[t_real]],
        "ball_seen": (~np.isnan(bxb)).astype(int), "players_seen": np.sum(~np.isnan(Xb), 1),
        "ball_in_air": air_t[t_real].astype(int),
    })
    if period_t is not None:
        per_real = period_t[t_real]
        shown_in_play = np.isin(period_t[t_label], ["1st_half", "2nd_half"])
        outB["period"] = per_real
        outB["in_play"] = np.isin(per_real, ["1st_half", "2nd_half"]).astype(int)
        # replays show play even at half-time; otherwise follow the real-time match state
        outB["team_or_none"] = np.where(shown_in_play & ((view == "replay") | (outB.in_play == 1)), outB.team, "none")
    in_play_b = outB.in_play.to_numpy() == 1 if "in_play" in outB else np.ones(nb, bool)
    infoB = dict(teams=teams, fps=fps, frames=nb,
                 possession={teams[t]: round(float(np.mean(labB[in_play_b] == t)), 4) for t in (0, 1)},
                 in_play_minutes=round(float(in_play_b.sum() / fps / 60), 1),
                 views={k: round(float(np.mean(view == k)), 3) for k in ("live", "replay", "other")},
                 possession_changes=int(np.sum(np.diff(labB) != 0)))
    sync_info = dict(offsets_frames=[int(o) for o in offsets], broadcast_live_share=round(float(live.mean()), 3),
                     halves_tactical_s=[[round(a / fps, 1), round(b / fps, 1)] for a, b in halves],
                     airborne_frames_tactical=int(air_t.sum()), **extra)
    return outT, infoT, outB, infoB, sync_info


# ----------------------------------------------------------------------------- match periods
def _blocks(mask):
    m = np.asarray(mask, bool); d = np.diff(np.r_[0, m.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def formation_frames(X, team_cols, pitch=(105.0, 68.0), min_each=8, tol=2.0, outliers=1):
    """Each team in its own half (>= 8 visible players each, allowing one stray track per team, e.g.
    a goalkeeper labelled with the wrong team): the line-up before a kick-off."""
    cx = pitch[0] / 2
    x0, x1 = X[:, team_cols[0]], X[:, team_cols[1]]
    n0, n1 = np.sum(~np.isnan(x0), 1), np.sum(~np.isnan(x1), 1)
    with np.errstate(invalid="ignore"):
        w0l = np.sum(x0 > cx + tol, 1); w0r = np.sum(x0 < cx - tol, 1)   # team 0 players NOT in left / right half
        w1l = np.sum(x1 > cx + tol, 1); w1r = np.sum(x1 < cx - tol, 1)
    a = (w0l <= outliers) & (w1r <= outliers)   # team 0 left, team 1 right
    b = (w0r <= outliers) & (w1l <= outliers)
    return (n0 >= min_each) & (n1 >= min_each) & (a | b)


def kickoff_frames(X, team_cols, bx, by, fps, pitch=(105.0, 68.0)):
    """Frames that look like a kick-off: ball on the centre spot, every visible player of one team
    in the left half and every visible player of the other in the right half (>= 8 each)."""
    cx, cy = pitch[0] / 2, pitch[1] / 2
    near = np.hypot(bx - cx, by - cy) < 1.5
    ok = np.zeros(len(bx), bool)
    idx = np.flatnonzero(near)
    if len(idx) == 0:
        return ok
    x0, x1 = X[idx][:, team_cols[0]], X[idx][:, team_cols[1]]
    n0, n1 = np.sum(~np.isnan(x0), 1), np.sum(~np.isnan(x1), 1)
    with np.errstate(invalid="ignore"):
        left0 = np.nanmax(x0, 1) < cx + 1.5; right0 = np.nanmin(x0, 1) > cx - 1.5
        left1 = np.nanmax(x1, 1) < cx + 1.5; right1 = np.nanmin(x1, 1) > cx - 1.5
    form = (n0 >= 8) & (n1 >= 8) & ((left0 & right1) | (right0 & left1))
    ok[idx[form]] = True
    return ok


def detect_periods(X, team_cols, bx, by, fps, S=None, pitch=(105.0, 68.0), min_players=18, merge_gap_s=240,
                   min_half_min=35):
    """Halves = the two longest stretches with a full complement of players (short breaks merged).
    Each half starts at the first kick-off formation in its stretch (if found) and ends when the ball
    was last tracked before the next half / the players leave."""
    n = len(bx)
    cnt = np.sum(~np.isnan(X), 1)
    on = pd.Series(cnt).rolling(int(60 * fps), center=True, min_periods=1).median().to_numpy() >= min_players
    merged = []
    for a, b in _blocks(on):
        if merged and a - merged[-1][1] <= merge_gap_s * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [m for m in merged if m[1] - m[0] >= min_half_min * 60 * fps]
    merged = sorted(sorted(merged, key=lambda m: m[1] - m[0], reverse=True)[:2])
    form = formation_frames(X, team_cols, pitch)
    with np.errstate(all="ignore"):
        speed = np.nanmean(S, 1) if S is not None else None
    halves = []
    for a, b in merged:
        lo, hi = max(0, a - int(240 * fps)), min(n, a + int(600 * fps))
        # first >= 2 s line-up near the start of the stretch; kick-off = when it breaks
        f = form[lo:hi]
        rl = pd.Series(f).groupby((~pd.Series(f)).cumsum()).transform("sum").to_numpy()
        cand = np.flatnonzero(f & (rl >= 5 * fps))   # players standing in their halves for >= 5 s
        if len(cand):
            first = lo + cand[0]
            brk = np.flatnonzero(~form[first:min(n, first + int(300 * fps))])
            ko = first + (brk[0] if len(brk) else 0)
            if speed is not None:  # players walk into place, stand still, then everyone accelerates
                w = pd.Series(speed[first:min(n, first + int(180 * fps))]).rolling(int(fps), min_periods=1).mean().to_numpy()
                still = np.flatnonzero(w < 0.8)
                if len(still):
                    mv = np.flatnonzero(w[still[0]:] > 1.5)
                    if len(mv):
                        ko = first + still[0] + mv[0]
        else:
            ko = a
        seen = np.flatnonzero(~np.isnan(bx[ko:b]))
        end = ko + (seen[-1] + 1 if len(seen) else b - ko)
        halves.append((int(ko), int(end)))
    return periods_from_halves(halves, n), halves


def periods_from_halves(halves, n, frame_index=None):
    """Period label per frame. frame_index (optional) = frame positions on the halves' clock."""
    t = np.arange(n) if frame_index is None else np.asarray(frame_index)
    period = np.full(n, "pre_match", dtype=object)
    if halves:
        a1, b1 = halves[0]
        period[(t >= a1) & (t < b1)] = "1st_half"
        if len(halves) > 1:
            a2, b2 = halves[1]
            period[(t >= b1) & (t < a2)] = "half_time"
            period[(t >= a2) & (t < b2)] = "2nd_half"
            period[t >= b2] = "post_match"
        else:
            period[t >= b1] = "post_match"
    return period


# ----------------------------------------------------------------------------- replays
def _cost_many(xb, yb, tb, Xt, Yt, team_t, cand):
    """Median nearest-same-team distance between one broadcast frame's players (xb, yb, tb)
    and many candidate tactical frames."""
    ds = []
    for tm in np.unique(tb):
        pb = np.flatnonzero(tb == tm)
        ct = np.flatnonzero(team_t == tm)
        dx = xb[pb][None, :, None] - Xt[cand][:, None, ct]
        dy = yb[pb][None, :, None] - Yt[cand][:, None, ct]
        d = np.hypot(dx, dy)
        ds.append(np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2))
    d = np.concatenate(ds, 1)
    d = np.where(np.isinf(d), np.nan, d)
    with np.errstate(all="ignore"):
        return np.nanmedian(d, 1)


def map_replays(Xb, Yb, team_b, Xt, Yt, team_t, live, t_realtime, fps=25, lookback_s=600,
                sample_s=0.4, min_players=6, max_cost_m=3.0, rel=0.35):
    """For broadcast frames that are not live, find the earlier tactical frame they show (replays).
    Returns tactical frame index per broadcast frame (-1 = no match, e.g. close-up / crowd / graphics)."""
    nb, nt = Xb.shape[0], Xt.shape[0]
    out = np.full(nb, -1)
    cost = np.full(nb, np.nan)
    nplay = np.sum(~np.isnan(Xb), 1)
    step = max(1, int(sample_s * fps))
    nl = ~live
    d = np.diff(np.r_[0, nl.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a < fps:  # < 1 s: not worth it
            continue
        pts = []
        for i in range(a, b, step):
            if nplay[i] < min_players:
                continue
            v = ~np.isnan(Xb[i])
            hi = int(min(max(t_realtime[i], 0), nt - 1))
            lo = max(0, hi - int(lookback_s * fps))
            coarse = np.arange(lo, hi, 5)
            if len(coarse) == 0:
                continue
            c = _cost_many(Xb[i, v], Yb[i, v], team_b[v], Xt, Yt, team_t, coarse)
            if np.all(np.isnan(c)):
                continue
            k = int(np.nanargmin(c))
            if c[k] > rel * np.nanmedian(c):  # best match must stand out from a typical frame
                continue
            fine = np.arange(max(0, coarse[k] - 5), min(nt, coarse[k] + 6))
            cf = _cost_many(Xb[i, v], Yb[i, v], team_b[v], Xt, Yt, team_t, fine)
            j = int(np.nanargmin(cf))
            if cf[j] <= max_cost_m:
                pts.append((i, int(fine[j]), float(cf[j])))
        if len(pts) < 2:
            continue
        p = np.array(pts)
        # keep matches consistent with one forward-moving replay (slow motion allowed: 0.15x - 1.3x)
        keep = [0]
        for k in range(1, len(p)):
            di, dt = p[k, 0] - p[keep[-1], 0], p[k, 1] - p[keep[-1], 1]
            if 0.15 * di - 10 <= dt <= 1.3 * di + 10:
                keep.append(k)
            else:  # a new replay angle starts: keep it as its own piece
                keep.append(k)
        p = p[keep]
        # interpolate between consecutive matches that belong to the same piece
        for k in range(len(p) - 1):
            i0, t0, i1, t1 = int(p[k, 0]), p[k, 1], int(p[k + 1, 0]), p[k + 1, 1]
            di, dt = i1 - i0, t1 - t0
            if di <= 3 * step and 0.15 * di - 10 <= dt <= 1.3 * di + 10:
                out[i0:i1 + 1] = np.round(np.linspace(t0, t1, i1 - i0 + 1)).astype(int)
                cost[i0:i1 + 1] = max(p[k, 2], p[k + 1, 2])
            else:
                out[i0] = int(t0); cost[i0] = p[k, 2]
        out[int(p[-1, 0])] = int(p[-1, 1]); cost[int(p[-1, 0])] = p[-1, 2]
    return out, cost


# ----------------------------------------------------------------------------- tags
TAG_MEANING = {  # +1: tagged team has the ball, -1: the other team has it
    "BUILD UP": 1, "PROGRESSION": 1, "ATTACKING TRANSITION": 1, "CREATING CHANCES": 1,
    "LONG BALL": 1, "BALL IN FINAL THIRD": 1, "BALL IN THE BOX": 1, "PLAYERS IN THE BOX": 1,
    "DEFENDING IN DEFENSIVE THIRD": -1, "DEFENDING IN MIDDLE THIRD": -1,
    "DEFENDING IN ATTACKING THIRD": -1, "DEFENSIVE TRANSITION": -1,
}


def load_tags(path):
    r = ET.parse(path).getroot()
    rows = []
    for i in r.iter("instance"):
        lab = {l.findtext("group"): l.findtext("text") for l in i.findall("label")}
        rows.append(dict(code=i.findtext("code"), start=float(i.findtext("start")),
                         end=float(i.findtext("end")), team=lab.get("Team"), type=lab.get("Type"),
                         half=lab.get("Half")))
    return pd.DataFrame(rows)


def tag_reference(tags, teams, n, fps):
    """Frame-level reference built from analyst phases (-1 where unknown or conflicting)."""
    votes = np.zeros((n, 2), np.int32)
    for _, t in tags.iterrows():
        if t.code not in TAG_MEANING or t.team not in teams:
            continue
        team = teams.index(t.team) if TAG_MEANING[t.code] > 0 else 1 - teams.index(t.team)
        a, b = int(round(t.start * fps)), int(round(t.end * fps))
        votes[max(a, 0):min(b, n), team] += 1
    ref = np.full(n, -1)
    ref[(votes[:, 0] > 0) & (votes[:, 1] == 0)] = 0
    ref[(votes[:, 1] > 0) & (votes[:, 0] == 0)] = 1
    return ref


def score_vs_tags(lab, tags, teams, fps):
    rows = []
    for code, g in tags.groupby("code"):
        if code not in TAG_MEANING:
            continue
        agree = tot = 0
        for _, t in g.iterrows():
            if t.team not in teams:
                continue
            exp = teams.index(t.team) if TAG_MEANING[code] > 0 else 1 - teams.index(t.team)
            a, b = int(round(t.start * fps)), int(round(t.end * fps))
            seg = lab[max(a, 0):min(b, len(lab))]
            seg = seg[seg >= 0]
            agree += (seg == exp).sum()
            tot += len(seg)
        if tot:
            rows.append((code, tot / fps / 60, agree / tot))
    df = pd.DataFrame(rows, columns=["tag", "minutes", "agreement"]).sort_values("minutes", ascending=False)
    overall = np.average(df.agreement, weights=df.minutes) if len(df) else np.nan
    return df, overall


# ----------------------------------------------------------------------------- pipeline
GAP_CAP_S = 120.0


def attack_signs_centroid(X, team_cols, fps, win_min=10):
    """Direction of play per frame from the two teams' average depth over a 10-min window.
    (Goalkeeper tracks in this data jump between goals, so they are not used.)"""
    c0 = np.nanmean(X[:, team_cols[0]], 1)
    c1 = np.nanmean(X[:, team_cols[1]], 1)
    diff = pd.Series(c0 - c1).rolling(int(win_min * 60 * fps), center=True,
                                      min_periods=int(30 * fps)).median().ffill().bfill().fillna(-1).to_numpy()
    s0 = np.where(diff < 0, 1, -1)  # team 0 deeper on the left -> attacks to the right
    return np.stack([s0, -s0])


def gap_features(shape, before, t_before, after, t_after):
    f = shape.copy()
    f["before"], f["after"] = before, after
    f["t_before"], f["t_after"] = np.minimum(t_before, GAP_CAP_S), np.minimum(t_after, GAP_CAP_S)
    f["frac"] = f.t_before / (f.t_before + f.t_after + 1e-6)
    return f


def train_gap_model(shape, lab, known, ok, players_ok, fps, n_samples=250000, seed=0):
    """Learn P(team 1 has the ball | players' shape, who had it before/after the gap and how
    long ago/ahead) by hiding the ball around frames where possession is known."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(lab)
    rl = run_lengths(ok)
    real = rl[(~ok) & (rl > 3 * fps)]
    L = real if len(real) else np.array([10 * fps])
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(known & players_ok)
    t = rng.choice(cand, n_samples)
    Ls = rng.choice(L, n_samples) + 2 * fps
    u = rng.random(n_samples)
    a, b = (u * Ls).astype(int) + 1, ((1 - u) * Ls).astype(int) + 1
    v = (t - a >= 0) & (t + b < n)
    t, a, b = t[v], a[v], b[v]
    before = np.where(known[t - a], lab[t - a], -1)
    after = np.where(known[t + b], lab[t + b], -1)
    F = gap_features(shape.iloc[t].reset_index(drop=True), before, a / fps, after, b / fps)
    y = lab[t]
    hold = (t // int(60 * fps)) % 5 == 0  # every 5th minute held out
    mk = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                                                l2_regularization=1.0, random_state=seed)
    clf = mk().fit(F[~hold], y[~hold])
    acc = float(np.mean(clf.predict(F[hold]) == y[hold]))
    carry = float(np.mean(np.where(before[hold] >= 0, before[hold], 0) == y[hold]))
    clf = mk().fit(F, y)
    return clf, acc, carry


def estimate_arrays(frame, X, Y, S, bx, by, teams, team_cols, fps, X_ctrl=None, Y_ctrl=None, ctrl_team_cols=None,
                    radius=1.5, margin=0.4, min_switch_s=0.8, switch_gap_s=0.4,
                    max_ball_gap_s=1.0, control_memory_s=2.0, use_gap_model=True, gap_model_on="all",
                    smooth_s=10.0, air_mask=None, period=None, verbose=True, name=""):
    """Core: arrays in metres. X_ctrl/Y_ctrl (optional) = extra player positions used only to decide
    who is touching the ball (e.g. players filled in from the other camera feed).
    air_mask (optional, bool per frame) = ball known to be in the air (e.g. two cameras disagree on
    its ground position), so nobody controls it in that frame.
    period (optional, per frame) = pre_match / 1st_half / half_time / 2nd_half / post_match; the gap
    model then learns only from match play, and a team_or_none column is added."""
    n = len(frame)
    # 1. ball control
    ball_seen = ~np.isnan(bx)
    gap = max(1, int(round(max_ball_gap_s * fps)))
    bxi, byi = interp_short_gaps(bx, gap), interp_short_gaps(by, gap)
    ok = ~np.isnan(bxi)
    if X_ctrl is None:
        X_ctrl, Y_ctrl, ctrl_team_cols = X, Y, team_cols
    raw, dbest, _ = ball_control(X_ctrl, Y_ctrl, bxi, byi, ctrl_team_cols, radius, margin)
    if air_mask is not None:
        raw = np.where(air_mask, -1, raw)

    # 2. hysteresis
    lab = hysteresis(raw, max(1, int(round(min_switch_s * fps))), max(1, int(round(switch_gap_s * fps))))
    last_ctrl = pd.Series(np.where(raw >= 0, np.arange(n), np.nan)).ffill().to_numpy()
    since_ctrl = np.arange(n) - np.nan_to_num(last_ctrl, nan=-1e9)
    known = ok & (since_ctrl <= control_memory_s * fps) & (lab >= 0)
    source = np.where(known, "ball", "carried").astype(object)
    p1 = np.where(known, (lab == 1).astype(float), np.nan)

    # 3. frames without a trustworthy ball: gap model
    players_ok = np.sum(~np.isnan(X), 1) >= 10
    in_play = np.ones(n, bool) if period is None else np.isin(period, ["1st_half", "2nd_half"])
    info_extra = {}
    if use_gap_model and known.sum() > 5000:
        signs = attack_signs_centroid(X, team_cols, fps)
        shape = shape_features(X, Y, S, bxi, byi, team_cols, signs)
        clf, acc, carry = train_gap_model(shape, lab, known & in_play, ok, players_ok, fps)
        target = ~ok if gap_model_on == "missing" else ~known
        idx = np.flatnonzero(target)
        prv = pd.Series(np.where(known, np.arange(n), np.nan)).ffill().to_numpy()[idx]
        nxt = pd.Series(np.where(known, np.arange(n), np.nan)).bfill().to_numpy()[idx]
        before = np.where(np.isnan(prv), -1, lab[np.nan_to_num(prv).astype(int)])
        after = np.where(np.isnan(nxt), -1, lab[np.nan_to_num(nxt).astype(int)])
        tb = np.where(np.isnan(prv), GAP_CAP_S, (idx - np.nan_to_num(prv)) / fps)
        ta = np.where(np.isnan(nxt), GAP_CAP_S, (np.nan_to_num(nxt) - idx) / fps)
        p = clf.predict_proba(gap_features(shape.iloc[idx].reset_index(drop=True), before, tb, after, ta))[:, 1]
        if smooth_s > 0:  # smooth within each gap so the label doesn't flicker
            full = pd.Series(np.nan, index=np.arange(n)); full.iloc[idx] = p
            gid = pd.Series(np.cumsum(~target), index=np.arange(n))
            w = max(1, int(round(smooth_s * fps)))
            p = full.groupby(gid).transform(lambda v: v.rolling(w, center=True, min_periods=1).mean()).to_numpy()[idx]
        lab[idx] = (p > 0.5).astype(int)
        p1[idx] = p
        source[idx] = "gap_model"
        info_extra = dict(gap_model_heldout_acc=round(acc, 4), carry_forward_heldout_acc=round(carry, 4))
        if verbose:
            log(f"{name}gap model on held-out simulated gaps: {acc:.1%} (carrying possession forward: {carry:.1%}); "
                f"used for {np.mean(target):.1%} of frames")

    if (lab < 0).any():
        s = pd.Series(np.where(lab >= 0, lab, np.nan))
        lab = s.ffill().bfill().fillna(0).astype(int).to_numpy()

    out = pd.DataFrame({
        "frame": frame, "time_s": np.round((frame - frame[0]) / fps, 2),
        "team_id": lab, "team": np.array(teams)[lab],
        "confidence": np.round(np.where(lab == 1, p1, 1 - p1), 3),
        "source": source, "raw_control": raw,
        "ball_dist_m": np.round(np.where(np.isfinite(dbest), dbest, np.nan), 2),
        "ball_seen": ball_seen.astype(int), "players_seen": np.sum(~np.isnan(X), 1),
    })
    if period is not None:
        out["period"] = period
        out["in_play"] = in_play.astype(int)
        out["team_or_none"] = np.where(in_play, out.team, "none")
    info = dict(teams=teams, fps=fps, frames=n,
                possession={teams[t]: round(float(np.mean(lab[in_play] == t)), 4) for t in (0, 1)},
                in_play_minutes=round(float(in_play.sum() / fps / 60), 1),
                ball_seen=round(float(ball_seen.mean()), 3),
                source_share=out.source.value_counts(normalize=True).round(3).to_dict(),
                possession_changes=int(np.sum(np.diff(lab) != 0)), **info_extra)
    return out, info


def estimate(xml, txt, verbose=True, **kw):
    t0 = time.time()
    meta, teams, fps, size = load_meta(xml)
    frame, X, Y, S, bx, by = load_tracking(txt, len(meta), size)
    team_cols = [np.flatnonzero(meta.team.to_numpy() == teams[t]) for t in (0, 1)]
    if verbose:
        log(f"{txt.split('/')[-1]}: {len(frame)} frames ({len(frame) / fps / 3600:.2f} h) loaded in "
            f"{time.time() - t0:.1f}s; ball present in {np.mean(~np.isnan(bx)):.0%} of frames")
    period, halves = detect_periods(X, team_cols, bx, by, fps, S=S)
    if len(halves) < 2:  # can't see the whole pitch (e.g. broadcast only): label every frame as play
        period = None
    elif verbose:
        log("halves: " + ", ".join(f"{a / fps / 60:.1f}-{b / fps / 60:.1f} min" for a, b in halves))
    return estimate_arrays(frame, X, Y, S, bx, by, teams, team_cols, fps, period=period, verbose=verbose, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True, help="tracking metadata (tactical feed if you have both)")
    ap.add_argument("--txt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--other_xml", help="second feed of the same match (broadcast) -> sync both feeds")
    ap.add_argument("--other_txt")
    ap.add_argument("--other_out")
    ap.add_argument("--extra_ball", help="extra ball detections for the first feed (yolo_ball_fill.py output)")
    ap.add_argument("--other_extra_ball", help="extra ball detections for the second feed")
    ap.add_argument("--tags", help="smart tagging xml on the first feed's clock, to measure accuracy")
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--margin", type=float, default=0.4)
    ap.add_argument("--min_switch_s", type=float, default=0.8,
                    help="control needed before possession changes (deflections shorter than this don't count)")
    ap.add_argument("--switch_gap_s", type=float, default=0.4)
    ap.add_argument("--max_ball_gap_s", type=float, default=1.0)
    ap.add_argument("--no_gap_model", action="store_true", help="just carry possession through ball gaps")
    ap.add_argument("--smooth_s", type=float, default=10.0, help="smoothing of the gap model inside each gap")
    a = ap.parse_args()
    kw = dict(radius=a.radius, margin=a.margin, min_switch_s=a.min_switch_s, switch_gap_s=a.switch_gap_s,
              max_ball_gap_s=a.max_ball_gap_s, use_gap_model=not a.no_gap_model, smooth_s=a.smooth_s)
    if a.other_xml:
        out, info, out2, info2, sync_info = run_match(a.xml, a.txt, a.other_xml, a.other_txt,
                                                      extra_ball_tac=a.extra_ball,
                                                      extra_ball_bro=a.other_extra_ball, **kw)
        out.to_csv(a.out, index=False)
        out2_path = a.other_out or a.other_txt.rsplit(".", 1)[0] + "_possession.csv"
        out2.to_csv(out2_path, index=False)
        log(f"wrote {a.out} and {out2_path}")
        log("sync:", json.dumps(sync_info))
        log("first feed:", json.dumps(info))
        log("second feed:", json.dumps(info2))
    else:
        if a.extra_ball:
            meta, teams, fps, size = load_meta(a.xml)
            frame, X, Y, S, bx, by = load_tracking(a.txt, len(meta), size)
            ex, ey = load_extra_ball(a.extra_ball, len(frame), frame[0])
            bx, by, used = plausible_extra_ball(bx, by, ex, ey, fps)
            log(f"extra ball positions used: {int(used.sum())}")
            team_cols = [np.flatnonzero(meta.team.to_numpy() == teams[t]) for t in (0, 1)]
            out, info = estimate_arrays(frame, X, Y, S, bx, by, teams, team_cols, fps, **kw)
        else:
            out, info = estimate(a.xml, a.txt, **kw)
        out.to_csv(a.out, index=False)
        log(f"wrote {a.out}")
        log(json.dumps(info))
    if a.tags:
        tags = load_tags(a.tags)
        df, overall = score_vs_tags(out.team_id.to_numpy(), tags, info["teams"], info["fps"])
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        log(f"agreement with analyst phases (time-weighted): {overall:.1%}")


if __name__ == "__main__":
    main()
