"""Frame-level H / A / D / X for the tactical video (More Than A Hack 2027, Phase 1)."""
import sys, os, time, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_tracking as PT
import dead_ball as DB

PITCH = (105.3, 68.0)          # games.py value, measured from the corrected tracking
HALVES = [(11185, 90651), (110808, 190197)]   # graded video frames, published in DATA.md


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load_feed(xml, txt):
    meta, teams, fps, _ = PT.load_meta(xml)
    fr, X, Y, S, bx, by = PT.load_tracking(txt, len(meta), PITCH)
    cols = [np.flatnonzero(meta.team.to_numpy() == t) for t in teams]
    return dict(meta=meta, teams=teams, fps=fps, frame=fr, X=X, Y=Y, S=S, bx=bx, by=by, cols=cols)


def possession_in_alive(raw, alive, K, G):
    """Hysteresis possession, restarted inside every live stretch: the first team to control the ball
    after a restart owns it from the restart (back-dated), then changes need K frames of control."""
    lab = np.full(len(raw), -1, int)
    for a, b in DB.runs_of(alive):
        seg = PT.hysteresis(raw[a:b], K, G)
        first = np.flatnonzero(raw[a:b] >= 0)
        if len(first):
            seg[seg < 0] = raw[a + first[0]]
        lab[a:b] = seg
    return lab


def run(tac_xml, tac_txt, bro_xml=None, bro_txt=None, tags_xml=None, radius=1.5, margin=0.4,
        min_switch_s=0.8, switch_gap_s=0.4, max_ball_gap_s=1.0, verbose=True):
    t0 = time.time()
    T = load_feed(tac_xml, tac_txt)
    fps, n = T["fps"], len(T["frame"])
    graded = np.zeros(n, bool)
    for a, b in HALVES:
        graded[a:b + 1] = True
    X, Y, S, bx, by, cols = T["X"], T["Y"], T["S"], T["bx"], T["by"], T["cols"]
    if verbose:
        log(f"tactical loaded ({n} frames, {time.time() - t0:.0f}s)")

    # 1. airborne ball from the second camera (optional)
    air = np.zeros(n, bool)
    if bro_xml:
        B = load_feed(bro_xml, bro_txt)
        offsets = PT.detect_offsets(B["bx"], B["by"], bx, by, fps)
        Sy = PT.sync_feeds(B["X"], B["Y"], B["bx"], B["by"], B["meta"].team.to_numpy(), X, Y, bx, by,
                           T["meta"].team.to_numpy(), offsets, fps)
        live = Sy.live.to_numpy()
        bi = np.flatnonzero(live)
        ti = Sy.t_frame_idx.to_numpy()[bi]
        bxb = np.full(n, np.nan, np.float32); byb = np.full(n, np.nan, np.float32)
        bxb[ti], byb[ti] = B["bx"][bi], B["by"][bi]
        air = PT.airborne_mask(bx, by, bxb, byb)
        if verbose:
            log(f"broadcast synced: offsets {offsets} frames, airborne frames {air.sum()}")

    # 2. dead ball
    tags = PT.load_tags(tags_xml) if tags_xml else None
    sp_starts = tags[tags.code == "SET PIECES"].start.to_numpy() if tags is not None else []
    form = PT.formation_frames(X, cols, pitch=PITCH)
    F = DB.kinematic_features(X, Y, S, bx, by, fps)
    F = F.drop(columns=[c for c in F.columns if c.split("_")[0] in ("bvis", "bsp", "near", "crowd")])
    plab = DB.pseudo_labels(X, Y, bx, by, cols, graded, sp_starts, fps, form)
    clf, auc = DB.fit_dead_model(F, plab, fps)
    p_dead = clf.predict_proba(F)[:, 1]
    dead = DB.segment_dead(p_dead, bx, by, graded, fps)
    alive = graded & ~dead
    if verbose:
        r = DB.runs_of(dead)
        log(f"dead ball: model AUC on held-out pseudo-labels {auc:.3f}; {dead[graded].mean():.1%} of graded "
            f"frames dead, {len(r)} stoppages")

    # 3. ball control on live frames
    gap = int(round(max_ball_gap_s * fps))
    bxi, byi = PT.interp_short_gaps(bx, gap), PT.interp_short_gaps(by, gap)
    raw, dbest, _ = PT.ball_control(X, Y, bxi, byi, cols, radius, margin)
    raw = np.where(air | ~alive, -1, raw)
    K, G = max(1, int(round(min_switch_s * fps))), max(1, int(round(switch_gap_s * fps)))
    lab = possession_in_alive(raw, alive, K, G)

    # 4. live frames with no recent ball control: gap model (players' shape + context)
    ok = ~np.isnan(bxi)
    last = pd.Series(np.where(raw >= 0, np.arange(n), np.nan)).ffill().to_numpy()
    known = alive & ok & ((np.arange(n) - np.nan_to_num(last, nan=-1e9)) <= 2 * fps) & (lab >= 0)
    players_ok = np.sum(~np.isnan(X), 1) >= 10
    signs = PT.attack_signs_centroid(X, cols, fps)
    shape = PT.shape_features(X, Y, S, bxi, byi, cols, signs)
    gclf, gacc, gcarry = PT.train_gap_model(shape, lab, known, ok, players_ok, fps)
    target = alive & ~known
    idx = np.flatnonzero(target)
    prv = pd.Series(np.where(known, np.arange(n), np.nan)).ffill().to_numpy()[idx]
    nxt = pd.Series(np.where(known, np.arange(n), np.nan)).bfill().to_numpy()[idx]
    before = np.where(np.isnan(prv), -1, lab[np.nan_to_num(prv).astype(int)])
    after = np.where(np.isnan(nxt), -1, lab[np.nan_to_num(nxt).astype(int)])
    tb = np.where(np.isnan(prv), PT.GAP_CAP_S, (idx - np.nan_to_num(prv)) / fps)
    ta = np.where(np.isnan(nxt), PT.GAP_CAP_S, (np.nan_to_num(nxt) - idx) / fps)
    pg = gclf.predict_proba(PT.gap_features(shape.iloc[idx].reset_index(drop=True), before, tb, after, ta))[:, 1]
    full = pd.Series(np.nan, index=np.arange(n)); full.iloc[idx] = pg
    gid = pd.Series(np.cumsum(~target), index=np.arange(n))
    pg = full.groupby(gid).transform(lambda v: v.rolling(int(10 * fps), center=True, min_periods=1).mean()).to_numpy()[idx]
    lab[idx] = (pg > 0.5).astype(int)
    if verbose:
        log(f"gap model on held-out simulated gaps {gacc:.1%} (carry-forward {gcarry:.1%}); used on "
            f"{target.sum() / alive.sum():.1%} of live frames")

    # 5. labels
    out = np.full(n, "X", dtype="<U1")
    out[alive] = np.where(lab[alive] == 0, "H", "A")
    out[alive & (lab < 0)] = "X"
    out[dead] = "D"
    info = dict(graded=int(graded.sum()), dead_share=round(float(dead[graded].mean()), 4),
                stoppages=len(DB.runs_of(dead)), dead_auc=round(auc, 4), gap_acc=round(gacc, 4),
                seconds=round(time.time() - t0, 1))
    return out, info


def write(path, labels):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_broadcast", action="store_true")
    a = ap.parse_args()
    d = a.data
    lab, info = run(f"{d}/tactical/BAR-ATH_tactical_atd.xml", f"{d}/tactical/BAR-ATH_tactical_atd.txt",
                    None if a.no_broadcast else f"{d}/broadcast/BAR-ATH_broadcast_atd.xml",
                    None if a.no_broadcast else f"{d}/broadcast/BAR-ATH_broadcast_atd.txt",
                    f"{d}/tactical/BAR-ATH_tactical_smart_tagging.xml")
    write(a.out, lab)
    log(json.dumps(info))
