#!/usr/bin/env python3
"""
hmm_detect.py

Loads an HMM trained by hmm_train.py (params.json), encodes a session with the
same observation template, runs Viterbi, and flags anomalies step-by-step.

Anomaly signals:
- Emission surprise:  -log( B[z_t, o_t] )
- Transition surprise: -log( A[z_{t-1}, z_t] )  (t > 0)
- Unknown token: observation not in training vocab

Defaults (tunable):
- --emit-thresh  : flag when emission_p < 1e-3  (≈ surprise > 6.9)
- --trans-thresh : flag when trans_p    < 1e-3
- --avg-thresh   : flag session if mean per-step total surprise is high

Usage:
  python hmm_detect.py --params params.json session_new.json
"""

import json, math, argparse, re
from pathlib import Path

# ---- Keep these helpers consistent with hmm_train.py ----

def map_kind(t: str | None) -> str:
    t = (t or "").lower()
    if t in ("start", "navigation"): return "screen"
    if t == "menu": return "menu"
    if t == "action": return "action"
    return "unknown"

def bucket(s: str | None, maxlen=24) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 _\\-\\.:/\\\\]", "", s)
    return s[:maxlen] if s else "∅"

def edge_to_observation(edge: dict, nodes_by_id: dict) -> str:
    src = nodes_by_id.get(edge.get("from"), {})
    dst = nodes_by_id.get(edge.get("to"), {})
    src_kind = map_kind(src.get("type"))
    dst_kind = map_kind(dst.get("type"))
    action_b = bucket(edge.get("action"))
    window_b = bucket(dst.get("window") or dst.get("label"))
    return f"{src_kind}->{dst_kind}|{window_b}|{action_b}"

def load_session(path: str):
    data = json.load(open(path, "r", encoding="utf-8"))
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    obs_tokens = [edge_to_observation(e, nodes) for e in data.get("edges", [])]
    return data, obs_tokens

def load_params(params_path: Path):
    obj = json.loads(params_path.read_text(encoding="utf-8"))
    return obj["pi"], obj["A"], obj["B"], obj["vocab"]

def viterbi(pi, A, B, obs_idx):
    """Return (path, logprob) using log-space Viterbi."""
    if not obs_idx: return [], 0.0
    K = len(pi); T = len(obs_idx)
    log = lambda x: math.log(max(x, 1e-12))
    dp = [[-1e18]*K for _ in range(T)]
    bp = [[-1]*K for _ in range(T)]

    # init
    o0 = obs_idx[0]
    for i in range(K):
        dp[0][i] = log(pi[i]) + log(B[i][o0])
    # rec
    for t in range(1, T):
        ot = obs_idx[t]
        for j in range(K):
            best, arg = -1e18, -1
            bj = log(B[j][ot])
            for i in range(K):
                val = dp[t-1][i] + log(A[i][j]) + bj
                if val > best: best, arg = val, i
            dp[t][j] = best; bp[t][j] = arg
    # backtrack
    last = max(range(K), key=lambda i: dp[T-1][i])
    path = [last]
    for t in range(T-1, 0, -1):
        path.append(bp[t][path[-1]])
    path.reverse()
    return path, dp[T-1][last]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True, help="params.json from hmm_train.py")
    ap.add_argument("--emit-thresh", type=float, default=1e-3, help="Emission prob threshold")
    ap.add_argument("--trans-thresh", type=float, default=1e-3, help="Transition prob threshold")
    ap.add_argument("--avg-thresh", type=float, default=7.5, help="Flag session if mean per-step surprise > this")
    ap.add_argument("session", help="Session JSON (flat nodes/edges schema)")
    args = ap.parse_args()

    pi, A, B, vocab = load_params(args.params)
    vocab_index = vocab
    inv_vocab = {i: o for o, i in vocab_index.items()}

    data, obs_tokens = load_session(args.session)

    # map observations to indices (unknowns flagged)
    obs_idx = []
    unknown_positions = []
    for t, tok in enumerate(obs_tokens):
        if tok in vocab_index:
            obs_idx.append(vocab_index[tok])
        else:
            # skip unknowns but record anomaly
            unknown_positions.append((t, tok))

    # if all unknown, just report and exit
    if not obs_idx:
        print(json.dumps({
            "file": args.session,
            "error": "All observations are unknown to the model (retrain with more data or coarser buckets).",
            "unknown_observations": unknown_positions
        }, indent=2))
        return

    path, logp = viterbi(pi, A, B, obs_idx)

    # Per-step scoring
    K = len(pi)
    def neglog(x): return -math.log(max(x, 1e-12))
    anomalies = []
    total_surprise = 0.0
    for local_t, zt in enumerate(path):
        t = local_t  # index within known-token subsequence
        o = obs_idx[t]
        emit_p = B[zt][o]
        emit_sur = neglog(emit_p)
        trans_sur = None
        if t > 0:
            zprev = path[t-1]
            tp = A[zprev][zt]
            trans_sur = neglog(tp)
        step_sur = emit_sur + (trans_sur or 0.0)
        total_surprise += step_sur

        reasons = []
        flagged = False
        if emit_p < args.emit_thresh:
            flagged = True; reasons.append(f"Low emission prob: {emit_p:.2e}")
        if trans_sur is not None:
            # Check transition probability threshold
            trans_p = math.exp(-trans_sur)
            if trans_p < args.trans_thresh:
                flagged = True; reasons.append(f"Low transition prob: {trans_p:.2e}")

        anomalies.append({
            "t": local_t,
            "observation": inv_vocab[o],
            "hidden_state": f"z{zt}",
            "emission_p": round(emit_p, 6),
            "transition_p": (None if trans_sur is None else round(math.exp(-trans_sur), 6)),
            "surprise_total": round(step_sur, 3),
            "flagged": flagged,
            "reasons": reasons
        })

    mean_surprise = total_surprise / len(path)
    session_flag = mean_surprise > args.avg_thresh

    # stitch in unknowns as separate anomalies
    for (t, tok) in unknown_positions:
        anomalies.append({
            "t": t,
            "observation": tok,
            "hidden_state": None,
            "emission_p": 0.0,
            "transition_p": None,
            "surprise_total": None,
            "flagged": True,
            "reasons": ["Unknown observation (not in training vocab)"]
        })

    out = {
        "file": args.session,
        "mean_surprise": round(mean_surprise, 3),
        "session_flagged": session_flag,
        "thresholds": {
            "emission_prob": args.emit_thresh,
            "transition_prob": args.trans_thresh,
            "mean_surprise": args.avg_thresh
        },
        "anomalies": anomalies
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
