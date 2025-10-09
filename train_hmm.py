#!/usr/bin/env python3
"""
hmm_train.py

Train a Hidden Markov Model (HMM) on UI telemetry (your flat nodes/edges JSON).
- Unsupervised Baum–Welch (EM) to learn π (initial), A (transition), B (emission)
- Builds observation tokens from each edge
- Handles one or many JSON files (each is one session / sequence)
- Uses scaling in forward/backward for numerical stability
- Prints learned matrices and the top emissions per hidden state
- Can Viterbi-decode a given session using the learned model

Usage:
  # Train on one or more sessions, specify number of hidden states
  python hmm_train.py --k 5 session1.json session2.json ...

  # Also decode a particular session (prints hidden state path)
  python hmm_train.py --k 5 --decode session1.json session2.json ...

  # Save learned parameters
  python hmm_train.py --k 5 --save params.json session*.json

  # Load and decode with existing params
  python hmm_train.py --load params.json --decode session_new.json
"""

import json, sys, math, argparse, re, random
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------
# Observation encoding
# ---------------------------

# def map_kind(t: str | None) -> str:
#     t = (t or "").lower()
#     if t in ("start", "navigation"): return "screen"
#     if t == "menu": return "menu"
#     if t == "action": return "action"
#     return "unknown"

def map_kind(node: dict | None) -> str:
    """
    Pass the WHOLE node dict here (not node['type']).
    For menu/action, use the node's label; for start/navigation map to 'screen'.
    """
    if not isinstance(node, dict):
        return "unknown"
    node_type = (node.get("type") or "").lower()
    label = normalize_label(node.get("label") or "")

    if node_type in ("menu", "action"):
        # prefer the semantic label (e.g., 'file menu', 'bold button')
        return label or node_type
    if node_type in ("start", "navigation"):
        return "screen"
    return "unknown"

def normalize_label(label: str | None) -> str:
    if not label:
        return "unknown"
    label = label.strip().lower()
    label = re.sub(r"\s+", " ", label)
    return label


def bucket(s: str | None, maxlen=24) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 _\-\.:/\\]", "", s)
    return s[:maxlen] if s else "∅"

def bucket_window(w: str | None) -> str:
    # keep stable part of titles: drop paths/suffixes
    s = (w or "").lower()
    s = re.sub(r"[a-z]:[/\\].*", "", s)   # drop "c:\...\..." tail
    s = re.sub(r".*[/\\]", "", s)         # keep basename if pathy
    s = s.split(" - ")[0]                 # keep left of " - "
    return bucket(s, maxlen=20)

def bucket_action(a: str | None) -> str:
    a = (a or "").lower()
    a = a.replace("user clicked on", "clicked")
    a = a.replace("user clicked", "clicked")
    a = re.sub(r"'([^']+)'", r"\1", a)
    return bucket(a, maxlen=28)

def edge_to_observation(edge: dict, nodes_by_id: dict) -> str:
    # NOTE: pass dicts to map_kind (not strings)
    src = nodes_by_id.get(edge.get("from"))
    dst = nodes_by_id.get(edge.get("to"))
    src_kind = map_kind(src)
    dst_kind = map_kind(dst)

    window_b = bucket_window((dst or {}).get("window") or (dst or {}).get("label") or (dst or {}).get("element_type"))
    action_b = bucket_action(edge.get("action"))

    return f"{src_kind}->{dst_kind}|{window_b}|{action_b}"

def load_sequence_from_json(path: str) -> list[str]:
    data = json.load(open(path, "r", encoding="utf-8")) if path != "-" else json.load(sys.stdin)
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    obs = [edge_to_observation(e, nodes) for e in data.get("edges", [])]
    return obs

# ---------------------------
# HMM core (discrete emissions)
# ---------------------------

def normalize(v):
    s = sum(v)
    if s <= 0: return [1.0/len(v)]*len(v)
    return [x/s for x in v]

def rand_stochastic(n, m=None):
    """Random row-stochastic matrix (n x m) or vector (n)."""
    if m is None:
        v = [random.random()+1e-3 for _ in range(n)]
        return normalize(v)
    A = []
    for _ in range(n):
        row = [random.random()+1e-3 for _ in range(m)]
        A.append(normalize(row))
    return A

class HMM:
    def __init__(self, K: int, V: int, smoothing: float = 1e-3, seed: int = 7):
        """
        K: hidden state count
        V: vocabulary size (number of observation symbols)
        """
        random.seed(seed)
        self.K = K; self.V = V
        self.smoothing = smoothing
        self.pi = rand_stochastic(K)
        self.A  = rand_stochastic(K, K)
        self.B  = rand_stochastic(K, V)

    # --------- Forward-Backward with scaling ---------
    def forward(self, obs_seq):
        T = len(obs_seq)
        alpha = [[0.0]*self.K for _ in range(T)]
        c = [0.0]*T  # scaling factors

        # init
        for i in range(self.K):
            alpha[0][i] = self.pi[i] * self.B[i][obs_seq[0]]
        c[0] = sum(alpha[0]) or 1e-12
        alpha[0] = [a/c[0] for a in alpha[0]]

        # induct
        for t in range(1, T):
            for j in range(self.K):
                s = 0.0
                for i in range(self.K):
                    s += alpha[t-1][i] * self.A[i][j]
                alpha[t][j] = s * self.B[j][obs_seq[t]]
            c[t] = sum(alpha[t]) or 1e-12
            alpha[t] = [a/c[t] for a in alpha[t]]

        loglik = sum(math.log(ct) for ct in c)  # log P(O) = - sum log c_t  (but we used 1/ct)
        return alpha, c, -loglik  # return true log-likelihood

    def backward(self, obs_seq, c):
        T = len(obs_seq)
        beta = [[0.0]*self.K for _ in range(T)]

        # init
        for i in range(self.K):
            beta[T-1][i] = 1.0 / c[T-1]

        # induct
        for t in range(T-2, -1, -1):
            for i in range(self.K):
                s = 0.0
                for j in range(self.K):
                    s += self.A[i][j] * self.B[j][obs_seq[t+1]] * beta[t+1][j]
                beta[t][i] = s / c[t]
        return beta

    # --------- EM (Baum–Welch) ---------
    def fit(self, sequences: list[list[int]], max_iter=50, tol=1e-4, verbose=True):
        last_ll = None
        for it in range(1, max_iter+1):
            # expected counts
            pi_hat = [self.smoothing]*self.K
            A_hat  = [[self.smoothing]*self.K for _ in range(self.K)]
            B_hat  = [[self.smoothing]*self.V for _ in range(self.K)]
            total_ll = 0.0

            for obs_seq in sequences:
                if not obs_seq: continue
                alpha, c, ll = self.forward(obs_seq)
                beta = self.backward(obs_seq, c)
                total_ll += ll

                T = len(obs_seq)

                # gamma_t(i) = P(z_t = i | O)
                gamma = [[0.0]*self.K for _ in range(T)]
                for t in range(T):
                    denom = 0.0
                    for i in range(self.K):
                        gamma[t][i] = alpha[t][i] * beta[t][i]
                        denom += gamma[t][i]
                    if denom <= 0: denom = 1e-12
                    for i in range(self.K):
                        gamma[t][i] /= denom

                # xi_t(i,j) = P(z_t=i, z_{t+1}=j | O)
                xi = [[[0.0]*self.K for _ in range(self.K)] for _ in range(T-1)]
                for t in range(T-1):
                    denom = 0.0
                    for i in range(self.K):
                        for j in range(self.K):
                            val = alpha[t][i] * self.A[i][j] * self.B[j][obs_seq[t+1]] * beta[t+1][j]
                            xi[t][i][j] = val
                            denom += val
                    if denom <= 0: denom = 1e-12
                    for i in range(self.K):
                        for j in range(self.K):
                            xi[t][i][j] /= denom

                # accumulate expected counts
                for i in range(self.K):  # initial
                    pi_hat[i] += gamma[0][i]

                for t in range(T-1):     # transitions
                    for i in range(self.K):
                        for j in range(self.K):
                            A_hat[i][j] += xi[t][i][j]

                for t in range(T):       # emissions
                    o = obs_seq[t]
                    for i in range(self.K):
                        B_hat[i][o] += gamma[t][i]

            # M-step: normalize to get distributions
            self.pi = normalize(pi_hat)
            self.A  = [normalize(row) for row in A_hat]
            self.B  = [normalize(row) for row in B_hat]

            if verbose:
                print(f"[EM] iter={it:02d}  loglik={total_ll:.4f}")

            if last_ll is not None and abs(total_ll - last_ll) < tol:
                if verbose:
                    print("[EM] converged.")
                break
            last_ll = total_ll

    # --------- Viterbi decoding ---------
    def viterbi(self, obs_seq: list[int]) -> tuple[list[int], float]:
        if not obs_seq:
            return [], 0.0
        T = len(obs_seq)
        dp = [[-1e18]*self.K for _ in range(T)]
        bp = [[-1]*self.K for _ in range(T)]

        # init
        o0 = obs_seq[0]
        for i in range(self.K):
            dp[0][i] = math.log(self.pi[i] + 1e-12) + math.log(self.B[i][o0] + 1e-12)
            bp[0][i] = -1

        # recursion
        for t in range(1, T):
            ot = obs_seq[t]
            for j in range(self.K):
                best = -1e18; arg = -1
                bj = math.log(self.B[j][ot] + 1e-12)
                for i in range(self.K):
                    val = dp[t-1][i] + math.log(self.A[i][j] + 1e-12) + bj
                    if val > best:
                        best, arg = val, i
                dp[t][j] = best
                bp[t][j] = arg

        # backtrack
        last_state = max(range(self.K), key=lambda i: dp[T-1][i])
        path = [last_state]
        for t in range(T-1, 0, -1):
            path.append(bp[t][path[-1]])
        path.reverse()
        return path, dp[T-1][last_state]

# ---------------------------
# CLI / glue
# ---------------------------

def build_vocab(sequences: list[list[str]]):
    vocab = {}
    for seq in sequences:
        for o in seq:
            if o not in vocab:
                vocab[o] = len(vocab)
    inv_vocab = {i: o for o, i in vocab.items()}
    return vocab, inv_vocab

def encode_sequences(seqs_str: list[list[str]], vocab: dict) -> list[list[int]]:
    return [[vocab[o] for o in seq if o in vocab] for seq in seqs_str]

def pretty_top_emissions(model: HMM, inv_vocab, top=8):
    out = []
    for k in range(model.K):
        pairs = [(p, inv_vocab[i]) for i, p in enumerate(model.B[k])]
        pairs.sort(reverse=True)
        out.append((k, pairs[:top]))
    return out

def save_params(path: Path, model: HMM, vocab: dict):
    payload = {
        "K": model.K,
        "V": model.V,
        "pi": model.pi,
        "A": model.A,
        "B": model.B,
        "vocab": vocab
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved parameters to {path}")

def load_params(path: Path) -> tuple[HMM, dict, dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    K, V = obj["K"], obj["V"]
    model = HMM(K, V)
    model.pi = obj["pi"]; model.A = obj["A"]; model.B = obj["B"]
    vocab = obj["vocab"]; inv_vocab = {i: o for o, i in vocab.items()}
    return model, vocab, inv_vocab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="Number of hidden states")
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--save", type=Path, help="Save learned params to JSON")
    ap.add_argument("--load", type=Path, help="Load params instead of training")
    ap.add_argument("--decode", action="store_true", help="After training/loading, Viterbi-decode the provided sessions")
    ap.add_argument("sessions", nargs="+", help="One or more session JSON files (flat nodes/edges schema)")
    args = ap.parse_args()

    if args.load:
        model, vocab, inv_vocab = load_params(args.load)
    else:
        # 1) Build observation sequences from sessions
        seqs_str = [load_sequence_from_json(p) for p in args.sessions]
        if all(len(s) == 0 for s in seqs_str):
            print("No observations found in input.", file=sys.stderr)
            sys.exit(1)
        vocab, inv_vocab = build_vocab(seqs_str)
        seqs = encode_sequences(seqs_str, vocab)

        # 2) Train HMM (EM)
        model = HMM(K=args.k, V=len(vocab))
        model.fit(seqs, max_iter=args.max_iter, tol=args.tol, verbose=True)

        # 3) Show top emissions per hidden state
        print("\nTop emissions per hidden state:")
        for k, top_list in pretty_top_emissions(model, inv_vocab, top=8):
            print(f"  z={k}:")
            for p, token in top_list:
                print(f"    {p:6.4f}  {token}")

        if args.save:
            save_params(args.save, model, vocab)

    # Optionally decode sequences with Viterbi
    if args.decode:
        print("\nViterbi decodes:")
        for p in args.sessions:
            seq_str = load_sequence_from_json(p)
            seq = [vocab[o] for o in seq_str if o in vocab] if not args.load else [vocab.get(o, None) for o in seq_str]
            seq = [x for x in seq if x is not None]
            path, logp = model.viterbi(seq)
            print(f"  {p}: logP={logp:.2f}")
            print("    obs:", " | ".join(seq_str))
            print("    hid:", " ".join([f"z{z}" for z in path]))


if __name__ == "__main__":
    main()
