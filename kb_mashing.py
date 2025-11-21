from collections import Counter
from datetime import datetime
from math import log2
import logging

log = logging.getLogger("SufferingDetector")  # configure elsewhere

class SufferingDetector:
    """
    Takes in the events and workflows and runs ML models to determine suffering
    """
    def __init__(self, events):
        self.keyboard_events = events.get('keyboard_events', [])
        self.mouse_events = events.get('mouse_events', [])

    def detect(self):
        # Keyboard Mashing
        keyboard_mash = self.detect_keyboard_mashing(self.keyboard_events)
        if keyboard_mash["is_mash"]:
            print(f"  ⚠️  Keyboard Mashing detected {keyboard_mash}")
            log.info(keyboard_mash)
            
            

    @staticmethod
    def detect_keyboard_mashing(
        events,
        window=10,
        # speed / distribution / word-like thresholds
        cps_thresh=8.0,
        entropy_high=0.80,
        topkey_dom=0.40,
        word_low=0.35,
        # NEW: repetition thresholds (for single-key spam)
        repeat_ratio_thresh=0.50,   # ≥50% of adjacent pairs are the same
        max_run_thresh=5,           # a run of ≥5 same chars is suspicious
        # weights
        w_speed=0.35,
        w_dist=0.45,                # a bit more weight on distribution (dominance/repeat)
        w_word=0.20,
        decision=0.55,
    ):
        """
        Rule-based keyboard mashing detector using the most recent `window` key presses.
        Flags random slamming, one-key spam, and non-word-like bursts.
        """

        # ------------------------------
        # Helpers
        # ------------------------------
        def parse_ts(ts: str) -> float:
            return datetime.fromisoformat(ts).timestamp()

        def normalize_key(k: str) -> str:
            # Convert "Key.space" -> " ", "Key.enter" -> "<ENTER>", etc.
            if isinstance(k, str) and k.startswith("Key."):
                name = k.split(".", 1)[1]
                if name == "space":
                    return " "
                return f"<{name.upper()}>"
            return k

        def normalized_entropy(tokens) -> float:
            """Shannon entropy normalized to [0,1]."""
            if not tokens:
                return 0.0
            counts = Counter(tokens)
            n = len(tokens)
            if len(counts) == 1:
                return 0.0
            H = -sum((c/n) * log2(c/n) for c in counts.values())
            return H / log2(len(counts))

        def letters_only(s: str) -> str:
            return "".join(ch for ch in s if "a" <= ch <= "z")

        COMMON_BIGRAMS = {
            "th","he","in","er","an","re","on","at","en","nd","ti","es","or",
            "te","of","ed","is","it","al","ar","st","to","nt","ng","se","ha",
            "as","ou","io","le"
        }

        def bigram_hit_rate(s: str) -> float:
            if len(s) < 2:
                return 0.0
            hits = sum(1 for i in range(len(s) - 1) if s[i:i+2] in COMMON_BIGRAMS)
            return hits / (len(s) - 1)

        # keys we should NOT treat as spam when repeated (editing/navigation/modifiers)
        IGNORED_REPEAT_TOKENS = {
            "Key.backspace", "Key.enter"
        }

        # -----------------------------------
        # 1) Take the last N keyboard presses
        # -----------------------------------
        presses = [e for e in events if e.get("type") == "keyboard_press"]
        if window > 0:
            presses = presses[-window:]
        if len(presses) < 4:
            return {"is_mash": False, "mash_score": 0.0, "features": {}}

        # -----------------------------------
        # 2) Timing stats (speed / CPS)
        # -----------------------------------
        ts = [parse_ts(e["timestamp"]) for e in presses]
        dts = [b - a for a, b in zip(ts, ts[1:]) if b >= a]
        total_time = sum(dts)
        cps = (len(presses) - 1) / total_time if total_time > 0 else 0.0  # characters per second

        # ------------------------------------------------
        # 3) Key normalization + distribution information
        # ------------------------------------------------
        raw_keys = [normalize_key(str(e["key"])) for e in presses]

        # Printable set for entropy/dominance (letters/digits/punct + space)
        printable = [k.lower() for k in raw_keys if (len(k) == 1 and 32 <= ord(k) <= 126) or k == " "]
        entropy_norm = normalized_entropy(printable)

        key_counts = Counter(printable)
        top_key_freq = (key_counts.most_common(1)[0][1] / len(printable)) if printable else 0.0

        # ------------------------------------------------------
        # 4) Word-likeness (letters ratio + common bigram hits)
        # ------------------------------------------------------
        letters_seq = letters_only("".join(ch for ch in printable if ch != " "))
        letters_ratio = (len(letters_seq) / max(1, len(printable))) if printable else 0.0
        bigram_rate = bigram_hit_rate(letters_seq)

        # ------------------------------------------------------
        # 5) NEW: Repetition features (single-key spam)
        #     - consecutive repeat ratio
        #     - maximum run length of the same key
        #     Ignore utility keys (Enter/Delete/etc.) for this part.
        # ------------------------------------------------------
        # Build a "repeatable" sequence that excludes ignored tokens and spaces
        repeatable = []
        for k in raw_keys:
            kl = k.upper() if k.startswith("<") else k.lower()
            if kl not in IGNORED_REPEAT_TOKENS and k != " ":
                # keep letters/digits/punct single-char or any other literal
                if (len(k) == 1 and 32 <= ord(k) <= 126) or not k.startswith("<"):
                    repeatable.append(k.lower())

        consec_same = 0
        max_run = 1
        curr_run = 1
        for i in range(1, len(repeatable)):
            if repeatable[i] == repeatable[i-1]:
                consec_same += 1
                curr_run += 1
                if curr_run > max_run:
                    max_run = curr_run
            else:
                curr_run = 1
        repeat_ratio = consec_same / max(1, len(repeatable) - 1)

        # Scale repetition to [0,1]
        repeat_score_a = 0.0 if repeat_ratio < repeat_ratio_thresh else min(1.0, (repeat_ratio - repeat_ratio_thresh) / (1.0 - repeat_ratio_thresh))
        repeat_score_b = 0.0 if max_run < max_run_thresh else min(1.0, (max_run - max_run_thresh) / max(1, window - max_run_thresh))
        repeat_score = max(repeat_score_a, repeat_score_b)

        # ------------------------------------------------------
        # 6) Convert raw features into normalized sub-scores
        # ------------------------------------------------------
        # Speed sub-score
        speed_score = max(0.0, min(1.0, (cps - cps_thresh) / max(1e-9, cps_thresh)))

        # Distribution sub-score: max of (uniform randomness, key dominance, explicit repetition)
        uniform_score = (
            max(0.0, min(1.0, (entropy_norm - entropy_high) / max(1e-9, 1.0 - entropy_high)))
            if entropy_norm >= entropy_high else 0.0
        )
        dominance_score = (
            max(0.0, min(1.0, (top_key_freq - topkey_dom) / max(1e-9, 1.0 - topkey_dom)))
            if top_key_freq >= topkey_dom else 0.0
        )
        dist_score = max(uniform_score, dominance_score, repeat_score)

        # Word-likeness → mashiness (invert)
        word_score = 0.6 * min(1.0, letters_ratio) + 0.4 * bigram_rate
        word_mashiness = (
            0.0 if word_score >= word_low
            else max(0.0, min(1.0, 1.0 - word_score / max(1e-9, word_low)))
        )

        # ---------------------------------------
        # 7) Final score (weighted combination)
        # ---------------------------------------
        mash_score = w_speed * speed_score + w_dist * dist_score + w_word * word_mashiness
        is_mash = mash_score >= decision

        # -----------------
        # 8) Pack results
        # -----------------
        return {
            "is_mash": is_mash,
            "mash_score": round(mash_score, 3),
            "features": {
                # core signals
                "cps": round(cps, 2),
                "entropy": round(entropy_norm, 3),
                "top_key_freq": round(top_key_freq, 3),
                "letters_ratio": round(letters_ratio, 3),
                "bigram_rate": round(bigram_rate, 3),
                # new repetition metrics
                "repeat_ratio": round(repeat_ratio, 3),
                "max_run": int(max_run),
                "repeat_score": round(repeat_score, 3),
                # sub-scores
                "speed_score": round(speed_score, 3),
                "dist_score": round(dist_score, 3),
                "word_mashiness": round(word_mashiness, 3),
                "window_presses": len(presses),
            },
        }