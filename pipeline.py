#!/usr/bin/env python3
"""
pipeline.py - Minimal Suffering Detection Pipeline (HMM + Heuristic)

Core flow:
    1. Capture GUI events
    2. Build workflow graph
    3. Run heuristic + HMM detection
    4. Log and export results

Usage:
    python pipeline.py --duration 60
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

# === Imports from existing modules ===
from logger import get_logger
from gui import run_sensor_session  # must return dict with 'mouse_events', 'keyboard_events'
from interpreter import WorkflowInterpreter
from detect_hmm import load_params, viterbi

log = get_logger("PIPELINE")


# ============================================================================
# CONFIG
# ============================================================================

class PipelineConfig:
    def __init__(self, output_dir="outputs", models_dir="outputs/models"):
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Pipeline initialized — outputs → {self.output_dir}")


# ============================================================================
# HEURISTIC DETECTOR
# ============================================================================

def detect_with_heuristics(events):
    """Simple rule-based frustration detector."""
    alerts = []
    if not events:
        return alerts

    log.debug(f"Running heuristic detection on {len(events)} events")

    # --- Rage click detection ---
    click_positions = {}
    for e in events:
        if e.get('type') != 'mouse_click':
            continue
        pos = e.get('position', {})
        x, y = pos.get('x', 0), pos.get('y', 0)
        area_key = f"{x//50}_{y//50}"
        click_positions.setdefault(area_key, []).append(e)

    for key, clicks in click_positions.items():
        if len(clicks) >= 3:
            times = [datetime.fromisoformat(c['timestamp']) for c in clicks]
            span = (times[-1] - times[0]).total_seconds()
            if span <= 2.0:
                alerts.append({
                    'type': 'rage_click',
                    'severity': 'high',
                    'detected_by': 'heuristic',
                    'confidence': min(1.0, len(clicks)/5.0),
                    'description': f"{len(clicks)} rapid clicks in {span:.1f}s"
                })
                log.info(f"[Heuristic] Rage click: {len(clicks)} in {span:.1f}s")

    # --- Long hesitation detection ---
    sorted_events = sorted(events, key=lambda e: e.get('timestamp', ''))
    for i in range(1, len(sorted_events)):
        t1 = datetime.fromisoformat(sorted_events[i-1]['timestamp'])
        t2 = datetime.fromisoformat(sorted_events[i]['timestamp'])
        delay = (t2 - t1).total_seconds()
        if delay > 10:
            alerts.append({
                'type': 'long_hesitation',
                'severity': 'medium',
                'detected_by': 'heuristic',
                'confidence': min(1.0, delay/30.0),
                'description': f"Long pause ({delay:.1f}s) between actions"
            })
            log.debug(f"[Heuristic] Long hesitation: {delay:.1f}s")

    return alerts


# ============================================================================
# HMM DETECTOR
# ============================================================================

def detect_with_hmm(workflow_graph, model_path="outputs/models/params.json"):
    """Detect anomalies in workflow graph using trained HMM."""
    alerts = []

    if not Path(model_path).exists():
        log.warning("HMM model not found — skipping")
        return alerts

    try:
        pi, A, B, vocab = load_params(model_path)
        nodes = {n["id"]: n for n in workflow_graph.get("nodes", [])}
        edges = workflow_graph.get("edges", [])
        if not edges:
            return alerts

        observations = []
        for edge in edges:
            src = nodes.get(edge.get("from", ""), {})
            dst = nodes.get(edge.get("to", ""), {})
            obs = f"{src.get('type', 'unknown')}->{dst.get('type', 'unknown')}|{edge.get('action', '')}"
            observations.append(obs)

        obs_idx = [vocab[o] for o in observations if o in vocab]
        unknown = [o for o in observations if o not in vocab]

        if not obs_idx:
            alerts.append({
                'type': 'unknown_workflow',
                'severity': 'high',
                'detected_by': 'hmm',
                'description': "Workflow entirely unfamiliar to HMM"
            })
            return alerts

        path, logp = viterbi(pi, A, B, obs_idx)

        # Flag low-probability emissions
        for t, state in enumerate(path):
            p = B[state][obs_idx[t]]
            if p < 0.001:
                alerts.append({
                    'type': 'anomalous_transition',
                    'severity': 'medium',
                    'detected_by': 'hmm',
                    'confidence': min(1.0, -p * 100),
                    'description': f"Unusual workflow transition (p={p:.2e})"
                })
                log.info(f"[HMM] Anomaly detected: p={p:.2e}")

        for obs in unknown:
            alerts.append({
                'type': 'unknown_observation',
                'severity': 'low',
                'detected_by': 'hmm',
                'description': f"New pattern observed: {obs}"
            })

    except Exception as e:
        log.error(f"HMM detection failed: {e}")

    return alerts


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run_pipeline(duration=30):
    """Main pipeline run (single-shot)."""
    log.info("=== START PIPELINE ===")

    # 1. Capture GUI events
    events = run_sensor_session(duration)
    log.info(f"Captured {len(events['mouse_events'])} mouse events")

    # 2. Build workflow graph
    interpreter = WorkflowInterpreter()
    interpreter.build_graph_from_events(events)
    workflow_graph = {
        "nodes": interpreter.nodes,
        "edges": interpreter.edges
    }
    log.info(f"Graph built: {len(interpreter.nodes)} nodes, {len(interpreter.edges)} edges")

    # 3. Run detectors
    heuristic_alerts = detect_with_heuristics(events["mouse_events"])
    hmm_alerts = detect_with_hmm(workflow_graph)
    alerts = heuristic_alerts + hmm_alerts

    # 4. Output results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_alerts": len(alerts),
            "heuristic": len(heuristic_alerts),
            "hmm": len(hmm_alerts)
        },
        "alerts": alerts
    }

    output_path = Path("outputs") / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Pipeline complete — {len(alerts)} alerts written to {output_path}")
    return output


# ============================================================================
# CLI ENTRY
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Suffering Detection Pipeline (HMM + Heuristic)")
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
    args = parser.parse_args()

    results = run_pipeline(duration=args.duration)
    print(f"\n=== PIPELINE SUMMARY ===")
    print(f"Total alerts: {results['summary']['total_alerts']}")
    print(f"Heuristic: {results['summary']['heuristic']} | HMM: {results['summary']['hmm']}")
    print("=========================\n")
