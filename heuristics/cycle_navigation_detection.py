import json
from pathlib import Path
from collections import defaultdict, Counter


# HELPERS

def extract_state_sequence(workflow_graph):
    """Return the ordered list of state labels."""
    nodes = {n["id"]: n for n in workflow_graph["nodes"]}

    # Ordered by edges (sorted by timestamp)
    edges = sorted(workflow_graph["edges"], key=lambda e: e["timestamp"])
    seq = []

    for e in edges:
        s = nodes[e["to"]]["label"]
        seq.append(s)

    return seq


def find_cycles_in_sequence(seq):
    """
    Detect cycles in the navigation sequence.
    Returns a list of alerts.
    """

    alerts = []

    # Count occurrences of screens
    freq = Counter(seq)

    # Any screen visited 3+ times = potential confusion
    for screen, count in freq.items():
        if count >= 3:
            alerts.append({
                "type": "cyclical_navigation",
                "severity": "medium",
                "pattern": "revisit_spike",
                "screen": screen,
                "count": count,
                "description": f"Screen '{screen}' revisited {count} times (possible navigation confusion)."
            })

    # Detect direct A <-> B oscillation (ping-pong)
    for i in range(len(seq) - 3):
        if seq[i] == seq[i+2] and seq[i+1] == seq[i+3] and seq[i] != seq[i+1]:
            alerts.append({
                "type": "cyclical_navigation",
                "severity": "high",
                "pattern": "ABAB",
                "cycle": [seq[i], seq[i+1]],
                "description": f"ABAB oscillation detected between '{seq[i]}' and '{seq[i+1]}'"
            })

    # Detect multi-step loops
    # Example: A, B, C, A  -> loop of 3
    for loop_size in range(2, 6):  # supports 2–5 element cycles
        for i in range(len(seq) - loop_size):
            block = seq[i:i+loop_size]
            next_block = seq[i+loop_size:i+2*loop_size]
            if block == next_block and len(set(block)) > 1:
                alerts.append({
                    "type": "cyclical_navigation",
                    "severity": "high",
                    "pattern": "multi_step_cycle",
                    "cycle_sequence": block,
                    "description": f"Detected repeated cycle: {block}"
                })

    return alerts



# ANALYZE ONE WORKFLOW_FILE.json


def analyze_workflow_file(path):
    try:
        data = json.loads(Path(path).read_text())
        seq = extract_state_sequence(data)
        return find_cycles_in_sequence(seq)
    except Exception as e:
        return [{
            "type": "error",
            "file": str(path),
            "error": str(e)
        }]



# BULK DIRECTORY SCANNER


def analyze_workflow_directory(dir_path):
    dir_path = Path(dir_path)
    results = []

    for file in dir_path.glob("workflow_*.json"):
        alerts = analyze_workflow_file(file)
        results.append({
            "file": file.name,
            "alerts": alerts
        })

    return results


# CLI (optional)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cyclical Navigation Detector")
    parser.add_argument("--dir", type=str, required=True, help="Directory of workflow_*.json")

    args = parser.parse_args()
    results = analyze_workflow_directory(args.dir)

    out = json.dumps(results, indent=2)
    print(out)
