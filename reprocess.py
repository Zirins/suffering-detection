import json
from pathlib import Path
from datetime import datetime

from pipeline import detect_with_heuristics
from heuristics.kb_mashing import SufferingDetector


def build_summary(alerts):
    # Make a quick summary of the alerts (how many, severity, types, and stuff)
    summary = {
        "total_alerts": len(alerts),
        "by_severity": {"high": 0, "medium": 0, "low": 0},
        "by_type": {}
    }

    for a in alerts:
        # Count the severity levels
        sev = a.get("severity", "low")
        summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1

        # Count how many of each alert type we found
        t = a.get("type", "unknown")
        summary["by_type"][t] = summary["by_type"].get(t, 0) + 1

    return summary


def reprocess_single_session(session_path: Path, output_dir: Path):
    # Load the old session JSON we want to rerun heuristics on
    data = json.load(open(session_path, "r", encoding="utf-8"))

    # Pull the event lists out so we can run the detectors on them
    events = {
        "mouse_events": data.get("mouse_events", []),
        "keyboard_events": data.get("keyboard_events", []),
    }


    # Run our mouse-click related heuristics again
    new_alerts = detect_with_heuristics(events["mouse_events"])

    # Check if the person was mashing keys
    detector = SufferingDetector(events)
    km = detector.detect_keyboard_mashing(detector.keyboard_events)

    if km["is_mash"]:
        # Add keyboard mashing result into the list of alerts
        new_alerts.append({
            "type": "keyboard_mashing",
            "severity": "high",
            "detected_by": "ml_heuristic",
            "confidence": km["mash_score"],
            "features": km["features"],
            "timestamp": (
                detector.keyboard_events[-1]["timestamp"]
                if detector.keyboard_events else None
            ),
            "description": f"Keyboard mashing detected (score={km['mash_score']:.2f})"
        })


    # OCR cancellation detection
    try:
        from ocr_detection import CancellationDetector
        ocr_detector = CancellationDetector()
        ocr_alerts = ocr_detector.analyze_events(events["mouse_events"])

        if ocr_alerts:
            new_alerts.extend(ocr_alerts)

    except ImportError:
        print("⚠️ OCR module not found, skipping OCR detection.")
    except Exception as e:
        print(f"⚠️ OCR detection failed: {e}")

    # Build metadata + a summary for output
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "session_file": session_path.name,
        "duration_seconds": data.get("metadata", {}).get("duration_seconds", None),
        "detection_method": "heuristic"
    }

    summary = build_summary(new_alerts)

    # Save everything into /reprocessed_alerts/

    out_file = output_dir / f"alerts_{session_path.stem}.json"

    output = {
        "metadata": metadata,
        "summary": summary,
        "alerts": new_alerts
    }

    json.dump(output, open(out_file, "w", encoding="utf-8"), indent=2)

    print(f"✔ Alerts regenerated with summary: {session_path.name} → {out_file.name}")


def batch_reprocess_sessions(root_dir="outputs"):
    """
    Go through every session_*.json file in outputs/sessions/
    and generate new alert files for all of them
    """

    root = Path(root_dir)
    sessions_dir = root / "sessions"
    out_dir = root / "reprocessed_alerts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the sessions folder actually exists
    if not sessions_dir.exists():
        print(f"Folder not found: {sessions_dir}")
        return

    # Grab all the session files
    session_files = list(sessions_dir.glob("session_*.json"))
    if not session_files:
        print("No sessions found to reprocess.")
        return

    print("\nBATCH ALERT REPROCESS MODE")
    print(f"Found {len(session_files)} sessions.\n")

    # Rerun heuristics on each session
    for sf in session_files:
        reprocess_single_session(sf, out_dir)

    print("\n=== DONE: All alerts regenerated with metadata + summary ===")


if __name__ == "__main__":
    batch_reprocess_sessions()
