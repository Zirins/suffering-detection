from datetime import datetime
from typing import List, Dict
import math

def detect_cursor_thrashing(
    events: List[Dict],
    window_seconds: float = 1.0,
    min_distance: float = 800.0,
    min_direction_changes: int = 4
):
    """
    Detects rapid, erratic mouse movement with no clicks
    """
    alerts = []

    # Only mouse movement events
    moves = [e for e in events if e.get("type") == "mouse_move"]
    if len(moves) < 5:
        return alerts

    window = []

    for e in moves:
        ts = datetime.fromisoformat(e["timestamp"])
        window.append((ts, e))

        # Drop old events
        window = [(t, ev) for t, ev in window if (ts - t).total_seconds() <= window_seconds]

        if len(window) < 5:
            continue

        # Compute total distance and direction changes
        total_distance = 0
        direction_changes = 0
        last_dx, last_dy = None, None

        for i in range(1, len(window)):
            x1, y1 = window[i-1][1]["position"].values()
            x2, y2 = window[i][1]["position"].values()

            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            total_distance += dist

            if last_dx is not None:
                if (dx * last_dx < 0) or (dy * last_dy < 0):
                    direction_changes += 1

            last_dx, last_dy = dx, dy

        if total_distance >= min_distance and direction_changes >= min_direction_changes:
            alerts.append({
                "type": "cursor_thrashing",
                "severity": "medium",
                "detected_by": "heuristic",
                "confidence": min(1.0, total_distance / 1500),
                "distance_px": round(total_distance, 1),
                "direction_changes": direction_changes,
                "timestamp": window[-1][1]["timestamp"],
                "description": "Rapid, erratic cursor movement without interaction"
            })
            window.clear()

    return alerts
