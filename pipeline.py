"""
Core flow:
1. Capture GUI events → save to sessions/
2. Build workflow graph → save to workflows/
3. Run heuristic detection → save to alerts/
4. Generate summary report
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
import os
import keyboard  # CTRL + F1 exit

# Imports from existing modules
from logger import get_logger
from gui import run_sensor_session
from interpreter import WorkflowInterpreter

log = get_logger("PIPELINE")

# ----------------------------
# BLOCK CTRL + C
# ----------------------------

def block_sigint(signum, frame):
    print("\n🚫 Ctrl + C is disabled. Use Ctrl + F1 to exit.")
    # Do nothing

# ----------------------------
# CONFIG
# ----------------------------

class PipelineConfig:
    """Configuration and directory setup"""
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.sessions_dir = self.output_dir / "sessions"
        self.workflows_dir = self.output_dir / "workflows"
        self.alerts_dir = self.output_dir / "alerts"

        # Create all directories
        for directory in [self.sessions_dir, self.workflows_dir, self.alerts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        log.info(f"Pipeline initialized → outputs in {self.output_dir}")


# ----------------------------
# HEURISTIC DETECTOR
# ----------------------------

def detect_with_heuristics(events, config=None):
    """Rule-based frustration detector with established thresholds."""

    # Default thresholds (can be overridden via config)
    thresholds = {
        "rage_click_count": 3,
        "rage_click_window": 2.0,
        "hesitation_seconds": 10.0,
        "retry_count": 2,
        "retry_window": 1.0
    }

    if config and hasattr(config, 'thresholds'):
        thresholds.update(config.thresholds)

    alerts = []
    if not events:
        return alerts

    log.debug(f"Running heuristic detection on {len(events)} events")

    #  1. RAGE CLICK DETECTION
    click_positions = {}
    for e in events:
        if e.get('type') != 'mouse_click':
            continue
        pos = e.get('position', {})
        x, y = pos.get('x', 0), pos.get('y', 0)
        timestamp = e.get('timestamp', '')
        area_key = f"{x//50}_{y//50}"

        if area_key not in click_positions:
            click_positions[area_key] = []
        click_positions[area_key].append({'time': timestamp, 'event': e})

    for key, clicks in click_positions.items():
        if len(clicks) >= thresholds["rage_click_count"]:
            try:
                times = [datetime.fromisoformat(c['time']) for c in clicks]
                span = (times[-1] - times[0]).total_seconds()

                if span <= thresholds["rage_click_window"]:
                    alerts.append({
                        'type': 'rage_click',
                        'severity': 'high',
                        'detected_by': 'heuristic',
                        'confidence': min(1.0, len(clicks)/5.0),
                        'click_count': len(clicks),
                        'time_span': round(span, 2),
                        'timestamp': clicks[0]['time'],
                        'location': key,
                        'description': f"{len(clicks)} rapid clicks in {span:.1f}s (possible UI freeze or user frustration)"
                    })
                    log.info(f"[Heuristic] Rage click detected: {len(clicks)} clicks in {span:.1f}s")
            except Exception as e:
                log.error(f"Error processing rage click: {e}")

    #  2. LONG HESITATION DETECTION
    sorted_events = sorted(events, key=lambda e: e.get('timestamp', ''))
    for i in range(1, len(sorted_events)):
        try:
            t1 = datetime.fromisoformat(sorted_events[i-1]['timestamp'])
            t2 = datetime.fromisoformat(sorted_events[i]['timestamp'])
            delay = (t2 - t1).total_seconds()

            if delay > thresholds["hesitation_seconds"]:
                alerts.append({
                    'type': 'long_hesitation',
                    'severity': 'medium',
                    'detected_by': 'heuristic',
                    'confidence': min(1.0, delay/30.0),
                    'delay_seconds': round(delay, 1),
                    'timestamp': sorted_events[i]['timestamp'],
                    'description': f"Long pause ({delay:.1f}s) between actions (possible confusion or distraction)"
                })
                log.debug(f"[Heuristic] Long hesitation: {delay:.1f}s")
        except (ValueError, KeyError) as e:
            log.error(f"Error detecting hesitation: {e}")

    #  3. RETRY PATTERN DETECTION
    # Look for same action repeated in short succession
    for i in range(1, len(sorted_events)):
        try:
            prev = sorted_events[i-1]
            curr = sorted_events[i]

            # Check if same action repeated
            if (prev.get('action_description') == curr.get('action_description') and
                    prev.get('action_description')):

                t1 = datetime.fromisoformat(prev['timestamp'])
                t2 = datetime.fromisoformat(curr['timestamp'])
                delay = (t2 - t1).total_seconds()

                if delay <= thresholds["retry_window"]:
                    alerts.append({
                        'type': 'retry_action',
                        'severity': 'medium',
                        'detected_by': 'heuristic',
                        'confidence': 0.8,
                        'action': prev.get('action_description'),
                        'retry_delay': round(delay, 2),
                        'timestamp': curr['timestamp'],
                        'description': f"Action repeated: '{prev.get('action_description')}' (possible UI unresponsiveness)"
                    })
                    log.debug(f"[Heuristic] Retry detected: {prev.get('action_description')}")
        except Exception as e:
            log.error(f"Error detecting retry: {e}")

    #  4. CLICK PATTERN ANALYSIS
    # Flag if there's anything weird from pattern detection (from gui.py)
    for e in events:
        patterns = e.get('patterns', {})
        if patterns.get('is_rage_click') and not any(a['type'] == 'rage_click' for a in alerts):
            alerts.append({
                'type': 'rage_click_pattern',
                'severity': 'high',
                'detected_by': 'heuristic',
                'confidence': 0.9,
                'click_count': patterns.get('count', 0),
                'timestamp': e.get('timestamp'),
                'description': f"Rage click pattern detected by real-time monitor"
            })

    return alerts


# ----------------------------
# PIPELINE RUNNER
# ----------------------------

def run_pipeline(duration=30, config=None):
    """Main pipeline execution (no hmm yet, we'll add later)"""
    if config is None:
        config = PipelineConfig()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize so accessible in finally block
    events = {'mouse_events': [], 'keyboard_events': []}

    print("\n" + "="*70)
    print("SUFFERING DETECTION PIPELINE")
    print("="*70)
    print(f"Duration: {duration}s")
    print(f"Output: {config.output_dir}")
    print(f"Detection: Heuristic-based")
    print("="*70 + "\n")

    log.info("PIPELINE START")

    try:
        # STEP 1: CAPTURE GUI EVENTS
        print("Welcome! Capturing GUI events...")
        log.info(f"Starting sensor session for {duration}s")

        events = run_sensor_session(duration)

    finally:
        # ALWAYS save whatever was captured
        mouse_count = len(events.get('mouse_events', []))
        keyboard_count = len(events.get('keyboard_events', []))

        if mouse_count == 0 and keyboard_count == 0:
            print("\n⚠️  No data captured")
            log.info("No events captured")
            return None

        print(f"\nCaptured {mouse_count} mouse events, {keyboard_count} keyboard events")
        log.info(f"Captured {mouse_count} mouse, {keyboard_count} keyboard events")

        # SAVE: Raw session data
        session_file = config.sessions_dir / f"session_{timestamp}.json"
        session_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "mouse_events_count": mouse_count,
                "keyboard_events_count": keyboard_count
            },
            "mouse_events": events.get('mouse_events', []),
            "keyboard_events": events.get('keyboard_events', [])
        }

        with open(session_file, "w", encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        print(f"Saved: {session_file.name}")
        log.info(f"Session data saved to {session_file}")

        # STEP 2: BUILD WORKFLOW GRAPH
        print("\nBuilding workflow graph...")
        log.info("Building semantic workflow graph")

        interpreter = WorkflowInterpreter()
        interpreter.build_graph_from_events(events)

        stats = interpreter.calculate_statistics()
        workflow_graph = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_session": f"session_{timestamp}.json",
                "statistics": stats
            },
            "nodes": interpreter.nodes,
            "edges": interpreter.edges,
            "workflows": interpreter.detect_workflows()
        }

        print(f"Built graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        log.info(f"Workflow graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        # SAVE: Workflow graph
        workflow_file = config.workflows_dir / f"workflow_{timestamp}.json"
        with open(workflow_file, "w", encoding='utf-8') as f:
            json.dump(workflow_graph, f, indent=2)
        print(f"Saved: {workflow_file.name}")
        log.info(f"Workflow graph saved to {workflow_file}")

        # STEP 3: RUN ANOMALY DETECTION
        print("\nRunning anomaly detection...")
        log.info("Starting heuristic detection")

        all_alerts = detect_with_heuristics(events.get("mouse_events", []), config=config)

        print(f"Detected {len(all_alerts)} anomalies")
        log.info(f"Detection complete: {len(all_alerts)} alerts")

        # SAVE: Detection results
        alerts_file = config.alerts_dir / f"alerts_{timestamp}.json"

        # Calculate summary statistics
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        type_counts = {}

        for alert in all_alerts:
            severity = alert.get('severity', 'low')
            alert_type = alert.get('type', 'unknown')

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1

        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "session_file": f"session_{timestamp}.json",
                "workflow_file": f"workflow_{timestamp}.json",
                "duration_seconds": duration,
                "detection_method": "heuristic"
            },
            "summary": {
                "total_alerts": len(all_alerts),
                "by_severity": severity_counts,
                "by_type": type_counts
            },
            "alerts": all_alerts
        }

        with open(alerts_file, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {alerts_file.name}")
        log.info(f"Alerts saved to {alerts_file}")


        # STEP 4: SUMMARY REPORT
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(f"Duration: {duration}s")
        print(f"Events: {mouse_count} mouse, {keyboard_count} keyboard")
        print(f"Workflow: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        if len(all_alerts) > 0:
            print(f"\nAlerts: {len(all_alerts)} total")
            print(f"  High: {severity_counts['high']}")
            print(f"  Medium: {severity_counts['medium']}")
            print(f"  Low: {severity_counts['low']}")

            if type_counts:
                print(f"\nBy Type:")
                for alert_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    print(f"{alert_type}: {count}")
        else:
            print("\n NICE! No anomalies detected")

        print(f"\nOutput Files:")
        print(f" -- sessions/{session_file.name}")
        print(f" -- workflows/{workflow_file.name}")
        print(f" -- alerts/{alerts_file.name}")
        print("="*70 + "\n")

        log.info("PIPELINE COMPLETE")

        return output


# ----------------------------
# CLI ENTRY POINT
# ----------------------------



def main():
    parser = argparse.ArgumentParser(
        description="Suffering Detection Pipeline (Heuristic-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Recording duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )

    args = parser.parse_args()

    # BLOCK CTRL + C
    signal.signal(signal.SIGINT, block_sigint)

    # TERMINAL-LEVEL DISABLE CTRL+C (Unix/Linux only)
    if os.name != 'nt':
        os.system("stty intr undef")

    # CTRL + F1 EXIT
    def force_exit():
        print("\n🔥 Ctrl + F1 pressed. Exiting pipeline...\n")
        log.warning("Pipeline manually exited via Ctrl + F1")
        import gui
        gui.EXIT_FLAG = True

        # Restore Ctrl+C (Unix/Linux only)
        if os.name != 'nt':
            os.system("stty intr ^C")

        keyboard.unhook_all()

    keyboard.add_hotkey("ctrl+f1", force_exit)

    # Initialize config
    config = PipelineConfig(output_dir=args.output)

    try:
        # Run pipeline
        results = run_pipeline(duration=args.duration, config=config)

        # Exit message
        if results and results['summary']['total_alerts'] > 0:
            print("Anomalies detected! Check outputs/alerts/ for details.\n")
        else:
            print("Session completed with no anomalies detected.\n")

    except KeyboardInterrupt:
        print("\n\n INTERRUPTION: Pipeline interrupted by user.\n")
        log.warning("Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n CRITICAL NOTICE: Pipeline failed: {e}\n")
        if os.name != 'nt':
            os.system("stty intr ^C")
        raise

    finally:
        if os.name != 'nt':
            os.system("stty intr ^C")  # ALWAYS restore ctrl+c!




if __name__ == "__main__":
    main()