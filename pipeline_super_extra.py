#!/usr/bin/env python3
"""
pipeline.py - Unified Suffering Detection Pipeline (Hybrid Multi-Model)

Orchestrates GUI sensor, workflow interpreter, and multiple detection models
(HMM, LSTM, Transformer, Heuristics) into one intelligent, continuous workflow.

Features:
  - Automatic model detection and ensemble detection
  - Smart batching (30s intervals, 5-min sliding window)
  - CLI flags for power users (--models, --disable)
  - Optional interactive mode (--interactive)
  - Comprehensive logging and session management

Modes:
  detect    - Real-time anomaly detection (default)
  collect   - Just capture data for training
  train     - Train models from collected sessions

Usage:
  # Default: Use all available models
  python pipeline.py

  # Select specific models
  python pipeline.py --models hmm,lstm

  # Disable specific models
  python pipeline.py --disable heuristic

  # Interactive configuration
  python pipeline.py --interactive

  # Collection mode
  python pipeline.py --mode collect --duration 600

  # Training mode
  python pipeline.py --mode train --train-hmm sessions/*.json
"""

import argparse
import json
import time
import threading
import signal
import sys
import math
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional

# Import logger (from team's logger.py)
try:
    from logger import get_logger
    log = get_logger("PIPELINE")
    LOGGER_AVAILABLE = True
except ImportError:
    # Fallback if logger.py not available
    import logging
    log = logging.getLogger("PIPELINE")
    LOGGER_AVAILABLE = False

# Import from existing modules
try:
    from gui import (
        start_mouse_listener,
        start_keyboard_listener,
        mouse_events,
        keyboard_events,
        element_position_cache
    )
    GUI_AVAILABLE = True
except ImportError:
    log.warning("gui.py not found. Running in mock mode.")
    GUI_AVAILABLE = False
    mouse_events = []
    keyboard_events = []

try:
    from interpreter import WorkflowInterpreter
    INTERPRETER_AVAILABLE = True
except ImportError:
    log.warning("interpreter.py not found. Workflow analysis disabled.")
    INTERPRETER_AVAILABLE = False

try:
    from detect_hmm import load_params, viterbi
    HMM_AVAILABLE = True
except ImportError:
    log.warning("detect_hmm.py not found. HMM detection disabled.")
    HMM_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

class PipelineConfig:
    """Central configuration for the pipeline"""
    def __init__(self):
        self.batch_interval = 30      # Process every 30 seconds
        self.window_size = 300         # Keep last 5 minutes in memory
        self.output_dir = Path("outputs")
        self.sessions_dir = self.output_dir / "sessions"
        self.workflows_dir = self.output_dir / "workflows"
        self.alerts_dir = self.output_dir / "alerts"
        self.models_dir = self.output_dir / "models"

        # Create directories
        for d in [self.sessions_dir, self.workflows_dir,
                  self.alerts_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        log.info(f"Pipeline configured with {self.batch_interval}s batches")


# ============================================================================
# Multi-Model Manager with Ensemble Detection
# ============================================================================

class ModelManager:
    """Manages multiple detection models with automatic discovery"""

    def __init__(self, models_dir: Path, enabled_models: Optional[List[str]] = None,
                 disabled_models: Optional[List[str]] = None):
        self.models_dir = Path(models_dir)
        self.available_models = {}
        self.enabled_models = enabled_models  # None = all enabled
        self.disabled_models = disabled_models or []

        self.detect_available_models()
        log.info(f"ModelManager initialized with {len(self.available_models)} models")

    def detect_available_models(self):
        """Scan for available trained models"""
        print("\n" + "="*70)
        print("🔍 DETECTING AVAILABLE MODELS")
        print("="*70)

        # Heuristics always available (unless explicitly disabled)
        if 'heuristic' not in self.disabled_models:
            self.available_models['heuristic'] = {
                'type': 'heuristic',
                'loaded': True,
                'description': 'Rule-based detection (rage clicks, patterns)'
            }
            print("  ✓ Heuristic detector (rule-based)")

        # Check for HMM model
        hmm_path = self.models_dir / "params.json"
        if hmm_path.exists() and HMM_AVAILABLE and 'hmm' not in self.disabled_models:
            try:
                pi, A, B, vocab = load_params(hmm_path)
                self.available_models['hmm'] = {
                    'type': 'hmm',
                    'loaded': True,
                    'pi': pi, 'A': A, 'B': B, 'vocab': vocab,
                    'path': hmm_path,
                    'description': f'Hidden Markov Model ({len(vocab)} vocab, {len(pi)} states)'
                }
                print(f"  ✓ HMM model - {len(vocab)} observations, {len(pi)} hidden states")
                log.info(f"Loaded HMM from {hmm_path}")
            except Exception as e:
                print(f"  ✗ HMM model found but failed to load: {e}")
                log.error(f"HMM load failed: {e}")
        elif 'hmm' not in self.disabled_models:
            print("  ○ HMM not available (train with: python train_hmm.py)")

        # LSTM (placeholder for future)
        lstm_path = self.models_dir / "lstm_model.pth"
        if lstm_path.exists() and 'lstm' not in self.disabled_models:
            print(f"  ○ LSTM found but loading not implemented yet")
            # Future: self.available_models['lstm'] = {...}

        # Transformer (placeholder for future)
        transformer_path = self.models_dir / "transformer_model.pth"
        if transformer_path.exists() and 'transformer' not in self.disabled_models:
            print(f"  ○ Transformer found but loading not implemented yet")
            # Future: self.available_models['transformer'] = {...}

        # Filter by enabled_models if specified
        if self.enabled_models:
            filtered = {k: v for k, v in self.available_models.items()
                        if k in self.enabled_models}
            self.available_models = filtered

        print(f"\n📊 Active models: {len(self.available_models)}")
        for name in self.available_models:
            print(f"   • {name}")
        print("="*70 + "\n")

    def detect_with_heuristics(self, events: List[Dict]) -> List[Dict]:
        """Heuristic-based detection (rage clicks, rapid retries, hesitations)"""
        alerts = []

        if not events:
            return alerts

        log.debug(f"Running heuristic detection on {len(events)} events")

        # --- Rage Click Detection ---
        click_positions = {}
        for event in events:
            if event.get('type') != 'mouse_click':
                continue

            pos = event.get('position', {})
            x, y = pos.get('x', 0), pos.get('y', 0)
            timestamp = event.get('timestamp', '')

            # 50x50 pixel grid
            area_key = f"{x//50}_{y//50}"

            if area_key not in click_positions:
                click_positions[area_key] = []
            click_positions[area_key].append({'time': timestamp, 'event': event})

        # Check for rage clicking: 3+ clicks in 2 seconds
        for area_key, clicks in click_positions.items():
            if len(clicks) >= 3:
                try:
                    times = [datetime.fromisoformat(c['time']) for c in clicks]
                    time_span = (times[-1] - times[0]).total_seconds()

                    if time_span <= 2.0:
                        alerts.append({
                            'type': 'rage_click',
                            'severity': 'high',
                            'detected_by': 'heuristic',
                            'confidence': min(1.0, len(clicks) / 5.0),  # Scale with click count
                            'click_count': len(clicks),
                            'time_span': round(time_span, 2),
                            'area': area_key,
                            'timestamp': clicks[0]['time'],
                            'description': f"Rage clicking: {len(clicks)} clicks in {time_span:.1f}s"
                        })
                        log.info(f"RAGE CLICK detected: {len(clicks)} clicks in {time_span:.1f}s")
                except Exception as e:
                    log.error(f"Error in rage click detection: {e}")

        # --- Long Hesitation Detection ---
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', ''))
        for i in range(1, len(sorted_events)):
            try:
                t1 = datetime.fromisoformat(sorted_events[i-1]['timestamp'])
                t2 = datetime.fromisoformat(sorted_events[i]['timestamp'])
                delay = (t2 - t1).total_seconds()

                if delay > 10:  # 10+ second pause
                    alerts.append({
                        'type': 'long_hesitation',
                        'severity': 'medium',
                        'detected_by': 'heuristic',
                        'confidence': min(1.0, delay / 30.0),
                        'delay_seconds': round(delay, 1),
                        'timestamp': sorted_events[i]['timestamp'],
                        'description': f"Long pause: {delay:.1f}s between actions"
                    })
                    log.debug(f"Long hesitation: {delay:.1f}s")
            except:
                pass

        return alerts

    def detect_with_hmm(self, workflow_graph: Dict) -> List[Dict]:
        """HMM-based workflow anomaly detection"""
        if 'hmm' not in self.available_models:
            return []

        alerts = []
        model = self.available_models['hmm']

        log.debug("Running HMM detection on workflow graph")

        try:
            nodes = {n['id']: n for n in workflow_graph.get('nodes', [])}
            edges = workflow_graph.get('edges', [])

            if not edges:
                return alerts

            # Build observations from edges
            observations = []
            for edge in edges:
                from_node = nodes.get(edge.get('from', ''), {})
                to_node = nodes.get(edge.get('to', ''), {})

                # Format: "source_type->dest_type|action"
                obs = f"{from_node.get('type', 'unknown')}->{to_node.get('type', 'unknown')}|{edge.get('action', '')}"
                observations.append(obs)

            # Encode observations
            vocab = model['vocab']
            obs_idx = []
            unknown_obs = []

            for i, obs in enumerate(observations):
                if obs in vocab:
                    obs_idx.append(vocab[obs])
                else:
                    unknown_obs.append((i, obs))

            if not obs_idx:
                # All observations unknown - potential anomaly
                alerts.append({
                    'type': 'unknown_workflow_pattern',
                    'severity': 'high',
                    'detected_by': 'hmm',
                    'confidence': 0.9,
                    'timestamp': datetime.now().isoformat(),
                    'description': 'Completely unfamiliar workflow (all observations unknown)'
                })
                log.warning("HMM: All observations unknown")
                return alerts

            # Run Viterbi algorithm
            path, logp = viterbi(model['pi'], model['A'], model['B'], obs_idx)

            # Analyze each step for anomalies
            for t, state in enumerate(path):
                if t >= len(obs_idx):
                    break

                obs = obs_idx[t]
                emission_prob = model['B'][state][obs]

                # Flag low-probability emissions (threshold: 0.001)
                if emission_prob < 0.001:
                    surprise = -math.log(max(emission_prob, 1e-12))

                    alerts.append({
                        'type': 'anomalous_transition',
                        'severity': 'medium',
                        'detected_by': 'hmm',
                        'confidence': min(1.0, surprise / 10.0),
                        'emission_prob': round(emission_prob, 6),
                        'surprise': round(surprise, 2),
                        'observation': observations[t] if t < len(observations) else 'unknown',
                        'hidden_state': f"z{state}",
                        'timestamp': datetime.now().isoformat(),
                        'description': f'Unusual workflow step (p={emission_prob:.2e})'
                    })
                    log.info(f"HMM anomaly: p={emission_prob:.2e} for '{observations[t]}'")

            # Flag unknown observations
            for idx, obs in unknown_obs:
                alerts.append({
                    'type': 'unknown_observation',
                    'severity': 'low',
                    'detected_by': 'hmm',
                    'confidence': 0.7,
                    'observation': obs,
                    'timestamp': datetime.now().isoformat(),
                    'description': f'Never-seen-before action: {obs}'
                })

        except Exception as e:
            log.error(f"HMM detection error: {e}")

        return alerts

    def aggregate_alerts(self, all_model_alerts: Dict[str, List[Dict]]) -> List[Dict]:
        """Combine alerts from multiple models, calculate consensus confidence"""
        # Flatten all alerts
        all_alerts = []
        for model_name, alerts in all_model_alerts.items():
            for alert in alerts:
                all_alerts.append(alert)

        # Group similar alerts (same type, similar timestamp)
        # For now, just return all (future: deduplicate/merge)

        return all_alerts

    def detect_anomalies(self, events: List[Dict], workflow_graph: Dict) -> Dict[str, Any]:
        """Run all available models and return aggregated results"""
        results = {
            'by_model': {},
            'summary': {},
            'alerts': []
        }

        log.info(f"Running anomaly detection with {len(self.available_models)} models")

        # Run each available model
        if 'heuristic' in self.available_models:
            heuristic_alerts = self.detect_with_heuristics(events)
            results['by_model']['heuristic'] = heuristic_alerts

        if 'hmm' in self.available_models:
            hmm_alerts = self.detect_with_hmm(workflow_graph)
            results['by_model']['hmm'] = hmm_alerts

        # Future: Add LSTM, Transformer
        # if 'lstm' in self.available_models and len(events) > 20:
        #     lstm_alerts = self.detect_with_lstm(events, workflow_graph)
        #     results['by_model']['lstm'] = lstm_alerts

        # Aggregate alerts
        results['alerts'] = self.aggregate_alerts(results['by_model'])

        # Summary statistics
        results['summary'] = {
            'total_alerts': len(results['alerts']),
            'by_type': {},
            'by_severity': {'high': 0, 'medium': 0, 'low': 0},
            'models_used': list(results['by_model'].keys())
        }

        for alert in results['alerts']:
            alert_type = alert.get('type', 'unknown')
            severity = alert.get('severity', 'low')

            results['summary']['by_type'][alert_type] = \
                results['summary']['by_type'].get(alert_type, 0) + 1
            results['summary']['by_severity'][severity] += 1

        return results


# ============================================================================
# Pipeline Core
# ============================================================================

class SufferingDetectionPipeline:
    """Main orchestrator for the detection pipeline"""

    def __init__(self, config: PipelineConfig, model_manager: ModelManager):
        self.config = config
        self.models = model_manager
        self.running = False
        self.session_start = None
        self.batch_count = 0
        self.total_alerts = 0

        log.info("Pipeline initialized")

    def start_monitoring(self):
        """Start GUI event capture"""
        if not GUI_AVAILABLE:
            log.error("GUI monitoring not available")
            print("⚠️  GUI module not found. Cannot start monitoring.")
            return None, None

        print("🎯 Starting GUI monitoring...")
        log.info("Starting mouse and keyboard listeners")

        mouse_listener = start_mouse_listener()
        keyboard_listener = start_keyboard_listener(record_text=False)

        return mouse_listener, keyboard_listener

    def process_batch(self):
        """Process accumulated events through interpreter and detectors"""
        self.batch_count += 1
        timestamp = datetime.now()

        print(f"\n{'='*70}")
        print(f"📦 BATCH #{self.batch_count} - {timestamp.strftime('%H:%M:%S')}")
        print(f"{'='*70}")

        log.info(f"Processing batch #{self.batch_count}")

        # Snapshot current events
        events_snapshot = {
            'mouse_events': list(mouse_events),
            'keyboard_events': list(keyboard_events),
            'timestamp': timestamp.isoformat(),
            'batch': self.batch_count
        }

        if not events_snapshot['mouse_events']:
            print("  ℹ️  No events captured in this batch\n")
            return

        print(f"  📊 Events: {len(events_snapshot['mouse_events'])} mouse, "
              f"{len(events_snapshot['keyboard_events'])} keyboard")

        # Save session
        session_file = self.config.sessions_dir / f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, 'w') as f:
            json.dump(events_snapshot, f, indent=2)
        print(f"  💾 Session saved: {session_file.name}")
        log.debug(f"Saved session to {session_file}")

        # Build workflow graph
        if not INTERPRETER_AVAILABLE:
            print("  ⚠️  Interpreter not available")
            return

        interpreter = WorkflowInterpreter()
        interpreter.build_graph_from_events(events_snapshot)

        stats = interpreter.calculate_statistics()
        workflow_graph = {
            'metadata': {
                'generated_at': timestamp.isoformat(),
                'batch': self.batch_count,
                'statistics': stats
            },
            'nodes': interpreter.nodes,
            'edges': interpreter.edges,
            'workflows': interpreter.detect_workflows()
        }

        # Save workflow
        workflow_file = self.config.workflows_dir / f"workflow_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(workflow_file, 'w') as f:
            json.dump(workflow_graph, f, indent=2)
        print(f"  📈 Workflow saved: {workflow_file.name}")
        print(f"     → {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        # Run anomaly detection
        print(f"\n  🔍 Running detection with {len(self.models.available_models)} models...")
        detection_results = self.models.detect_anomalies(
            events_snapshot['mouse_events'],
            workflow_graph
        )

        # Display results
        summary = detection_results['summary']
        total = summary['total_alerts']
        self.total_alerts += total

        if total > 0:
            print(f"\n  {'🚨 ALERTS DETECTED' if total > 0 else '✅ No anomalies'}")
            print(f"  Total: {total} alerts")
            print(f"  High: {summary['by_severity']['high']}, "
                  f"Medium: {summary['by_severity']['medium']}, "
                  f"Low: {summary['by_severity']['low']}")

            # Show breakdown by model
            for model_name, alerts in detection_results['by_model'].items():
                if alerts:
                    print(f"\n    [{model_name.upper()}] {len(alerts)} alert(s):")
                    for alert in alerts[:2]:  # Show first 2
                        severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(alert.get('severity', 'low'), '⚪')
                        print(f"      {severity_emoji} {alert['type']}: {alert['description']}")
                    if len(alerts) > 2:
                        print(f"      ... and {len(alerts) - 2} more")

            # Save alerts
            alert_file = self.config.alerts_dir / f"alerts_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(alert_file, 'w') as f:
                json.dump(detection_results, f, indent=2)
            print(f"\n  💾 Alerts saved: {alert_file.name}")
            log.warning(f"Batch #{self.batch_count}: {total} alerts detected")
        else:
            print(f"\n  ✅ No anomalies detected")

        print(f"\n{'='*70}\n")

    def run_detection_mode(self):
        """Continuous real-time detection"""
        print("\n" + "="*70)
        print("🚀 SUFFERING DETECTION PIPELINE - LIVE MODE")
        print("="*70)
        print(f"⏱️  Batch interval: {self.config.batch_interval}s")
        print(f"📁 Output: {self.config.output_dir}")
        print(f"🤖 Models: {len(self.models.available_models)}")
        print("\nPress Ctrl+C to stop\n")
        print("="*70)

        log.info("Starting detection mode")

        # Start monitoring
        mouse_listener, keyboard_listener = self.start_monitoring()

        if not mouse_listener:
            print("❌ Failed to start monitoring")
            return

        self.running = True
        self.session_start = datetime.now()

        try:
            while self.running:
                time.sleep(self.config.batch_interval)
                self.process_batch()

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("🛑 STOPPING PIPELINE")
            print("="*70)

        finally:
            # Cleanup
            if mouse_listener:
                mouse_listener.stop()
            if keyboard_listener:
                keyboard_listener.stop()

            # Final stats
            duration = (datetime.now() - self.session_start).total_seconds()
            print(f"\n📊 SESSION SUMMARY:")
            print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
            print(f"  Batches: {self.batch_count}")
            print(f"  Events: {len(mouse_events)} mouse, {len(keyboard_events)} keyboard")
            print(f"  Alerts: {self.total_alerts} total")
            print("\n✅ Pipeline stopped cleanly\n")

            log.info(f"Session ended: {self.batch_count} batches, {self.total_alerts} alerts")

    def run_collection_mode(self, duration: int):
        """Data collection mode (no detection)"""
        print("\n" + "="*70)
        print("📦 DATA COLLECTION MODE")
        print("="*70)
        print(f"⏱️  Duration: {duration}s ({duration/60:.1f} min)")
        print("\nPress Ctrl+C to stop early\n")

        log.info(f"Starting collection mode for {duration}s")

        mouse_listener, keyboard_listener = self.start_monitoring()

        if not mouse_listener:
            return

        self.session_start = datetime.now()

        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\n⚠️  Collection stopped early")
        finally:
            if mouse_listener:
                mouse_listener.stop()
            if keyboard_listener:
                keyboard_listener.stop()

            # Save collected data
            timestamp = datetime.now()
            session_file = self.config.sessions_dir / f"collected_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

            data = {
                'session': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': timestamp.isoformat(),
                    'duration_sec': (timestamp - self.session_start).total_seconds()
                },
                'mouse_events': list(mouse_events),
                'keyboard_events': list(keyboard_events)
            }

            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"\n✅ Data saved: {session_file}")
            print(f"   Events: {len(mouse_events)} mouse, {len(keyboard_events)} keyboard\n")

            log.info(f"Collection complete: {len(mouse_events)} events saved")


# ============================================================================
# Interactive Mode (CLI Menu)
# ============================================================================

def interactive_mode():
    """Interactive configuration menu"""
    print("\n" + "="*70)
    print("🎛️  INTERACTIVE PIPELINE CONFIGURATION")
    print("="*70)

    # Mode selection
    print("\n1️⃣  Select Mode:")
    print("  1. Detection (real-time anomaly detection)")
    print("  2. Collection (just capture data)")
    print("  3. Training (train models from data)")

    mode_choice = input("\nChoice [1]: ").strip() or "1"
    mode_map = {"1": "detect", "2": "collect", "3": "train"}
    mode = mode_map.get(mode_choice, "detect")

    # Model selection (if detection mode)
    selected_models = None
    if mode == "detect":
        print("\n2️⃣  Select Models (or press Enter for all):")
        print("  Available: heuristic, hmm, lstm, transformer")
        models_input = input("\nModels (comma-separated) [all]: ").strip()
        if models_input:
            selected_models = [m.strip() for m in models_input.split(',')]

    # Duration (if collection/train)
    duration = 300
    if mode in ["collect", "train"]:
        duration_input = input("\n⏱️  Duration in seconds [300]: ").strip()
        if duration_input.isdigit():
            duration = int(duration_input)

    # Build args
    class Args:
        pass

    args = Args()
    args.mode = mode
    args.models = Path("outputs/models")
    args.output = Path("outputs")
    args.interval = 30
    args.duration = duration
    args.enabled_models = selected_models
    args.disabled_models = []
    args.interactive = False  # Already in interactive

    print("\n" + "="*70)
    print("✅ Configuration complete! Starting pipeline...")
    print("="*70 + "\n")

    return args


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Suffering Detection Pipeline - Hybrid Multi-Model System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    parser.add_argument('--mode', choices=['detect', 'collect', 'train'],
                        default='detect',
                        help='Pipeline mode (default: detect)')

    # Model control
    parser.add_argument('--models', type=str,
                        help='Comma-separated list of models to use (e.g., hmm,lstm)')
    parser.add_argument('--disable', type=str,
                        help='Comma-separated list of models to disable')
    parser.add_argument('--model-dir', type=Path, default=Path('outputs/models'),
                        help='Directory containing trained models')

    # General options
    parser.add_argument('--output', type=Path, default=Path('outputs'),
                        help='Output directory')
    parser.add_argument('--interval', type=int, default=30,
                        help='Batch interval in seconds (default: 30)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration for collect/train mode (seconds)')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive configuration menu')

    args = parser.parse_args()

    # Interactive mode overrides CLI args
    if args.interactive:
        args = interactive_mode()

    # Parse model selections
    enabled_models = None
    if args.models:
        enabled_models = [m.strip() for m in args.models.split(',')]

    disabled_models = []
    if args.disable:
        disabled_models = [m.strip() for m in args.disable.split(',')]

    # Setup
    config = PipelineConfig()
    config.output_dir = args.output
    config.models_dir = args.model_dir
    config.batch_interval = args.interval

    # Load models
    model_manager = ModelManager(
        config.models_dir,
        enabled_models=enabled_models,
        disabled_models=disabled_models
    )

    # Create pipeline
    pipeline = SufferingDetectionPipeline(config, model_manager)

    # Run
    if args.mode == 'detect':
        pipeline.run_detection_mode()
    elif args.mode == 'collect':
        pipeline.run_collection_mode(args.duration)
    elif args.mode == 'train':
        print("\n⚠️  Training mode not yet implemented in pipeline")
        print("   Use train_hmm.py directly for now:")
        print("   python train_hmm.py --k 5 --save outputs/models/params.json session*.json")
        sys.exit(1)


if __name__ == '__main__':
    main()