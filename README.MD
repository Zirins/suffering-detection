# Suffering Detection Pipeline - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Core Components](#core-components)
5. [Detection Methods](#detection-methods)
6. [Usage Guide](#usage-guide)
7. [Output Structure](#output-structure)
8. [Extending the System](#extending-the-system)

---

## Overview

The Suffering Detection Pipeline is a user experience monitoring system that captures GUI interactions, builds semantic workflow graphs, and detects user frustration patterns in real-time. It combines heuristic rules with machine learning techniques to identify when users are struggling with software applications.

### Key Features
- **Real-time GUI event capture** (mouse clicks, keyboard inputs)
- **Semantic workflow graph construction** from raw interactions
- **Multi-method anomaly detection**:
  - Heuristic rules (rage clicks, hesitation, retries)
  - ML-based keyboard mashing detection
  - OCR-based cancellation pattern recognition
  - HMM-based sequence anomaly detection
- **Structured output** with severity levels and confidence scores

### Use Cases
- UX research and usability testing
- Software quality assurance
- Customer support automation
- User behavior analytics

---

## Architecture

### System Overview

```
┌─────────────────┐
│   User Actions  │
│  (Mouse + KB)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GUI Sensor     │ ← gui.py
│  (Capture)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Interpreter    │ ← interpreter.py
│  (Graph Build)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Detection Layer                │
│  ┌──────────┐  ┌──────────┐   │
│  │Heuristics│  │   ML     │   │
│  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐   │
│  │   OCR    │  │   HMM    │   │
│  └──────────┘  └──────────┘   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Alert Output   │
│  + Reports      │
└─────────────────┘
```

### Directory Structure

```
project/
├── gui.py                      # Event capture module
├── interpreter.py              # Workflow graph builder
├── pipeline.py                 # Main orchestrator
├── reprocess.py               # Batch reprocessing utility
├── logger.py                  # Centralized logging
├── detect_hmm.py              # HMM-based detection
├── heuristics/
│   ├── kb_mashing.py         # Keyboard mashing detector
│   ├── cancel_detection.py   # OCR cancellation detector
│   └── toggle_detection.py   # (Placeholder)
├── outputs/
│   ├── sessions/             # Raw event captures
│   ├── workflows/            # Semantic graphs
│   ├── alerts/               # Detection results
│   └── reprocessed_alerts/   # Batch reprocessed results
└── logs/                     # Application logs
```

---

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Windows OS** (for Win32 GUI automation)

### Dependencies

```bash
pip install pywinauto pynput keyboard pytesseract pillow
```

### Additional Setup

1. **Tesseract OCR** (for cancel detection):
   ```
   Download from: https://github.com/tesseract-ocr/tesseract
   Install to: C:\Program Files\Tesseract-OCR\
   ```

2. **Permissions**:
   - Run with administrator privileges for full GUI access
   - Allow Python through Windows Firewall if prompted

### Quick Start

```bash
# Run a 30-second monitoring session
python pipeline.py

# Custom duration (60 seconds)
python pipeline.py --duration 60

# Custom output directory
python pipeline.py --output my_results

# Reprocess existing sessions
python reprocess.py
```

---

## Core Components

### 1. GUI Sensor (`gui.py`)

**Purpose**: Captures low-level mouse and keyboard events with semantic context.

#### Key Functions

##### `get_element_at_position(x, y)`
Detects UI element under cursor using Win32 + pywinauto:
```python
element_info = {
    "type": "Button",           # Control type
    "name": "Save",             # Element label
    "window": "Notepad",        # Parent window
    "error": None               # Error if detection failed
}
```

##### `get_semantic_action(element_info, button)`
Converts raw events into human-readable descriptions:
```python
# Input: Click on "File" menu
# Output: "User clicked 'File' menu"

# Input: Click in empty pane
# Output: "User clicked in editor area of 'Notepad'"
```

##### `detect_click_pattern(x, y, timestamp)`
Identifies rapid clicking patterns:
```python
pattern = {
    "is_rage_click": True,      # 3+ clicks in <3s
    "is_retry": False,          # 2+ clicks in <1s
    "delay": 0.5,               # Time since last click
    "count": 4                  # Clicks in this area
}
```

#### Caching System

Element information is cached spatially to handle frozen UIs:
- **Grid-based**: Screen divided into 10x10 pixel regions
- **Temporal**: Automatic LRU eviction (max 1000 entries)
- **Fallback**: Uses nearby cached elements if direct query fails

#### Event Structure

**Mouse Event:**
```json
{
  "timestamp": "2025-01-15T14:32:45.123456",
  "type": "mouse_click",
  "button": "Button.left",
  "position": {"x": 450, "y": 320},
  "element": {
    "type": "Button",
    "name": "Submit",
    "window": "Form - Application"
  },
  "action_description": "User clicked 'Submit' button",
  "patterns": {
    "is_rage_click": false,
    "delay": 2.3,
    "count": 1
  }
}
```

**Keyboard Event:**
```json
{
  "timestamp": "2025-01-15T14:32:46.789012",
  "type": "keyboard_press",
  "key": "a"
}
```

---

### 2. Workflow Interpreter (`interpreter.py`)

**Purpose**: Transforms raw event logs into semantic state graphs.

#### Graph Construction Process

1. **Node Creation** (states):
   - Filter significant events (menus, buttons, tabs)
   - Skip noise (generic pane clicks, text edits)
   - Classify node type (menu, action, input, navigation)

2. **Edge Creation** (transitions):
   - Link consecutive states
   - Calculate transition duration
   - Preserve action descriptions

3. **Workflow Detection**:
   - Identify repeated patterns
   - Group related sequences
   - Extract common paths

#### Node Classification

```python
def classify_node_type(event):
    # Returns one of:
    return "menu"         # Menu interactions
    return "action"       # Button/checkbox clicks
    return "input"        # Text field interactions
    return "selection"    # Tab/list selections
    return "navigation"   # Window transitions
    return "interaction"  # Other UI interactions
```

#### Output Structure

```json
{
  "metadata": {
    "generated_at": "2025-01-15T14:35:00.000000",
    "statistics": {
      "total_nodes": 25,
      "total_edges": 24,
      "node_types": {
        "menu": 5,
        "action": 12,
        "navigation": 8
      },
      "total_duration_ms": 45000,
      "avg_transition_time_ms": 1875
    }
  },
  "nodes": [...],
  "edges": [...],
  "workflows": [...]
}
```

---

### 3. Pipeline Orchestrator (`pipeline.py`)

**Purpose**: Coordinates all components and manages execution flow.

#### Execution Stages

```python
def run_pipeline(duration, config):
    # Stage 1: Capture
    events = run_sensor_session(duration)
    save_to(sessions_dir)
    
    # Stage 2: Graph Building
    interpreter = WorkflowInterpreter()
    interpreter.build_graph_from_events(events)
    save_to(workflows_dir)
    
    # Stage 3: Detection
    alerts = []
    alerts += detect_with_heuristics(events)
    alerts += detect_keyboard_mashing(events)
    alerts += detect_cancellations(events)
    save_to(alerts_dir)
    
    # Stage 4: Reporting
    generate_summary(alerts)
```

#### Safety Features

- **CTRL+C Disabled**: Prevents accidental interruption
- **CTRL+F1 Exit**: Designated exit hotkey
- **Graceful Cleanup**: Always saves captured data
- **Error Recovery**: Continues pipeline on component failure

---

## Detection Methods

### 1. Heuristic Rules (`pipeline.py::detect_with_heuristics`)

#### Rage Click Detection
```python
Trigger: ≥3 clicks in same 50x50px area within 2 seconds
Severity: HIGH
Confidence: min(1.0, click_count / 5.0)
Interpretation: UI freeze or user frustration
```

#### Long Hesitation
```python
Trigger: >10 second delay between actions
Severity: MEDIUM
Confidence: min(1.0, delay / 30.0)
Interpretation: Confusion or distraction
```

#### Retry Pattern
```python
Trigger: Same action repeated within 1 second
Severity: MEDIUM
Confidence: 0.8
Interpretation: UI unresponsiveness
```

### 2. Keyboard Mashing (`heuristics/kb_mashing.py`)

Uses multi-factor scoring with configurable thresholds:

#### Features Analyzed

**Speed (35% weight)**
```python
cps_thresh = 8.0  # Characters per second
speed_score = (cps - thresh) / thresh
```

**Distribution (45% weight)**
```python
# Checks for:
- High entropy (uniform randomness)
- Key dominance (one key >40%)
- Repetition ratio (adjacent pairs)
- Maximum run length (≥5 same chars)

dist_score = max(uniform_score, dominance_score, repeat_score)
```

**Word-likeness (20% weight)**
```python
# Penalizes:
- Low letter ratio (<35%)
- Uncommon bigrams (<35% hit rate)

word_mashiness = 1.0 - (0.6 * letter_ratio + 0.4 * bigram_rate)
```

#### Final Decision

```python
mash_score = (0.35 * speed_score + 
              0.45 * dist_score + 
              0.20 * word_mashiness)

is_mash = mash_score >= 0.55
```

**Example Output:**
```json
{
  "is_mash": true,
  "mash_score": 0.847,
  "features": {
    "cps": 12.5,
    "entropy": 0.89,
    "top_key_freq": 0.15,
    "repeat_ratio": 0.62,
    "max_run": 7,
    "speed_score": 0.562,
    "dist_score": 0.824,
    "word_mashiness": 0.891
  }
}
```

### 3. OCR Cancellation (`heuristics/cancel_detection.py`)

Detects repeated "Cancel/Close" button clicks using Tesseract OCR:

```python
1. Capture 120x40px region around each click
2. OCR with PSM 8 (single word mode)
3. Match against: {cancel, close, exit, dismiss, abort, quit}
4. Track clicks in 15-second sliding window
5. Alert if ≥3 cancellations detected
```

**Alert Structure:**
```json
{
  "type": "repeated_cancellation",
  "severity": "high",
  "detected_by": "ocr",
  "confidence": 0.85,
  "cancel_count": 3,
  "detected_text": "cancel",
  "description": "User clicked cancel/close 3 times in 15s"
}
```

### 4. HMM Sequence Anomaly (`detect_hmm.py`)

Hidden Markov Model for detecting unusual interaction sequences.

#### Training Phase (Separate Script)

```bash
python hmm_train.py session1.json session2.json ... --out params.json
```

Learns normal patterns:
- Transition probabilities A[i,j] (state i → state j)
- Emission probabilities B[i,o] (state i → observation o)
- Initial state distribution π[i]

#### Detection Phase

```bash
python detect_hmm.py --params params.json session_new.json \
  --emit-thresh 1e-3 \
  --trans-thresh 1e-3 \
  --avg-thresh 7.5
```

**Observation Encoding:**
```python
# Format: src_kind->dst_kind|window|action
"menu->action|notepad|clicked save"
"screen->screen|chrome|navigated"
```

**Anomaly Signals:**
```python
emission_surprise = -log(B[z_t, o_t])      # Rare observation
transition_surprise = -log(A[z_{t-1}, z_t]) # Rare transition
unknown_token = observation not in vocab    # Novel pattern
```

**Per-step Analysis:**
```json
{
  "t": 5,
  "observation": "menu->action|notepad|clicked save",
  "hidden_state": "z2",
  "emission_p": 0.00043,
  "transition_p": 0.00012,
  "surprise_total": 11.247,
  "flagged": true,
  "reasons": [
    "Low emission prob: 4.30e-04",
    "Low transition prob: 1.20e-04"
  ]
}
```

---

## Usage Guide

### Basic Workflow

#### 1. Run Monitoring Session

```bash
# Standard 30-second capture
python pipeline.py

# Extended 5-minute capture
python pipeline.py --duration 300

# Save to custom location
python pipeline.py --output client_study_2024
```

**During Capture:**
- Perform normal application tasks
- System captures all mouse clicks and keystrokes
- UI interactions are semantically labeled
- Press CTRL+F1 to stop early

#### 2. Review Results

```bash
outputs/
├── sessions/session_20250115_143245.json    # Raw events
├── workflows/workflow_20250115_143245.json  # Graph
└── alerts/alerts_20250115_143245.json       # Detections
```

#### 3. Analyze Alerts

```python
import json

# Load detection results
alerts = json.load(open("outputs/alerts/alerts_*.json"))

# Get summary
print(f"Total alerts: {alerts['summary']['total_alerts']}")
print(f"High severity: {alerts['summary']['by_severity']['high']}")

# Filter specific alert types
rage_clicks = [a for a in alerts['alerts'] 
               if a['type'] == 'rage_click']
```

### Batch Reprocessing

Rerun detection on existing sessions with updated rules:

```bash
python reprocess.py
```

This:
1. Scans `outputs/sessions/` for all sessions
2. Applies current detection algorithms
3. Saves results to `outputs/reprocessed_alerts/`

**Use cases:**
- After tuning detection thresholds
- When adding new detection methods
- For comparative analysis

### HMM Training & Detection

#### Step 1: Collect Training Data

```bash
# Capture normal usage sessions
python pipeline.py --duration 120  # Session 1
python pipeline.py --duration 120  # Session 2
python pipeline.py --duration 120  # Session 3
```

#### Step 2: Train HMM

```bash
python hmm_train.py \
  outputs/sessions/session_*.json \
  --states 8 \
  --out params.json
```

Parameters:
- `--states`: Number of hidden states (default: 8)
- `--iterations`: EM iterations (default: 50)

#### Step 3: Detect Anomalies

```bash
# Test new session against trained model
python detect_hmm.py \
  --params params.json \
  outputs/sessions/session_new.json \
  --emit-thresh 1e-3 \
  --trans-thresh 1e-3 \
  --avg-thresh 7.5
```

Thresholds:
- `--emit-thresh`: Flag rare observations (default: 1e-3)
- `--trans-thresh`: Flag rare transitions (default: 1e-3)
- `--avg-thresh`: Flag session if mean surprise > threshold (default: 7.5)

---

## Output Structure

### Session File Format

```json
{
  "metadata": {
    "timestamp": "2025-01-15T14:32:45.000000",
    "duration_seconds": 30,
    "mouse_events_count": 47,
    "keyboard_events_count": 203
  },
  "mouse_events": [
    {
      "timestamp": "2025-01-15T14:32:45.123456",
      "type": "mouse_click",
      "button": "Button.left",
      "position": {"x": 450, "y": 320},
      "element": {
        "type": "Button",
        "name": "Submit",
        "window": "Application"
      },
      "action_description": "User clicked 'Submit' button",
      "patterns": {
        "is_rage_click": false,
        "is_retry": false,
        "delay": 2.3,
        "count": 1
      }
    }
  ],
  "keyboard_events": [
    {
      "timestamp": "2025-01-15T14:32:46.789012",
      "type": "keyboard_press",
      "key": "a"
    }
  ]
}
```

### Workflow Graph Format

```json
{
  "metadata": {
    "timestamp": "2025-01-15T14:35:00.000000",
    "source_session": "session_20250115_143245.json",
    "statistics": {
      "total_nodes": 25,
      "total_edges": 24,
      "node_types": {"menu": 5, "action": 12},
      "total_duration_ms": 45000
    }
  },
  "nodes": [
    {
      "id": "state_1",
      "label": "Application Start",
      "type": "start",
      "timestamp": "2025-01-15T14:32:45.000000"
    },
    {
      "id": "state_2",
      "label": "File Menu",
      "type": "menu",
      "timestamp": "2025-01-15T14:32:47.500000",
      "element_type": "MenuItem",
      "window": "Notepad",
      "position": {"x": 50, "y": 30}
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "from": "state_1",
      "to": "state_2",
      "action": "User clicked 'File' menu",
      "timestamp": "2025-01-15T14:32:47.500000",
      "duration_ms": 2500
    }
  ],
  "workflows": [
    {
      "id": "workflow_1",
      "states": ["state_1", "state_2", "state_3"],
      "start": "state_1",
      "end": "state_3"
    }
  ]
}
```

### Alert File Format

```json
{
  "metadata": {
    "timestamp": "2025-01-15T14:35:30.000000",
    "session_file": "session_20250115_143245.json",
    "workflow_file": "workflow_20250115_143245.json",
    "duration_seconds": 30,
    "detection_method": "heuristic"
  },
  "summary": {
    "total_alerts": 5,
    "by_severity": {
      "high": 2,
      "medium": 3,
      "low": 0
    },
    "by_type": {
      "rage_click": 1,
      "keyboard_mashing": 1,
      "long_hesitation": 2,
      "retry_action": 1
    }
  },
  "alerts": [
    {
      "type": "rage_click",
      "severity": "high",
      "detected_by": "heuristic",
      "confidence": 0.8,
      "click_count": 4,
      "time_span": 1.8,
      "timestamp": "2025-01-15T14:33:00.000000",
      "location": "9_6",
      "description": "4 rapid clicks in 1.8s (possible UI freeze)"
    },
    {
      "type": "keyboard_mashing",
      "severity": "high",
      "detected_by": "ml_heuristic",
      "confidence": 0.847,
      "features": {
        "cps": 12.5,
        "entropy": 0.89,
        "repeat_ratio": 0.62,
        "mash_score": 0.847
      },
      "timestamp": "2025-01-15T14:34:15.000000",
      "description": "Keyboard mashing detected (score: 0.85)"
    }
  ]
}
```

---

## Extending the System

### Adding New Heuristic Detectors

1. **Create detector file** in `heuristics/`:

```python
# heuristics/my_detector.py

def detect_my_pattern(events, threshold=5.0):
    """
    Detect custom pattern in events.
    
    Args:
        events: List of event dicts
        threshold: Detection sensitivity
        
    Returns:
        List of alert dicts
    """
    alerts = []
    
    # Your detection logic
    for event in events:
        if is_anomalous(event, threshold):
            alerts.append({
                'type': 'my_pattern',
                'severity': 'medium',
                'detected_by': 'heuristic',
                'confidence': 0.75,
                'timestamp': event['timestamp'],
                'description': 'Custom pattern detected'
            })
    
    return alerts
```

2. **Integrate into pipeline** (`pipeline.py`):

```python
from heuristics.my_detector import detect_my_pattern

# In run_pipeline(), add to detection section:
all_alerts = detect_with_heuristics(events["mouse_events"])
all_alerts += detect_my_pattern(events, threshold=5.0)  # Add this
```

### Custom Alert Types

Define new alert structures:

```python
{
    'type': 'custom_alert_name',
    'severity': 'high' | 'medium' | 'low',
    'detected_by': 'heuristic' | 'ml' | 'ocr' | 'hmm',
    'confidence': 0.0 to 1.0,
    'timestamp': 'ISO 8601 datetime',
    'description': 'Human-readable explanation',
    
    # Custom fields
    'custom_metric': 123,
    'custom_data': {...}
}
```

### Tuning Detection Thresholds

Modify in `pipeline.py::detect_with_heuristics()`:

```python
thresholds = {
    "rage_click_count": 3,        # Clicks needed
    "rage_click_window": 2.0,     # Time window (seconds)
    "hesitation_seconds": 10.0,   # Delay threshold
    "retry_count": 2,             # Retry detection
    "retry_window": 1.0           # Retry time window
}
```

For keyboard mashing, edit `heuristics/kb_mashing.py`:

```python
@staticmethod
def detect_keyboard_mashing(
    events,
    window=10,              # Keys to analyze
    cps_thresh=8.0,         # Speed threshold
    entropy_high=0.80,      # Randomness threshold
    repeat_ratio_thresh=0.50,  # Repetition threshold
    decision=0.55           # Final decision threshold
):
```

### Adding ML Models

1. **Create model file**:

```python
# models/my_model.py
import joblib

class MyFrustrationModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict(self, events):
        features = self.extract_features(events)
        predictions = self.model.predict_proba(features)
        return predictions
    
    def extract_features(self, events):
        # Feature engineering
        return feature_vector
```

2. **Integrate into pipeline**:

```python
from models.my_model import MyFrustrationModel

model = MyFrustrationModel('models/trained_model.pkl')
predictions = model.predict(events)

if predictions > threshold:
    alerts.append({...})
```

### Custom Workflow Analysis

Extend `WorkflowInterpreter`:

```python
class EnhancedInterpreter(WorkflowInterpreter):
    def detect_circular_paths(self):
        """Detect users going in circles."""
        cycles = []
        for workflow in self.detect_workflows():
            states = workflow['states']
            if len(set(states)) < len(states):
                cycles.append(workflow)
        return cycles
    
    def measure_efficiency(self):
        """Calculate optimal vs actual path length."""
        # Your metric here
        pass
```

---

## Best Practices

### Data Collection

- **Session Duration**: 30-60 seconds for focused tasks, 5-10 minutes for full workflows
- **Sample Size**: 10+ sessions for HMM training
- **Context**: Record what task users are performing
- **Consent**: Inform users of monitoring

### Threshold Tuning

1. **Start conservative** (high thresholds)
2. **Collect baseline** from normal usage
3. **Analyze false positives/negatives**
4. **Iterate based on precision/recall**

### Performance Optimization

- **Limit event buffer**: Process in chunks for long sessions
- **Cache aggressively**: Spatial caching prevents redundant queries
- **Async processing**: Run detection in background thread

### Troubleshooting

**No events captured:**
- Run as administrator
- Check permissions in Windows Security
- Verify pywinauto backend ("uia" vs "win32")

**OCR not working:**
- Install Tesseract correctly
- Check path in `cancel_detection.py`
- Test with `pytesseract.get_tesseract_version()`

**HMM poor performance:**
- Increase training data (20+ sessions)
- Adjust number of hidden states
- Refine observation bucketing

---

## API Reference

### GUI Sensor API

```python
# Programmatic session capture
from gui import run_sensor_session

data = run_sensor_session(duration=30)
# Returns: {'mouse_events': [...], 'keyboard_events': [...]}
```

### Interpreter API

```python
from interpreter import WorkflowInterpreter

interpreter = WorkflowInterpreter()
events = interpreter.load_events('session.json')
interpreter.build_graph_from_events(events)
interpreter.export_graph('output.json')
stats = interpreter.calculate_statistics()
```

### Detection API

```python
from pipeline import detect_with_heuristics
from heuristics.kb_mashing import SufferingDetector

# Heuristic detection
alerts = detect_with_heuristics(mouse_events)

# Keyboard mashing
detector = SufferingDetector(events)
result = detector.detect_keyboard_mashing(detector.keyboard_events)
```

---

## License & Credits

This documentation covers the Suffering Detection Pipeline system architecture, implementation, and usage patterns. For questions or contributions, refer to the project repository.

**Key Technologies:**
- pywinauto (GUI automation)
- pynput (Event capture)
- pytesseract (OCR)
- Win32 API (Windows integration)

**Version**: 1.0  
**Last Updated**: January 2025