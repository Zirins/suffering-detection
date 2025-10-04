import sys
import time
import re
import argparse
import json
from datetime import datetime
from typing import Iterable, Set, Optional, Dict, Any
import pygetwindow as gw
import win32gui
from pywinauto import Desktop, Application
from pywinauto.base_wrapper import BaseWrapper
from pywinauto.findwindows import ElementNotFoundError
import pyautogui
import pytesseract
from PIL import Image
import io
from pynput import mouse
from pynput import keyboard
from pynput.mouse import Button
import win32api

"""
IMPORTANT PLEASE READ - 
Tesseract won't work unless you define a path for the executable (available here: https://github.com/UB-Mannheim/tesseract/wiki)
Simply running pip install tesseract won't work. Install this and point the tesseract_cmd variable to wherever it installs to as shown below.
"""
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")

# Global variables for mouse tracking
mouse_events = []           # every click event we capture
click_patterns = {}         # repeated clicks in same small areas
last_click_time = None      # time of previous clicks
last_click_pos = None       # position of previous click

# Global variables for keyboard tracking
keyboard_events = []

# Cache for element positions (for frozen app detection)
element_position_cache = {}

# TODO: get back to this later
session_meta = {
    "start_time": None,
    "end_time": None,
    "duration_sec": None,
}

# ----------------------------
# Action Sensor - Semantic Action Detection
# ----------------------------

def get_semantic_action(element_info: Dict[str, Any], button: Button) -> str:
    """
    Turns raw click + element info into a simple action text (semantic description)
    like: "User clicked File menu"
    """
    #
    elem_type = element_info.get('type', 'unknown')
    elem_name = element_info.get('name', '').strip()
    window = element_info.get('window', 'unknown window')
    button_str = str(button).replace('Button.', '').lower()

    # Handle empty names with better context
    if not elem_name or elem_name == '':
        # Right-click typically opens context menus
        if button_str == 'right':
            return f"User right-clicked to open context menu in '{window}'"

        # Panes without names - provide window context
        if elem_type == 'Pane':
            return f"User clicked in editor area of '{window}'"

        # Windows without specific names
        if elem_type == 'Window' or elem_type == 'window':
            return f"User clicked in '{window}'"

    # Handle different control types with appropriate semantic descriptions
    if elem_type == 'unknown' or elem_type == 'error':
        return f"User {button_str}-clicked at unknown location"

    # Menu items
    if 'menu' in elem_type.lower() or elem_type == 'MenuItem':
        if elem_name:
            return f"User clicked '{elem_name}' menu"
        return f"User clicked menu item"

    # Buttons
    if elem_type == 'Button' or elem_type == 'button':
        if elem_name:
            # Common button patterns
            if any(word in elem_name.lower() for word in ['save', 'ok', 'cancel', 'close', 'open', 'submit']):
                return f"User clicked '{elem_name}' button"
            return f"User clicked '{elem_name}' button"
        return f"User clicked button"

    # Text input fields
    if elem_type == 'Edit' or elem_type == 'edit':
        if elem_name:
            return f"User clicked in '{elem_name}' text field"
        return f"User clicked in text field"

    # Hyperlinks
    if elem_type == 'Hyperlink' or 'link' in elem_type.lower():
        if elem_name:
            return f"User clicked '{elem_name}' link"
        return f"User clicked hyperlink"

    # Checkboxes and radio buttons
    if elem_type == 'CheckBox':
        if elem_name:
            return f"User toggled '{elem_name}' checkbox"
        return f"User toggled checkbox"

    if elem_type == 'RadioButton':
        if elem_name:
            return f"User selected '{elem_name}' radio button"
        return f"User selected radio button"

    # Tabs
    if elem_type == 'TabItem':
        if elem_name:
            return f"User switched to '{elem_name}' tab"
        return f"User clicked tab"

    # List items
    if elem_type == 'ListItem' or elem_type == 'TreeItem':
        if elem_name:
            return f"User selected '{elem_name}' from list"
        return f"User clicked list item"

    # Window controls (minimize, maximize, close)
    if 'window' in elem_type.lower():
        if elem_name:
            return f"User clicked on '{elem_name}' window"
        return f"User clicked window at '{window}'"

    # Scroll bars
    if 'scroll' in elem_type.lower():
        return f"User interacted with scrollbar"

    # Generic fallback with as much info as possible
    if elem_name:
        return f"User clicked '{elem_name}' ({elem_type})"

    return f"User clicked {elem_type} in '{window}'"

def cache_element_position(x: int, y: int, element_info: Dict[str, Any]) -> None:
    """
    Save element info for each screen region.
    Used later if app is frozen and we can't query UI directly.
    """
    # Create a small region key (10x10 pixel grid)
    region_key = f"{x//10}_{y//10}"

    element_position_cache[region_key] = {
        "element_info": element_info,
        "exact_position": (x, y),
        "timestamp": datetime.now(),
        "access_count": element_position_cache.get(region_key, {}).get("access_count", 0) + 1
    }

    # Clean cache if it gets too large (keep last 1000 entries)
    if len(element_position_cache) > 1000:
        # Remove oldest entries
        sorted_cache = sorted(element_position_cache.items(),
                              key=lambda x: x[1]["timestamp"])
        for key, _ in sorted_cache[:200]:
            del element_position_cache[key]

def get_cached_element(x: int, y: int) -> Optional[Dict[str, Any]]:
    """
    Try to retrieve element info from cache (useful when app is frozen).
    Checks nearby regions for a match.
    """
    region_key = f"{x//10}_{y//10}"

    # Check exact region first
    if region_key in element_position_cache:
        return element_position_cache[region_key]["element_info"]

    # Check adjacent regions (within 20 pixels)
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            nearby_key = f"{(x//10)+dx}_{(y//10)+dy}"
            if nearby_key in element_position_cache:
                cached = element_position_cache[nearby_key]
                cached_x, cached_y = cached["exact_position"]
                distance = ((x - cached_x)**2 + (y - cached_y)**2)**0.5
                if distance < 20:  # Within 20 pixels
                    return cached["element_info"]

    return None

# ----------------------------
# Mouse Tracking
# ----------------------------

def get_element_at_position(x: int, y: int) -> Dict[str, Any]:
    """
    Find the UI element under the mouse pointer.
    If UI lookup fails (frozen app), fall back to cached info.
    """
    try:
        # Step 1: Try normal UI automation first
        hwnd = win32gui.WindowFromPoint((x, y))
        if not hwnd:
            # Fallback to cache
            cached = get_cached_element(x, y)
            if cached:
                return cached
            return {"type": "unknown", "window": "none", "element": None, "name": ""}

        window_title = win32gui.GetWindowText(hwnd)

        # Step 2: Try UI automation to get richer element info
        try:
            desktop = Desktop(backend="uia")
            element = desktop.from_point(x, y)
            if element:
                try:
                    control_type = element.element_info.control_type
                    name = element.element_info.name or ""
                    result = {
                        "type": control_type,
                        "name": name,
                        "window": window_title,
                        "element": str(element.element_info.class_name) if hasattr(element.element_info, 'class_name') else ""
                    }
                    # Cache this for future frozen app scenarios
                    cache_element_position(x, y, result)
                    return result
                except:
                    pass
        except:
            # If UI automation fails, try cache
            cached = get_cached_element(x, y)
            if cached:
                return cached

        # Fallback: we at least know the window title - ALWAYS CACHE THIS TOO
        result = {
            "type": "window",
            "window": window_title,
            "element": None,
            "name": ""
        }
        cache_element_position(x, y, result)
        return result

    except Exception as e:
        # Last resort: check cache
        cached = get_cached_element(x, y)
        if cached:
            return cached
        return {"type": "error", "window": "unknown", "element": None, "error": str(e), "name": ""}

def detect_click_pattern(x: int, y: int, timestamp: datetime) -> Dict[str, Any]:
    """Our heuristic to classify click behaviors"""
    global last_click_time, last_click_pos, click_patterns

    pattern_info = {
        "is_rage_click": False,
        "is_retry": False,
        "delay_from_last": 0,
        "click_count_in_area": 1
    }

    if last_click_time and last_click_pos:
        delay = (timestamp - last_click_time).total_seconds()
        pattern_info["delay_from_last"] = delay

        distance = ((x - last_click_pos[0])**2 + (y - last_click_pos[1])**2)**0.5

        if distance < 50:
            area_key = f"{x//50}_{y//50}"

            if area_key not in click_patterns:
                click_patterns[area_key] = {"count": 0, "last_time": timestamp, "positions": []}

            click_patterns[area_key]["count"] += 1
            click_patterns[area_key]["last_time"] = timestamp
            click_patterns[area_key]["positions"].append((x, y))

            pattern_info["click_count_in_area"] = click_patterns[area_key]["count"]

            # Rage clicking: 3+ clicks in same area within 3 seconds
            if (click_patterns[area_key]["count"] >= 3 and delay < 3.0):
                pattern_info["is_rage_click"] = True

            # Retry: 2+ clicks in same area within 1 second
            if (click_patterns[area_key]["count"] >= 2 and delay < 1.0):
                pattern_info["is_retry"] = True

    # Clean old patterns
    current_time = timestamp
    areas_to_remove = []
    for area_key, data in click_patterns.items():
        if (current_time - data["last_time"]).total_seconds() > 10:
            areas_to_remove.append(area_key)

    for area in areas_to_remove:
        del click_patterns[area]

    last_click_time = timestamp
    last_click_pos = (x, y)

    return pattern_info

def on_mouse_click(x: int, y: int, button: Button, pressed: bool) -> None:
    """Main mouse hook. We only record on press (not on release)"""
    if pressed:
        timestamp = datetime.now()

        # Get element info at click position
        element_info = get_element_at_position(x, y)

        # === ACTION SENSOR: Get semantic action description ===
        action_description = get_semantic_action(element_info, button)

        # Detect click patterns
        pattern_info = detect_click_pattern(x, y, timestamp)

        # Build the event payload
        event = {
            "timestamp": timestamp.isoformat(),
            "type": "mouse_click",
            "button": str(button),
            "position": {"x": x, "y": y},
            "element": element_info,
            "action_description": action_description,  # Added semantic action
            "patterns": pattern_info
        }

        mouse_events.append(event)

        # === Print ACTION SENSOR output (main requirement) ===
        print(f"\n{'='*60}")
        print(f"[{timestamp.strftime('%H:%M:%S')}] ACTION: {action_description}")
        print(f"{'='*60}")

        # Print pattern warnings if detected
        if pattern_info["is_rage_click"]:
            print(f"  ⚠️  RAGE CLICK detected - {pattern_info['click_count_in_area']} clicks in same area")
        elif pattern_info["is_retry"]:
            print(f"  ⚠️  RETRY detected - delay: {pattern_info['delay_from_last']:.2f}s")
        elif pattern_info["delay_from_last"] > 5:
            print(f"  ℹ️  Long hesitation ({pattern_info['delay_from_last']:.2f}s) before action")

        # Print technical details
        print(f"  Element type: {element_info['type']}")
        print(f"  Window: {element_info['window']}")
        if element_info.get('name'):
            print(f"  Element name: {element_info['name']}")
        print(f"  Position: ({x}, {y})")
        print()

        sys.stdout.flush()

def start_mouse_listener() -> mouse.Listener:
    """Start listening for mouse clicks in the background"""
    listener = mouse.Listener(on_click=on_mouse_click)
    listener.start()
    return listener

# ----------------------------
# Keyboard Tracking
# ----------------------------

def log_key_events(kind: str, key_str: str) -> None:
    event = {
        "timestamp": datetime.now().isoformat(),
        "type": f"keyboard_{kind}",
        "key": key_str,
    }

    keyboard_events.append(event)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {event['type']}: {key_str}")
    sys.stdout.flush()

def on_key_press(key, record_text: bool = True):
    try:
        key_str = key.char if (hasattr(key, "char") and key.char and record_text) else str(key)
    except Exception:
        key_str = str(key)
    log_key_events("press", key_str)

def on_key_release(key, record_text: bool = True):
    try:
        key_str = key.char if (hasattr(key, "char") and key.char and record_text) else str(key)
    except Exception:
        key_str = str(key)
    log_key_events("release", key_str)

def start_keyboard_listener(record_text: bool = True) -> keyboard.Listener:
    listener = keyboard.Listener(
        on_press=lambda k: on_key_press(k, record_text=record_text),
        on_release=lambda k: on_key_release(k, record_text=record_text),
    )
    listener.start()
    return listener

# ----------------------------
# Logging Helpers
# ----------------------------

def log_event_with_timestamp(event_type: str, data: Dict[str, Any]) -> None:
    """Print any event with a current timestamp (simple console logger)"""
    timestamp = datetime.now()
    event = {
        "timestamp": timestamp.isoformat(),
        "type": event_type,
        "data": data
    }
    print(f"[{timestamp.strftime('%H:%M:%S')}] {event_type}: {data}")

def save_events_to_file(filename: str = "gui_events.json") -> None:
    """Write ALL captured mouse and keyboard events to disk as JSON"""
    try:
        payload = {
            "session": session_meta,
            "mouse_events": mouse_events,
            "keyboard_events": keyboard_events,
            "element_cache_size": len(element_position_cache)
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(
            f"\nSaved {len(mouse_events)} mouse events and {len(keyboard_events)} keyboard events to {filename}"
        )
        print(f"Element position cache size: {len(element_position_cache)} regions")
    except Exception as e:
        print(f"Error saving events: {e}")

# ----------------------------
# Original GUI.py functions
# ----------------------------

def normalize_text(s: str) -> str:
    """Normalize whitespace and trim."""
    s = s.replace("\u00a0", " ")
    s = s.replace("\u200b", "")
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip()

def filter_lines(lines: Iterable[str], min_len: int = 2) -> Iterable[str]:
    """Clean, dedupe, and filter tiny/noisy lines."""
    seen: Set[str] = set()
    for line in lines:
        if not line:
            continue
        line = normalize_text(line)
        if len(line) < min_len:
            continue
        if line in seen:
            continue
        seen.add(line)
        yield line

def safe_get_textpattern(wrapper: BaseWrapper) -> Optional[str]:
    """Try to read the whole document via UIA TextPattern if available."""
    try:
        rng = wrapper.iface_text.DocumentRange
        return rng.GetText(-1)
    except Exception:
        return None

# ----------------------------
# Foreground tracking
# ----------------------------

def get_foreground_hwnd() -> Optional[int]:
    """Return the current foreground window handle (HWND), or None if unavailable."""
    try:
        hwnd = win32gui.GetForegroundWindow()
        return hwnd if hwnd else None
    except Exception:
        return None

def connect_wrapper_to_hwnd(hwnd: int) -> Optional[BaseWrapper]:
    """Build a pywinauto wrapper for the given HWND."""
    try:
        desktop = Desktop(backend="uia")
        return desktop.window(handle=hwnd)
    except (ElementNotFoundError, Exception):
        return None

def extract_accessible_text(root: BaseWrapper, max_controls: int = 5000, record_interactives: bool = False) -> Iterable[str]:
    """Collect a broad set of human-readable strings from the active window subtree."""
    doc_grabbed = False
    try:
        for doc in root.descendants(control_type="Document"):
            big = safe_get_textpattern(doc)
            if big:
                for line in big.splitlines():
                    yield line
                doc_grabbed = True
    except Exception:
        pass

    try:
        for txt in root.descendants(control_type="Text")[:max_controls]:
            try:
                s = txt.window_text()
                if not s:
                    s = txt.element_info.name
                if s:
                    yield s
            except Exception:
                continue
    except Exception:
        pass

    try:
        for edit in root.descendants(control_type="Edit")[:max_controls]:
            try:
                val = edit.get_value()
            except Exception:
                try:
                    val = edit.window_text()
                except Exception:
                    val = None
            if val:
                yield val
    except Exception:
        pass

    if record_interactives:
        interactive_types = [
            "Button", "MenuItem", "Hyperlink", "TabItem",
            "CheckBox", "RadioButton", "ListItem", "TreeItem"
        ]
        try:
            for ctype in interactive_types:
                for w in root.descendants(control_type=ctype)[:max_controls]:
                    try:
                        nm = w.element_info.name or w.window_text()
                        if nm:
                            yield nm
                    except Exception:
                        continue
        except Exception:
            pass

    if not doc_grabbed:
        try:
            for w in root.descendants()[:max_controls]:
                try:
                    nm = w.element_info.name
                    if nm:
                        yield nm
                except Exception:
                    continue
        except Exception:
            pass

# ----------------------------
# OCR Fallback Option
# ----------------------------

def screenshot_and_ocr(window_title: str) -> str:
    """Takes a screenshot, performs OCR, and prints the extracted text."""
    try:
        try:
            target_window = gw.getWindowsWithTitle(window_title)[0]
        except IndexError:
            print(f"Window '{window_title}' not found.")
            return ""

        left, top, width, height = target_window.left, target_window.top, target_window.width, target_window.height
        print("OCR Triggered")

        screenshot = pyautogui.screenshot(region=(left, top, width, height))

        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img = Image.open(io.BytesIO(img_byte_arr))

        text = pytesseract.image_to_string(img)
        print(text)
        return text

    except Exception as e:
        print(f"Error in OCR: {e}")
        return ""

# ----------------------------
# Formatter / Printer
# ----------------------------

def print_snapshot(title: str, texts: Iterable[str], min_len: int, include_banner: bool) -> None:
    """Print a snapshot with timestamp."""
    timestamp = datetime.now()
    lines = list(filter_lines(texts, min_len=min_len))

    log_event_with_timestamp("window_focus", {
        "window_title": title,
        "text_lines_count": len(lines)
    })

    if include_banner:
        print("=" * 80)
        print(f"[{timestamp.strftime('%H:%M:%S')}] [ACTIVE]: {title}")
        print("-" * 80)
    if lines:
        print("\n".join(lines))
    else:
        print("(no accessible text found)")
    if include_banner:
        print("=" * 80)
    sys.stdout.flush()

def active_window_title(hwnd: int) -> str:
    try:
        return win32gui.GetWindowText(hwnd) or f"HWND={hwnd}"
    except Exception:
        return f"HWND={hwnd}"

# ----------------------------
# Main loop
# ----------------------------

def run_follow_foreground(poll_sec: float, min_len: int, print_banner: bool) -> None:
    """Poll the foreground window and print accessible text whenever focus changes."""
    print("="*80)
    print("ACTION SENSOR - GUI Monitoring System")
    print("="*80)
    print("Monitoring user actions with semantic interpretation...")
    print("Actions will be logged as: 'User clicked [element name]'")
    print("Element positions cached for frozen app detection.")
    print("\nPress Ctrl+C to stop.\n")
    print("="*80 + "\n")

    # Start mouse listener and keyboard listener
    mouse_listener = start_mouse_listener()
    keyboard_listener = start_keyboard_listener(record_text=True)

    last_hwnd = None
    ocr_programs = ["Mozilla Firefox", "Google Chrome", "Microsoft Edge", "GUI.py"]
    last_snapshot_key = None
    program_timer = 0

    try:
        while True:
            try:
                hwnd = get_foreground_hwnd()
                if (hwnd and hwnd != last_hwnd) or (hwnd == last_hwnd and program_timer > 20):
                    last_hwnd = hwnd
                    title = active_window_title(hwnd)
                    wrapper = connect_wrapper_to_hwnd(hwnd)

                    if not wrapper:
                        print_snapshot(title, [], min_len, include_banner=print_banner)
                        time.sleep(poll_sec)
                        continue

                    ocr_extracted = False
                    for program in ocr_programs:
                        if program in title:
                            ocr_extracted = True
                            raw_texts = [screenshot_and_ocr(title)]
                            break

                    if not ocr_extracted:
                        raw_texts = list(extract_accessible_text(wrapper))

                    snapshot_key = (title, tuple(sorted(filter_lines(raw_texts, min_len=min_len))))
                    if (snapshot_key != last_snapshot_key) or (program_timer > 10):
                        last_snapshot_key = snapshot_key
                        print_snapshot(title, raw_texts, min_len, include_banner=print_banner)

                    program_timer = 0
                else:
                    program_timer += 1
                time.sleep(poll_sec)

            except KeyboardInterrupt:
                print("\n" + "="*80)
                print("Stopping ACTION SENSOR...")
                print("="*80)
                print(f"\nSession Statistics:")
                print(f"  Total mouse events captured: {len(mouse_events)}")
                print(f"  Total keyboard events captured: {len(keyboard_events)}")
                print(f"  Element positions cached: {len(element_position_cache)}")

                # Save events before exiting
                save_events_to_file()

                # Stop listeners
                mouse_listener.stop()
                keyboard_listener.stop()
                print("\nShutdown complete.")
                break
            except Exception as e:
                print(f"[warn] {type(e).__name__}: {e}")
                time.sleep(poll_sec)

    except Exception as e:
        print(f"Fatal error: {e}")
        mouse_listener.stop()
        keyboard_listener.stop()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Action Sensor: Detects semantic user actions (e.g., 'User clicked File menu')"
    )
    p.add_argument("--interval", type=float, default=0.5,
                   help="Polling interval in seconds (default: 0.5)")
    p.add_argument("--min-line-length", type=int, default=2,
                   help="Minimum length of a line to print (default: 2)")
    p.add_argument("--no-banner", action="store_true",
                   help="Do not print banner/title separators for each snapshot")
    p.add_argument("--interactive", type=bool, default=False,
                   help="Include interactive elements such as buttons and hyperlinks")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    run_follow_foreground(
        poll_sec=args.interval,
        min_len=args.min_line_length,
        print_banner=not args.no_banner,
    )

if __name__ == "__main__":
    main()