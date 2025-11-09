import time
from datetime import datetime
from typing import Dict, Any, Optional
import win32gui
from pywinauto import Desktop
from pynput import mouse, keyboard
from pynput.mouse import Button

# ==============================================================
# GLOBAL STATE
# ==============================================================

mouse_events = []
keyboard_events = []
element_position_cache = {}
click_patterns = {}
last_click_time = None
last_click_pos = None


# ==============================================================
# SEMANTIC ACTION DETECTION
# ==============================================================

def get_semantic_action(element_info: Dict[str, Any], button: Button) -> str:
    """
    Turns raw click + element info into a simple action text (semantic description)
    like: "User clicked File menu"
    """
    elem_type = element_info.get("type", "unknown")
    elem_name = element_info.get("name", "").strip()
    window = element_info.get("window", "unknown window")
    button_str = str(button).replace("Button.", "").lower()

    if "listitem" in elem_type.lower() or "treeitem" in elem_type.lower():
        return f"User selected '{elem_name}' from list" if elem_name else "User clicked list item"

    if "scroll" in elem_type.lower():
        return "User interacted with scrollbar"

    if elem_type.lower() == "pane" and not elem_name:
        return f"User clicked in editor area of '{window}'"

    if not elem_name:
        if button_str == "right":
            return f"User right-clicked in '{window}'"
        return f"User clicked in '{window}'"

    if "menu" in elem_type.lower():
        return f"User clicked '{elem_name}' menu"
    if "button" in elem_type.lower():
        return f"User clicked '{elem_name}' button"
    if "tab" in elem_type.lower():
        return f"User switched to '{elem_name}' tab"
    if "checkbox" in elem_type.lower():
        return f"User toggled '{elem_name}' checkbox"
    if "radio" in elem_type.lower():
        return f"User selected '{elem_name}' option"
    if "edit" in elem_type.lower() or "text" in elem_type.lower():
        return f"User interacted with text field '{elem_name}'"
    if "link" in elem_type.lower():
        return f"User clicked link '{elem_name}'"

    return f"User {button_str}-clicked '{elem_name}' in '{window}'"


# ==============================================================
# ELEMENT CACHING
# ==============================================================

def cache_element_position(x: int, y: int, element_info: Dict[str, Any]) -> None:
    """
    Save element info for each screen region.
    Used later if app is frozen and we can't query UI directly.
    """
    """Cache element info by screen region for fallback if UI query fails."""
    region_key = f"{x//10}_{y//10}"
    element_position_cache[region_key] = {
        "element_info": element_info,
        "timestamp": datetime.now()
    }
    # Keep cache from growing endlessly
    if len(element_position_cache) > 1000:
        oldest_keys = sorted(
            element_position_cache.items(),
            key=lambda x: x[1]["timestamp"]
        )[:200]
        for k, _ in oldest_keys:
            del element_position_cache[k]


def get_cached_element(x: int, y: int) -> Optional[Dict[str, Any]]:
    """Retrieve element info from nearby cached positions."""
    region_key = f"{x//10}_{y//10}"
    if region_key in element_position_cache:
        return element_position_cache[region_key]["element_info"]

    # Search nearby regions
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            k = f"{(x//10)+dx}_{(y//10)+dy}"
            if k in element_position_cache:
                return element_position_cache[k]["element_info"]
    return None


# ==============================================================
# ELEMENT DETECTION
# ==============================================================

def get_element_at_position(x: int, y: int) -> Dict[str, Any]:
    """Detect UI element under mouse position using Win32 + Pywinauto."""
    try:
        hwnd = win32gui.WindowFromPoint((x, y))
        if not hwnd:
            cached = get_cached_element(x, y)
            return cached or {"type": "unknown", "window": "none", "name": ""}

        window_title = win32gui.GetWindowText(hwnd)
        desktop = Desktop(backend="uia")

        try:
            element = desktop.from_point(x, y)
            control_type = element.element_info.control_type
            name = element.element_info.name or ""
            result = {"type": control_type, "name": name, "window": window_title}
            cache_element_position(x, y, result)
            return result
        except Exception:
            cached = get_cached_element(x, y)
            if cached:
                return cached

        return {"type": "window", "name": "", "window": window_title}
    except Exception as e:
        cached = get_cached_element(x, y)
        if cached:
            return cached
        return {"type": "error", "window": "unknown", "name": "", "error": str(e)}


# ==============================================================
# CLICK PATTERN DETECTION
# ==============================================================

def detect_click_pattern(x: int, y: int, timestamp: datetime) -> Dict[str, Any]:
    """Detect rage clicks, retries, or hesitation patterns."""
    global last_click_time, last_click_pos, click_patterns

    pattern = {"is_rage_click": False, "is_retry": False, "delay": 0.0, "count": 1}

    if last_click_time and last_click_pos:
        delay = (timestamp - last_click_time).total_seconds()
        pattern["delay"] = delay

        distance = ((x - last_click_pos[0]) ** 2 + (y - last_click_pos[1]) ** 2) ** 0.5
        if distance < 50:
            area = f"{x//50}_{y//50}"
            if area not in click_patterns:
                click_patterns[area] = {"count": 0, "last": timestamp}
            click_patterns[area]["count"] += 1
            click_patterns[area]["last"] = timestamp
            pattern["count"] = click_patterns[area]["count"]

            if click_patterns[area]["count"] >= 3 and delay < 3.0:
                pattern["is_rage_click"] = True
            elif click_patterns[area]["count"] >= 2 and delay < 1.0:
                pattern["is_retry"] = True

    last_click_time = timestamp
    last_click_pos = (x, y)
    return pattern


# ==============================================================
# MOUSE EVENT HANDLING
# ==============================================================

def on_mouse_click(x: int, y: int, button: Button, pressed: bool) -> None:
    """Callback for mouse clicks; capture and label events."""
    if not pressed:
        return

    timestamp = datetime.now()
    element_info = get_element_at_position(x, y)
    pattern_info = detect_click_pattern(x, y, timestamp)
    action_desc = get_semantic_action(element_info, button)

    event = {
        "timestamp": timestamp.isoformat(),
        "type": "mouse_click",
        "button": str(button),
        "position": {"x": x, "y": y},
        "element": element_info,
        "action_description": action_desc,
        "patterns": pattern_info
    }

    mouse_events.append(event)
    print(f"[{timestamp.strftime('%H:%M:%S')}] ACTION: {action_desc}")


def start_mouse_listener() -> mouse.Listener:
    """Start the mouse listener thread."""
    listener = mouse.Listener(on_click=on_mouse_click)
    listener.start()
    return listener


# ==============================================================
# KEYBOARD EVENT HANDLING
# ==============================================================

def on_key_press(key) -> None:
    """Record key press events."""
    try:
        key_str = key.char if hasattr(key, "char") and key.char else str(key)
    except Exception:
        key_str = str(key)
    keyboard_events.append({
        "timestamp": datetime.now().isoformat(),
        "type": "keyboard_press",
        "key": key_str
    })


def on_key_release(key) -> None:
    """Record key release events."""
    try:
        key_str = key.char if hasattr(key, "char") and key.char else str(key)
    except Exception:
        key_str = str(key)
    keyboard_events.append({
        "timestamp": datetime.now().isoformat(),
        "type": "keyboard_release",
        "key": key_str
    })


def start_keyboard_listener() -> keyboard.Listener:
    """Start the keyboard listener thread."""
    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.start()
    return listener


# ==============================================================
# RUNNER WRAPPER (For Pipeline Integration)
# ==============================================================

def run_sensor_session(duration: int = 30) -> Dict[str, Any]:
    """
    Run a full mouse + keyboard monitoring session for the given duration.
    Returns: { 'mouse_events': [...], 'keyboard_events': [...] }
    """
    print(f"\n🎯 Starting GUI action capture for {duration}s...\n")

    m_listener = start_mouse_listener()
    k_listener = start_keyboard_listener()

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("⏹️  Interrupted manually.")

    m_listener.stop()
    k_listener.stop()

    print(f"\n✅ Capture complete: {len(mouse_events)} mouse events, {len(keyboard_events)} keyboard events.\n")

    return {
        "mouse_events": list(mouse_events),
        "keyboard_events": list(keyboard_events)
    }


# ==============================================================
# DEBUG MODE
# ==============================================================

if __name__ == "__main__":
    data = run_sensor_session(duration=10)
    print(data)
