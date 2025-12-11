# heuristics/app_switching.py

import time
import psutil
import ctypes
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

APP_CATEGORIES = {
    'browsers': ['chrome.exe', 'firefox.exe', 'msedge.exe', 'brave.exe', 'opera.exe'],
    'email': ['outlook.exe', 'thunderbird.exe', 'hiri.exe', 'mail.exe'],
    'editors': ['code.exe', 'notepad++.exe', 'sublime_text.exe', 'pycharm.exe', 'idea64.exe'],
    'office': ['excel.exe', 'winword.exe', 'powerpnt.exe', 'calc.exe', 'writer.exe']
}


def get_active_process_window() -> Tuple[Optional[str], Optional[str]]:
    """Get current foreground app name and window title"""
    try:
        hWnd = ctypes.windll.user32.GetForegroundWindow()
        if not hWnd:
            return None, None

        pid = ctypes.c_ulong(0)
        ctypes.windll.user32.GetWindowThreadProcessId(hWnd, ctypes.byref(pid))

        process = psutil.Process(pid.value)
        name = process.name().lower()

        length = ctypes.windll.user32.GetWindowTextLengthW(hWnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hWnd, buf, length + 1)
        title = buf.value

        # Ignore system apps
        if name in ('explorer.exe', 'searchui.exe', 'python.exe', 'cmd.exe'):
            return None, None

        return name, title
    except Exception:
        return None, None


def get_category(app_name: str, window_title: str) -> Optional[str]:
    """Categorize app with context awareness"""
    for category, apps in APP_CATEGORIES.items():
        if app_name in apps:
            # Smart browser categorization
            if category == 'browsers' and window_title:
                title_lower = window_title.lower()
                if any(x in title_lower for x in ['outlook', 'mail', 'inbox', 'gmail']):
                    return 'email'
                if any(x in title_lower for x in ['word', 'excel', 'sheet', 'docs']):
                    return 'office'
            return category
    return None


def monitor_app_switches(duration_seconds: int, check_interval: float = 2.0,
                         switch_window: float = 30.0) -> List[Dict[str, Any]]:
    """
    Monitor app switching for a session duration.

    Args:
        duration_seconds: How long to monitor
        check_interval: Polling frequency in seconds
        switch_window: Time window to flag rapid switches

    Returns:
        List of app switch events with anomaly flags
    """
    switches = []
    last_app_name = None
    last_app_title = ""
    last_switch_time = time.time()

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        current_name, current_title = get_active_process_window()

        if current_name:
            # Detect actual switch
            if current_name != last_app_name or (current_name == last_app_name and current_title != last_app_title):

                if last_app_name:
                    curr_cat = get_category(current_name, current_title)
                    last_cat = get_category(last_app_name, last_app_title)
                    time_delta = time.time() - last_switch_time

                    # Record the switch
                    switch_event = {
                        'timestamp': datetime.now().isoformat(),
                        'from_app': last_app_name,
                        'to_app': current_name,
                        'from_category': last_cat,
                        'to_category': curr_cat,
                        'time_since_last_switch': round(time_delta, 2),
                        'is_anomalous': False
                    }

                    # Check for anomaly: same category, rapid switch
                    if (curr_cat and last_cat) and (curr_cat == last_cat) and (time_delta <= switch_window):
                        # Exclude same browser switches (tab changes)
                        is_same_browser = (current_name == last_app_name and curr_cat == 'browsers')

                        if not is_same_browser:
                            switch_event['is_anomalous'] = True
                            switch_event['anomaly_reason'] = 'rapid_category_switch'

                    switches.append(switch_event)

                last_app_name = current_name
                last_app_title = current_title
                last_switch_time = time.time()

        time.sleep(check_interval)

    return switches


def detect_app_switching_anomalies(switches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze app switches and generate alerts.

    Args:
        switches: List of switch events from monitor_app_switches()

    Returns:
        List of alert dicts
    """
    alerts = []

    # Count anomalies by category
    category_switches = defaultdict(list)

    for switch in switches:
        if switch.get('is_anomalous'):
            category = switch.get('to_category')
            category_switches[category].append(switch)

    # Generate alerts for each category with anomalies
    for category, cat_switches in category_switches.items():
        if len(cat_switches) >= 2:  # At least 2 rapid switches
            alerts.append({
                'type': 'rapid_app_switching',
                'severity': 'medium',
                'detected_by': 'heuristic',
                'confidence': min(1.0, len(cat_switches) / 5.0),
                'category': category,
                'switch_count': len(cat_switches),
                'switches': [
                    f"{s['from_app']} → {s['to_app']} ({s['time_since_last_switch']}s)"
                    for s in cat_switches
                ],
                'timestamp': cat_switches[0]['timestamp'],
                'description': f"User rapidly switched between {category} apps {len(cat_switches)} times"
            })

    return alerts