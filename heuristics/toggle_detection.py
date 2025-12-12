import time
import usb.core
import hashlib


# Rolling history buffers
hardware_history = []
setting_history = {}


# ----------------------------
# HELPER: GET USB SIGNATURE
# ----------------------------
def get_usb_signature():
    devices = usb.core.find(find_all=True)
    sigs = []
    for d in devices:
        sigs.append(f"{d.idVendor}:{d.idProduct}")
    sigs.sort()

    sig_hash = hashlib.sha256(",".join(sigs).encode()).hexdigest()
    return sig_hash, sigs


# ---------------------------------------------------------
# DETECT HARDWARE TOGGLING (ABAB pattern)
# ---------------------------------------------------------
def detect_hardware_toggling(max_window=5, max_age=30):
    global hardware_history

    sig_hash, dev_list = get_usb_signature()

    # Add if new state
    if not hardware_history or hardware_history[0][0] != sig_hash:
        hardware_history.insert(0, (sig_hash, time.time(), dev_list))

    # Trim by time + length
    hardware_history = [
        h for h in hardware_history if time.time() - h[1] < max_age
    ][:max_window]

    if len(hardware_history) < 4:
        return None

    sigs = [h[0] for h in hardware_history]
    lists = [h[2] for h in hardware_history]

    # Single device difference?
    diff_now = set(lists[0]) ^ set(lists[1])
    if len(diff_now) != 1:
        return None

    toggled_device = list(diff_now)[0]

    # ABAB detection
    if sigs[0] == sigs[2] and sigs[1] == sigs[3] and sigs[0] != sigs[1]:
        return {
            "type": "hardware_toggling",
            "severity": "medium",
            "detected_by": "heuristic",
            "device": toggled_device,
            "description": f"Device {toggled_device} repeatedly connected/disconnected (ABAB pattern)."
        }

    return None


# ----------------------------
# DETECT SETTING TOGGLING FOR ONE SETTING
# ----------------------------
def detect_setting_toggling(setting_name, current_value, max_window=5, max_age=20):
    global setting_history

    # Initialize list for setting if needed
    if setting_name not in setting_history:
        setting_history[setting_name] = []

    history = setting_history[setting_name]

    val_hash = hashlib.sha256(str(current_value).encode()).hexdigest()

    # Add new entry only if changed
    if not history or history[0][0] != val_hash:
        history.insert(0, (val_hash, time.time(), current_value))

    # Trim
    history = [
        h for h in history if time.time() - h[1] < max_age
    ][:max_window]

    setting_history[setting_name] = history

    if len(history) < 4:
        return None

    sigs = [h[0] for h in history]
    vals = [h[2] for h in history]

    # ABAB pattern = toggling
    if sigs[0] == sigs[2] and sigs[1] == sigs[3] and vals[0] != vals[1]:
        return {
            "type": "setting_toggling",
            "severity": "medium",
            "detected_by": "heuristic",
            "setting": setting_name,
            "previous_value": vals[1],
            "current_value": vals[0],
            "description": f"User repeatedly toggled '{setting_name}' (ABAB pattern)."
        }

    return None