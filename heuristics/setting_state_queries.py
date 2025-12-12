import subprocess

# -------------------------
# WIFI STATE
# -------------------------
def get_wifi_state():
    """
    Returns True if WiFi is enabled, False if disabled.
    """
    try:
        out = subprocess.check_output(
            "netsh interface show interface", shell=True
        ).decode()

        return "Enabled" in out and "Wi-Fi" in out
    except Exception:
        return None


# -------------------------
# BLUETOOTH STATE
# -------------------------
def get_bluetooth_state():
    """
    Returns True if Bluetooth is ON, False if OFF.
    """
    try:
        cmd = 'powershell "(Get-PnpDevice -Class Bluetooth).Status"'
        out = subprocess.check_output(cmd, shell=True).decode()

        return "OK" in out
    except Exception:
        return None


# -------------------------
# AIRPLANE MODE STATE
# -------------------------
def get_airplane_mode_state():
    """
    Returns True if Airplane Mode is ON, False if OFF.
    """
    try:
        cmd = r'reg query "HKLM\System\CurrentControlSet\Control\RadioManagement\SystemRadioState"'
        out = subprocess.check_output(cmd, shell=True).decode()

        # 0x1 == airplane ON, 0x0 == OFF
        return "0x1" in out
    except Exception:
        return None
