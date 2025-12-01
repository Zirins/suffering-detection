# New file: ocr_detection.py

import pytesseract
from PIL import ImageGrab, ImageOps
import time
from collections import defaultdict
from datetime import datetime

# Try to find Tesseract in default path, else handle with grace
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except:
    pass  # Will use system PATH

CANCEL_WORDS = {"cancel", "close", "exit", "dismiss", "abort", "quit"}

class CancellationDetector:
    def __init__(self):
        self.cancel_clicks = []  # List of timestamps

    def check_click_for_cancellation(self, x, y, timestamp):
        """
        OCR around click position to see if it's on cancel/close button.
        Returns alert dict if cancellation pattern detected, None otherwise.
        """
        # Crop around click
        W, H = 120, 40
        left = max(0, int(x - W//2))
        top = max(0, int(y - H//2))

        try:
            crop = ImageGrab.grab(bbox=(left, top, left + W, top + H)).convert("L")
            crop = ImageOps.autocontrast(crop).resize((crop.width*2, crop.height*2))

            text = pytesseract.image_to_string(
                crop,
                config="--psm 8"  # Single word
            ).strip().lower()

            # Check if any cancel word found
            if any(word in text for word in CANCEL_WORDS):
                self.cancel_clicks.append(timestamp)

                # Keep only last 15 seconds
                cutoff = datetime.now().timestamp() - 15
                self.cancel_clicks = [t for t in self.cancel_clicks
                                      if datetime.fromisoformat(t).timestamp() > cutoff]

                # If 3+ in 15s, return alert
                if len(self.cancel_clicks) >= 3:
                    return {
                        'type': 'repeated_cancellation',
                        'severity': 'high',
                        'detected_by': 'ocr',
                        'confidence': 0.85,
                        'cancel_count': len(self.cancel_clicks),
                        'timestamp': timestamp,
                        'detected_text': text,
                        'description': f"User clicked cancel/close {len(self.cancel_clicks)} times in 15s"
                    }
        except Exception as e:
            # OCR failed, that's ok
            pass

        return None

    def analyze_events(self, mouse_events):
        """
        Check all mouse events for cancellation patterns.
        Returns list of alerts.
        """
        alerts = []

        for event in mouse_events:
            if event.get('type') != 'mouse_click':
                continue

            pos = event.get('position', {})
            x, y = pos.get('x', 0), pos.get('y', 0)
            timestamp = event.get('timestamp')

            alert = self.check_click_for_cancellation(x, y, timestamp)
            if alert:
                alerts.append(alert)

        return alerts