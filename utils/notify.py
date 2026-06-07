# notify.py
"""
Desktop notification helper using plyer.
Gracefully falls back to console print if plyer is not installed.
"""

# BUG FIX #15: Wrap import in try/except so the entire client doesn't
# crash at startup if plyer is missing — falls back to print.
try:
    from plyer import notification as _notification
    _HAS_PLYER = True
except ImportError:
    _HAS_PLYER = False
    print("[WARN] notify.py: 'plyer' not installed. "
          "Desktop notifications disabled. Run: pip install plyer")


def send_alert(alert_message: str, timeout: int = 5) -> None:
    """
    Send a desktop notification.
    Falls back to printing if plyer is unavailable.

    Args:
        alert_message: The notification body text.
        timeout:       How many seconds to show the notification.
    """
    if _HAS_PLYER:
        try:
            _notification.notify(
                title="🌩️ Disaster Alert Detected!",
                message=alert_message,
                timeout=timeout,
            )
        except Exception as e:
            # Notification failed at runtime (e.g. no display server on headless machine)
            print(f"[WARN] Desktop notification failed: {e}")
            print(f"[ALERT] {alert_message}")
    else:
        # Graceful fallback
        print(f"[ALERT] {alert_message}")
