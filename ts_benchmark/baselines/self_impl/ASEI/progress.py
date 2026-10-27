"""Wall-clock progress without CUDA calls from the reporting thread."""

import logging
import threading
import time


logger = logging.getLogger(__name__)
IMPLEMENTATION_VERSION = "asei-single-device-2"


class Progress:
    def __init__(self, label, total=None, interval=30):
        self.label, self.total, self.interval = label, total, interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.completed, self.current_phase, self.detail = 0, "setup", ""

    def __enter__(self):
        self.started = self.last_progress = self.last_log = time.monotonic()
        self._emit("start")
        if logger.isEnabledFor(logging.INFO):
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
        return self

    def phase(self, name):
        with self._lock:
            self.current_phase = name

    def update(self, completed, detail=""):
        now = time.monotonic()
        with self._lock:
            self.completed, self.detail, self.last_progress = completed, detail, now
            emit = completed == 1 or completed == self.total or now - self.last_log >= self.interval
        if emit:
            self._emit("update")

    def _emit(self, event):
        now = time.monotonic()
        with self._lock:
            completed, phase, detail = self.completed, self.current_phase, self.detail
            elapsed = now - self.started
            idle = now - self.last_progress
            self.last_log = now
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = ((self.total - completed) / rate if self.total is not None and rate > 0 else None)
        if event == "finished":
            phase, remaining = "completed", 0.0
        if self.total is None:
            logger.info("ASEI %s %s phase=%s elapsed_s=%.1f %s", self.label, event, phase, elapsed, detail)
            return
        logger.info(
            "ASEI %s %s progress=%d/%s phase=%s elapsed_s=%.1f since_progress_s=%.1f rate=%.3f/s eta_s=%s %s",
            self.label, event, completed, self.total if self.total is not None else "?",
            phase, elapsed, idle, rate, f"{max(0, remaining):.1f}" if remaining is not None else "unknown", detail,
        )

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self._emit("heartbeat")

    def __exit__(self, kind, value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._emit("failed" if kind is not None else "finished")

