"""Timed stage progress, also readable when a long experiment runs under nohup."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class StageProgress:
    def __init__(self, stage, total, interval=30.0, path=None):
        self.stage, self.total, self.interval = stage, total, interval
        self.path = Path(path) if path is not None else None
        self.completed, self.first_update, self.status = 0, True, "running"

    def __enter__(self):
        self.started = self.last_logged = time.perf_counter()
        self._emit()
        return self

    def update(self, completed, detail=""):
        self.completed = completed
        now = time.perf_counter()
        if self.first_update or completed == self.total or now - self.last_logged >= self.interval:
            self.first_update = False
            self.status = "completed" if completed == self.total else "running"
            self._emit(detail)

    def _emit(self, detail=""):
        now = time.perf_counter()
        elapsed = now - self.started
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.completed) / rate if rate > 0 else None
        record = {"stage": self.stage, "status": self.status, "completed": self.completed, "total": self.total,
                  "elapsed_seconds": elapsed, "items_per_second": rate, "eta_seconds": eta,
                  "updated_at": datetime.now(timezone.utc).isoformat(), "detail": detail}
        logger.info("%s %s %d/%d (%.1f%%) elapsed=%.1fs rate=%.3f/s ETA=%s%s",
                    self.stage, self.status, self.completed, self.total, 100 * self.completed / self.total,
                    elapsed, rate, "pending" if eta is None else "{:.1f}s".format(eta),
                    " " + detail if detail else "")
        if self.path is not None:
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        self.last_logged = now

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            self.status = "failed"
            self._emit(str(error))
            logger.error("%s failed", self.stage, exc_info=(error_type, error, traceback))
        elif self.status != "completed":
            self.completed, self.status = self.total, "completed"
            self._emit()
        return False
