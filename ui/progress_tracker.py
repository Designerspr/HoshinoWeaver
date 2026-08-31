"""Qt adapter for reporting DAG progress without importing GUI modules."""

from typing import Optional

from PySide6.QtCore import QObject, Signal

from hoshicore.component.progress import DummyTracker


class QtSignalTracker(DummyTracker, QObject):
    """Aggregate DAG node progress and report it through Qt signals."""

    progress_updated = Signal(int, str)  # (percent, active_op_desc)
    finished = Signal()

    def __init__(self):
        QObject.__init__(self)
        DummyTracker.__init__(self)
        self._slots: set[str] = set()
        self._bars: dict[str, dict] = {}
        self._last_active_name: Optional[str] = None

    def pre_register(self, name: str) -> None:
        self._slots.add(name)
        self._emit()

    def create_bar(self, name, total, desc=None, unit="imgs"):
        self._slots.add(name)
        self._bars[name] = {"total": total, "progress": 0, "desc": desc or name}
        self._emit()

    def update(self, name, n=1):
        bar = self._bars.get(name)
        if bar:
            bar["progress"] = min(bar["progress"] + n, bar["total"])
            self._last_active_name = name
            self._emit()

    def close_bar(self, name):
        pass

    def close_all(self):
        self._slots.clear()
        self._bars.clear()
        self._last_active_name = None
        self.finished.emit()

    def reset_bar(self, name, total, desc=None):
        self.create_bar(name, total, desc)

    def _emit(self):
        n = len(self._slots) or 1
        done = sum(
            bar["progress"] / bar["total"] if bar["total"] > 0 else 0.0
            for bar in self._bars.values()
        )
        pct = min(int(done / n * 100), 99)
        active_bar = self._bars.get(self._last_active_name)
        if active_bar is None or active_bar["progress"] >= active_bar["total"]:
            active_bar = next(
                (
                    bar
                    for bar in self._bars.values()
                    if bar["progress"] < bar["total"]
                ),
                None,
            )
        active = ""
        if active_bar is not None:
            active_pct = (
                int(active_bar["progress"] / active_bar["total"] * 100)
                if active_bar["total"] > 0
                else 0
            )
            active = (
                f"正在执行：{active_bar['desc']}({active_pct}%) | "
                f"总进度 {pct}%"
            )
        self.progress_updated.emit(pct, active)
