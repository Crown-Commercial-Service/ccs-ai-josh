"""Session-level policy violation tracking, kept separate from chat history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable

LOCK_DURATION = timedelta(minutes=10)
RESET_AFTER = timedelta(minutes=15)
MAX_STRIKES = 3


@dataclass(frozen=True)
class ViolationState:
    strike_count: int = 0
    last_violation_time: datetime | None = None
    locked_until: datetime | None = None
    # The store supplies its clock when returning state. This keeps lock checks
    # deterministic in tests and avoids mixing an injected clock with wall time.
    _evaluated_at: datetime | None = field(default=None, repr=False, compare=False)

    @property
    def is_locked(self) -> bool:
        now = self._evaluated_at or datetime.now(timezone.utc)
        return bool(self.locked_until and self.locked_until > now)


class InMemoryViolationStore:
    """Thread-safe backend violation store for one application process.

    Records live only on the backend and are deliberately independent from both
    LangGraph/LLM messages and Flask's client-side session cookie. Deployments
    with multiple workers can replace this class with a Redis implementation
    exposing the same ``get`` and ``record_violation`` methods.
    """

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._records: dict[str, ViolationState] = {}
        self._lock = RLock()

    def _normalised(self, subject_id: str, now: datetime) -> ViolationState:
        state = self._records.get(subject_id, ViolationState())
        if (
            state.last_violation_time is not None
            and now - state.last_violation_time >= RESET_AFTER
        ):
            state = ViolationState()
            self._records.pop(subject_id, None)
        return replace(state, _evaluated_at=now)

    def get(self, subject_id: str) -> ViolationState:
        now = self._clock()
        with self._lock:
            return self._normalised(subject_id, now)

    def record_violation(self, subject_id: str) -> ViolationState:
        now = self._clock()
        with self._lock:
            current = self._normalised(subject_id, now)
            strikes = current.strike_count + 1
            locked_until = now + LOCK_DURATION if strikes >= MAX_STRIKES else None
            state = ViolationState(
                strike_count=strikes,
                last_violation_time=now,
                locked_until=locked_until,
                _evaluated_at=now,
            )
            self._records[subject_id] = state
            return state

    def snapshot(self, subject_id: str) -> dict[str, object]:
        """Return a diagnostic copy without exposing the mutable backing store."""
        snapshot = asdict(self.get(subject_id))
        snapshot.pop("_evaluated_at", None)
        return snapshot
