from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any, Callable, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class ServiceMetrics:
    total_requests: int = 0
    forecast_requests: int = 0
    prediction_requests: int = 0
    error_requests: int = 0
    total_latency_ms: float = 0.0
    last_updated: str = ""
    last_error: str = ""
    by_route: dict[str, int] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        average_latency = self.total_latency_ms / self.total_requests if self.total_requests else 0.0
        return {
            "total_requests": self.total_requests,
            "forecast_requests": self.forecast_requests,
            "prediction_requests": self.prediction_requests,
            "error_requests": self.error_requests,
            "average_latency_ms": average_latency,
            "last_updated": self.last_updated,
            "last_error": self.last_error,
            "by_route": dict(self.by_route),
            "by_status": dict(self.by_status),
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state = ServiceMetrics()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def record_request(self, route: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            self._state.total_requests += 1
            self._state.total_latency_ms += duration_ms
            self._state.by_route[route] = self._state.by_route.get(route, 0) + 1
            status_key = str(status_code)
            self._state.by_status[status_key] = self._state.by_status.get(status_key, 0) + 1

    def record_forecast(self) -> None:
        with self._lock:
            self._state.forecast_requests += 1

    def record_prediction(self) -> None:
        with self._lock:
            self._state.prediction_requests += 1

    def record_error(self, error_message: str) -> None:
        with self._lock:
            self._state.error_requests += 1
            self._state.last_error = error_message

    def mark_updated(self, route: str) -> None:
        with self._lock:
            from datetime import datetime, timezone

            self._state.last_updated = datetime.now(timezone.utc).isoformat()
            self._state.by_route[route] = self._state.by_route.get(route, 0)


REGISTRY = MetricsRegistry()
