"""In-memory rate limits for workspace clients (contract section 5).

A token bucket per client, and a failed-authentication counter per address: after 10 failures in
a minute, further *failures* get HTTP 429. Successes are never counted or refused. The address
table holds at most 1,024 entries. Both are process-local; the classes take an injectable clock.
"""

import time
from collections import OrderedDict, deque
from collections.abc import Callable

FAILURE_LIMIT = 10
FAILURE_WINDOW = 60.0
MAX_ADDRESSES = 1024


class TokenBuckets:
    """One token bucket per client id; ``clock`` returns monotonic seconds."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client_id: str, rate_per_min: int) -> bool:
        """Take one token from the client's bucket.

        Args:
            client_id: The verified client's id.
            rate_per_min: Bucket capacity and refill per minute.

        Returns:
            True if a token was available; False if the call must be refused.
        """
        if rate_per_min <= 0:
            return False
        now = self._clock()
        capacity = float(rate_per_min)
        tokens, last = self._buckets.get(client_id, (capacity, now))
        tokens = min(capacity, tokens + (now - last) * capacity / 60.0)
        if tokens < 1.0:
            self._buckets[client_id] = (tokens, now)
            return False
        self._buckets[client_id] = (tokens - 1.0, now)
        return True


class AuthFailureLimiter:
    """Failed authentications per address in a sliding minute, at most ``max_entries`` held."""

    def __init__(
        self, clock: Callable[[], float] = time.monotonic, max_entries: int = MAX_ADDRESSES
    ) -> None:
        self._clock = clock
        self._max = max_entries
        self._table: OrderedDict[str, deque] = OrderedDict()

    def __len__(self) -> int:
        return len(self._table)

    def _prune_stale(self, cutoff: float) -> None:
        for addr in [a for a, times in self._table.items() if times[-1] <= cutoff]:
            del self._table[addr]

    def record_failure(self, addr: str) -> bool:
        """Record one failed authentication from ``addr``.

        Call this only after a token failed to verify; never for a success.

        Args:
            addr: The remote address.

        Returns:
            True when this failure is past the limit and must get HTTP 429 instead of 401.
        """
        now = self._clock()
        cutoff = now - FAILURE_WINDOW
        times = self._table.get(addr)
        if times is None:
            if len(self._table) >= self._max:
                self._prune_stale(cutoff)
            # Reason: the cap bounds memory under an address flood; evicting the least recently
            # failing address costs the least, since a flood that large evades a per-address
            # limit anyway.
            while len(self._table) >= self._max:
                self._table.popitem(last=False)
            times = deque(maxlen=FAILURE_LIMIT + 1)
            self._table[addr] = times
        while times and times[0] <= cutoff:
            times.popleft()
        blocked = len(times) >= FAILURE_LIMIT
        times.append(now)
        self._table.move_to_end(addr)
        return blocked


_buckets = TokenBuckets()
_failures = AuthFailureLimiter()


def allow_call(client_id: str, rate_per_min: int) -> bool:
    """Take one call from the client's bucket; False means ``rate_limited``."""
    return _buckets.allow(client_id, rate_per_min)


def auth_failure_blocked(addr: str) -> bool:
    """Record a failed authentication; True (answer 429) after 10 in a minute from ``addr``.

    Call only on failure: a verified token skips this entirely, so it is never refused here.
    """
    return _failures.record_failure(addr)


def reset() -> None:
    """Clear both default limiters (for tests)."""
    global _buckets, _failures
    _buckets = TokenBuckets()
    _failures = AuthFailureLimiter()
