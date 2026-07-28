"""
Functional composition utilities for error handling and control flow.

These wrappers compose at call sites - each state/agent picks what it needs.
No monolithic central handler, but also no scattered duplication.

Usage:
    from compass.core.compose import with_retry, with_fallback, with_transform, tap

    # Compose for Actor (retry 3x, log, return None on failure)
    ask = with_fallback(with_logging(with_retry(oracle.ask, 3)), None)

    # Transform result
    get_action = with_transform(oracle.ask, lambda r: r.get("action"))

    # Debug without breaking the chain
    ask = tap(oracle.ask, lambda r: print(f"Got: {r}"))

    # Try multiple approaches
    fetch = first_success(fetch_from_cache, fetch_from_api, fallback={})
"""

from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from compass.core.reasoning import debug

T = TypeVar("T")


def with_retry(
    fn: Callable[..., T],
    retries: int = 3,
    on_error: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[..., T]:
    """
    Retry fn on failure.

    Args:
        fn: Function to wrap
        retries: Number of attempts (default 3)
        on_error: Optional callback(exception, attempt) on each failure
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        last_error = None
        for attempt in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                if on_error:
                    on_error(e, attempt)
        raise last_error
    return wrapper


def with_logging(
    fn: Callable[..., T],
    name: Optional[str] = None,
) -> Callable[..., T]:
    """
    Log calls and errors.

    Args:
        fn: Function to wrap
        name: Optional name for logging (defaults to fn.__name__)
    """
    label = name or getattr(fn, "__name__", "fn")

    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        try:
            result = fn(*args, **kwargs)
            debug(f"{label}: ok")
            return result
        except Exception as e:
            debug(f"{label}: {e}")
            raise
    return wrapper


def with_fallback(
    fn: Callable[..., T],
    fallback: T,
) -> Callable[..., T]:
    """
    Return fallback value on any error.

    Args:
        fn: Function to wrap
        fallback: Value to return on failure
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        except Exception:
            return fallback
    return wrapper


def with_catch(
    fn: Callable[..., T],
    exc_type: type,
    handler: Callable[[Exception], T],
) -> Callable[..., T]:
    """
    Catch specific exception type and handle it.

    Args:
        fn: Function to wrap
        exc_type: Exception type to catch
        handler: Function that receives exception and returns fallback
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        except exc_type as e:
            return handler(e)
    return wrapper


def with_timeout(
    fn: Callable[..., T],
    seconds: float,
    fallback: Optional[T] = None,
) -> Callable[..., T]:
    """
    Timeout after seconds, return fallback.

    Note: Only works for I/O-bound operations. CPU-bound will not interrupt.

    Args:
        fn: Function to wrap
        seconds: Timeout in seconds
        fallback: Value to return on timeout
    """
    import signal

    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds}s")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return fn(*args, **kwargs)
        except TimeoutError:
            return fallback
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    return wrapper


def compose(*fns: Callable) -> Callable:
    """
    Compose functions right-to-left: compose(f, g, h)(x) = f(g(h(x)))

    Args:
        *fns: Functions to compose
    """
    def composed(x):
        for fn in reversed(fns):
            x = fn(x)
        return x
    return composed


def pipe(*fns: Callable) -> Callable:
    """
    Pipe functions left-to-right: pipe(f, g, h)(x) = h(g(f(x)))

    Args:
        *fns: Functions to pipe
    """
    def piped(x):
        for fn in fns:
            x = fn(x)
        return x
    return piped


def with_transform(
    fn: Callable[..., T],
    transform: Callable[[T], T],
) -> Callable[..., T]:
    """
    Transform the result of fn.

    Args:
        fn: Function to wrap
        transform: Function to apply to result
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        return transform(fn(*args, **kwargs))
    return wrapper


def tap(
    fn: Callable[..., T],
    side_effect: Callable[[T], None],
) -> Callable[..., T]:
    """
    Execute side effect on result without modifying it.

    Useful for debugging/logging in a compose chain.

    Args:
        fn: Function to wrap
        side_effect: Function to call with result (return value ignored)
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        result = fn(*args, **kwargs)
        side_effect(result)
        return result
    return wrapper


def first_success(
    *fns: Callable[..., T],
    fallback: Optional[T] = None,
) -> Callable[..., T]:
    """
    Try functions in order, return first successful result.

    Args:
        *fns: Functions to try
        fallback: Value to return if all fail
    """
    def wrapper(*args, **kwargs) -> T:
        for fn in fns:
            try:
                return fn(*args, **kwargs)
            except Exception:
                continue
        return fallback
    return wrapper
