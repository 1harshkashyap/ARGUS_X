"""
ARGUS-X — Request Context
Provides a per-request context variable for correlation IDs.

Uses Python contextvars.ContextVar which is:
  - Async-safe: each coroutine inherits a copy of the parent context
  - Concurrent-safe: different requests have isolated contexts
  - Automatically propagated to awaited functions and create_task()
  - Zero overhead when not set (default is empty string)

Usage:
  # In chat handler (start of request):
  from utils.context import set_request_id, new_request_id
  token = set_request_id(new_request_id())

  # In logger formatter (automatic — no manual calls needed):
  from utils.context import get_request_id
  req_id = get_request_id()  # Returns '' if not in a request context
"""

import contextvars
import uuid

# Module-level ContextVar — one per process, many values (one per context)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def new_request_id() -> str:
    """Generate a new 8-character uppercase hex request ID."""
    return uuid.uuid4().hex[:8].upper()


def set_request_id(req_id: str) -> contextvars.Token:
    """
    Set the request ID for the current async context.
    Returns a Token that can be used to reset to the previous value.
    Call at the very start of each request handler.
    """
    return request_id_var.set(req_id)


def get_request_id() -> str:
    """
    Get the current request ID.
    Returns '' if called outside a request context (startup, background
    system tasks, battle engine ticks — these correctly show no ID).
    """
    return request_id_var.get()


def reset_request_id(token: contextvars.Token) -> None:
    """
    Reset the context var to its previous value using the token
    returned by set_request_id(). Call in a finally block.
    """
    request_id_var.reset(token)
