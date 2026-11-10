"""Safe Azure OpenAI filtering responses and session-level strike lockouts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable

from flask import jsonify, render_template, request, session
from markupsafe import Markup, escape
from openai import BadRequestError

from src.security_tracking import InMemoryViolationStore, MAX_STRIKES

BLOCKED_MESSAGE = "I cannot fulfill this request as it violates safety guidelines."
LOCKED_MESSAGE = (
    "Access temporarily suspended due to repeated policy violations. "
    "Please try again later."
)
LOCKOUT_STARTED_MESSAGE = (
    "Multiple policy violations detected. You have been locked out for 10 minutes."
)
_GENERIC_MESSAGE = "The request could not be processed by the AI service. Please try rephrasing it."

# Backend-only state: it is never copied into LangGraph messages or Flask's cookie.
violation_store = InMemoryViolationStore()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def is_jailbreak_filter_error(exc: BaseException) -> bool:
    """Inspect supported Azure error shapes without exposing their contents."""
    body = _as_mapping(getattr(exc, "body", None))
    error = _as_mapping(body.get("error"))
    container = error or body
    innererror = _as_mapping(container.get("innererror"))
    filter_result = _as_mapping(innererror.get("content_filter_result"))
    jailbreak = _as_mapping(filter_result.get("jailbreak"))
    return jailbreak.get("detected") is True


def blocked_payload(message: str = BLOCKED_MESSAGE) -> dict[str, str]:
    return {"status": "blocked", "message": message}


def locked_payload(message: str = LOCKED_MESSAGE) -> dict[str, str]:
    return {"status": "locked", "message": message}


def browser_message(message: str, warning_label: str = "Safety notice") -> Markup:
    """Build trusted UI chrome while escaping variable text."""
    return Markup(
        '<div class="app-inline-warning" role="status" aria-label="Safety warning">'
        '<span class="app-inline-warning__icon" aria-hidden="true">!</span>'
        f"<span>{escape(warning_label)}</span></div><p>{escape(message)}</p>"
    )


def assistant_warning_item(message: str, warning_label: str = "Safety notice") -> dict[str, Any]:
    """Create a display-only assistant bubble; this is never LLM history."""
    return {
        "role": "assistant",
        "content": browser_message(message, warning_label),
        "raw_content": None,
        "sources": "",
        "full_table": None,
    }


def _subject_id() -> str:
    """Get a stable session subject without logging or exposing it."""
    subject = session.get("user_id")
    if subject:
        return str(subject)
    import uuid

    subject = str(uuid.uuid4())
    session["user_id"] = subject
    return subject


def _safe_history(history_provider: Callable[[], list[dict[str, Any]]] | None):
    try:
        return list(history_provider() if history_provider else [])
    except Exception:
        return []


def _response(message: str, status: str, http_status: int, history_provider=None):
    if request.is_json or request.path.startswith("/api/"):
        return jsonify({"status": status, "message": message}), http_status
    history = _safe_history(history_provider)
    history.append(
        assistant_warning_item(
            message, "Access temporarily suspended" if status == "locked" else "Safety notice"
        )
    )
    return render_template("index.html", messages=history, sanitisation_error=None), http_status


def register_openai_error_handler(
    app: Any,
    *,
    history_provider: Callable[[], list[dict[str, Any]]] | None = None,
    failed_turn_cleanup: Callable[[], None] | None = None,
    logger: logging.Logger | None = None,
    store: InMemoryViolationStore | None = None,
) -> None:
    """Install pre-request lock checking and sanitised OpenAI error handling.

    ``failed_turn_cleanup`` lets the application roll its graph back to the
    message IDs captured before the request, ensuring a rejected prompt and any
    intermediate SQL/tool messages cannot poison later LLM context.
    """
    log = logger or logging.getLogger(__name__)
    backend = store or violation_store

    @app.before_request
    def reject_locked_chat_requests():
        # Feedback/static/navigation remain usable. No LLM/Vanna code runs when
        # a locked user submits to a chat endpoint.
        is_chat_post = request.method == "POST" and (
            request.endpoint == "home" or request.path.startswith("/api/chat")
        )
        if not is_chat_post:
            return None
        if backend.get(_subject_id()).is_locked:
            log.warning("SECURITY_REQUEST_REJECTED reason=active_lockout path=%s", request.path)
            return _response(LOCKED_MESSAGE, "locked", 429, history_provider)
        return None

    @app.errorhandler(BadRequestError)
    def handle_openai_bad_request(exc: BadRequestError):
        blocked = is_jailbreak_filter_error(exc)
        # Never log prompts, Azure bodies, URLs, exception strings, or tracebacks.
        log.warning(
            "AZURE_OPENAI_REQUEST_REJECTED category=%s path=%s",
            "jailbreak" if blocked else "bad_request",
            request.path,
        )
        if blocked and failed_turn_cleanup is not None:
            try:
                failed_turn_cleanup()
            except Exception:
                # Do not turn safe error handling into a raw 500. Log only a
                # fixed event name, never exception details or message content.
                log.error("SECURITY_FAILED_TURN_CLEANUP_FAILED")

        if not blocked:
            return _response(_GENERIC_MESSAGE, "error", 502, history_provider)

        state = backend.record_violation(_subject_id())
        if state.strike_count >= MAX_STRIKES:
            return _response(LOCKOUT_STARTED_MESSAGE, "locked", 429, history_provider)
        warning = f"This request violates safety guidelines. Warning {state.strike_count}/3."
        return _response(warning, "blocked", 400, history_provider)
