"""Safe handling and Flask integration for Azure OpenAI content-filter errors."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable

from flask import jsonify, render_template, request
from markupsafe import Markup, escape
from openai import BadRequestError

BLOCKED_MESSAGE = "I cannot fulfill this request as it violates safety guidelines."
_GENERIC_MESSAGE = "The request could not be processed by the AI service. Please try rephrasing it."


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mapping-like SDK models as a mapping where possible."""
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def is_jailbreak_filter_error(exc: BaseException) -> bool:
    """Inspect supported Azure error-body shapes without exposing raw details."""
    body = _as_mapping(getattr(exc, "body", None))
    error = _as_mapping(body.get("error"))
    container = error or body
    innererror = _as_mapping(container.get("innererror"))
    filter_result = _as_mapping(innererror.get("content_filter_result"))
    jailbreak = _as_mapping(filter_result.get("jailbreak"))
    return jailbreak.get("detected") is True


def blocked_payload() -> dict[str, str]:
    """Return the stable, client-safe API response for a blocked request."""
    return {"status": "blocked", "message": BLOCKED_MESSAGE}


def _browser_message(message: str, blocked: bool) -> Markup:
    """Build trusted UI chrome while escaping all variable message text."""
    warning = ""
    if blocked:
        warning = (
            '<div class="app-inline-warning" role="status" aria-label="Safety warning">'
            '<span class="app-inline-warning__icon" aria-hidden="true">!</span>'
            "<span>Safety notice</span></div>"
        )
    return Markup(f"{warning}<p>{escape(message)}</p>")


def register_openai_error_handler(
    app: Any,
    *,
    history_provider: Callable[[], list[dict[str, Any]]] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Register a sanitising handler for Azure/OpenAI HTTP 400 responses.

    JSON/API clients receive a stable machine-readable response. Browser form
    clients receive the normal ``index.html`` chat interface with the safety
    response in an assistant bubble, so the input remains available.
    """
    log = logger or logging.getLogger(__name__)

    @app.errorhandler(BadRequestError)
    def handle_openai_bad_request(exc: BadRequestError):
        blocked = is_jailbreak_filter_error(exc)
        # Deliberately log no prompt, response body, URL, exception text, or traceback.
        log.warning(
            "AZURE_OPENAI_REQUEST_REJECTED category=%s path=%s",
            "jailbreak" if blocked else "bad_request",
            request.path,
        )

        message = BLOCKED_MESSAGE if blocked else _GENERIC_MESSAGE
        status = "blocked" if blocked else "error"
        http_status = 400 if blocked else 502
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"status": status, "message": message}), http_status

        history = list(history_provider() if history_provider else [])
        user_text = request.form.get("message", "").strip()
        if user_text:
            history.append(
                {
                    "role": "user",
                    "content": escape(user_text),
                    "raw_content": user_text,
                    "sources": "",
                    "full_table": None,
                }
            )
        history.append(
            {
                "role": "assistant",
                "content": _browser_message(message, blocked),
                "raw_content": None,
                "sources": "",
                "full_table": None,
            }
        )
        return render_template("index.html", messages=history, sanitisation_error=None), http_status
