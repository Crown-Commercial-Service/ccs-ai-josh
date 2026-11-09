"""Python startup hook that installs safe Azure OpenAI error handling.

The application creates its Flask instance in ``app.py``.  Patching the Flask
class before that module is imported ensures errors raised anywhere in the LLM,
SQL-generation, or retrieval pipeline are sanitised consistently, including
JSON/API requests.
"""

from __future__ import annotations

try:
    import flask

    from src.content_filtering import register_openai_error_handler

    _BaseFlask = flask.Flask

    class _ContentFilterSafeFlask(_BaseFlask):
        """Flask application with the Azure/OpenAI safety handler pre-registered."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            register_openai_error_handler(self)

    flask.Flask = _ContentFilterSafeFlask
except ImportError:
    # Keep command-line utilities usable in minimal environments where Flask or
    # OpenAI dependencies are intentionally not installed.
    pass
