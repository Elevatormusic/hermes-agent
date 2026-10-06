"""One same-prefix request for context compaction.

The host keeps the last completed Chat Completions request of the session. A
compaction attempt can send that request once more with the history rows that
came after it (the reply, tool results, a new user message) and one appended
user message. The system prompt, tools, tool choice, reasoning settings, and
extra body stay the same, so a server with prefix caching can reuse the cached
prefix instead of reading the whole conversation again. This works for a manual
/compress and for automatic compaction before or inside a turn.

The request is not streamed, dispatches no tools, and saves nothing to the
session. It does not prove that the server reuses its cache; the reply usage
reports cached tokens only when the server sends them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time

from agent.message_metadata import without_persistence_fields

logger = logging.getLogger(__name__)

_WIRE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")
_UNSUPPORTED_SETTINGS = ("response_format", "grammar", "functions", "function_call", "modalities", "audio")
_DEFAULT_OUTPUT_RESERVE = 4096
_MAX_INSTRUCTION_BYTES = 65536
_MAX_TIMEOUT_S = 600


class PrefixRequestError(RuntimeError):
    """The same-prefix request is not available for this attempt. The message is a fixed reason code."""


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     default=str).encode()).hexdigest()


def _source(messages):
    return [{k: v for k, v in without_persistence_fields(m).items() if not k.startswith("_")} for m in messages]


def _key(row):
    return {k: row.get(k) for k in _WIRE_KEYS}


def _wire(row):
    """The Chat Completions form of a history row that came after the captured request."""
    out = {k: row[k] for k in _WIRE_KEYS if row.get(k) is not None}
    if row.get("role") == "assistant":
        out.setdefault("content", None if out.get("tool_calls") else "")
    return out


def _shape(row):
    """The structure that the wire copy of a history row keeps: role, answered call id, call ids and names.
    The content can differ: the host replays the bytes it sent, adds request-time context, and normalizes
    tool calls."""
    calls = row.get("tool_calls") or []
    names = []
    for call in calls if isinstance(calls, list) else [calls]:
        function = call.get("function") if isinstance(call, dict) else None
        names.append((call.get("id") if isinstance(call, dict) else None,
                      function.get("name") if isinstance(function, dict) else None))
    return row.get("role"), row.get("tool_call_id"), tuple(names)


def _words(content):
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return " ".join(str(content or "").split())


def _arguments(value):
    """The JSON value of tool-call arguments: spacing and key order are not a change."""
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return " ".join(value.split())


def _same_row(wire, row):
    """The sent row carries the stored row: the same shape, the stored text inside the sent text (the host
    adds request-time context), and the same tool-call arguments. A row that a hook or middleware rewrote
    would make the handoff summarize text that is not in the history it replaces."""
    if _shape(wire) != _shape(row) or _words(row.get("content")) not in _words(wire.get("content")):
        return False
    calls = lambda r: [_arguments((c.get("function") or {}).get("arguments")) for c in r.get("tool_calls") or []  # noqa: E731
                       if isinstance(c, dict)]
    return calls(wire) == calls(row)


def _route(agent):
    client_settings = getattr(agent, "_client_kwargs", {}) or {}
    client_binding = {key: str(client_settings.get(key, "")) for key in
                      ("base_url", "api_key", "organization", "project", "default_headers", "default_query")}
    return (_digest({key: getattr(agent, key, None) for key in
                     ("session_id", "model", "provider", "base_url", "api_mode",
                      "tools", "reasoning_config", "_cached_system_prompt")}),
            _digest(client_binding), id(getattr(agent, "client", None)), id(agent.context_compressor))


def enabled(agent):
    """Capture only when the context engine asks for the same-prefix request."""
    return getattr(getattr(agent, "context_compressor", None), "wants_prefix_request", False) is True


def final_body(kwargs):
    """Merge extra_body like the SDK does. Request-local headers and queries are not supported."""
    if kwargs.get("extra_query") or kwargs.get("extra_headers"):
        raise PrefixRequestError("request_options_unsupported")
    extra = kwargs.get("extra_body") or {}
    if type(extra) is not dict:
        raise PrefixRequestError("request_options_unsupported")
    body = {k: v for k, v in kwargs.items() if k not in {"extra_body", "extra_headers", "extra_query", "timeout"}}
    body.update(extra)
    return _copy(body)


def begin_capture(agent, kwargs):
    """Record the final body of one physical ordinary call before it is sent."""
    if not enabled(agent):
        return
    agent._prefix_capsule = None
    agent._prefix_capture = None
    source = getattr(agent, "_prefix_source_messages", None)
    if type(source) is not list:
        return
    try:
        agent._prefix_capture = {"source": _copy(_source(source)), "route": _route(agent), "body": final_body(kwargs)}
    except (PrefixRequestError, TypeError, ValueError):
        # An unsupported capture must never stop an ordinary request.
        return


def capture_response(agent, kwargs, response):
    """Bind the capture to the response that this exact body produced. Streamed and plain responses."""
    if not enabled(agent):
        return response
    capture = getattr(agent, "_prefix_capture", None)
    try:
        if capture is None or final_body(kwargs) != capture["body"]:
            return response
        object.__setattr__(response, "_hermes_prefix_capture", capture)
    except (PrefixRequestError, TypeError, ValueError, AttributeError):
        pass
    return response


def _single_reply(response):
    """Return (finish_reason, message) of a one-choice response, else (None, None)."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return None, None
    return getattr(choices[0], "finish_reason", None), getattr(choices[0], "message", None)


def publish_response(agent, response):
    """Keep the capture for a complete reply (text or tool calls) that passed the host redirect check.
    The reply itself is not kept: the history rows after the captured request carry it."""
    if not enabled(agent):
        return
    capture = getattr(response, "_hermes_prefix_capture", None)
    if capture is None or capture is not getattr(agent, "_prefix_capture", None) or capture["route"] != _route(agent):
        return
    finish, message = _single_reply(response)
    if (finish not in ("stop", "tool_calls") or getattr(message, "role", None) != "assistant"
            or getattr(message, "function_call", None) or getattr(message, "refusal", None)):
        return
    agent._prefix_capsule = {**capture, "usage": _usage(response), "completed_at": time.monotonic()}


def _usage(response):
    usage = getattr(response, "usage", None)
    raw = usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else usage
    raw = raw if isinstance(raw, dict) else {}
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    count = lambda value: value if type(value) is int and value >= 0 else None  # noqa: E731
    # A missing cached-token count is unknown, not zero.
    return {"prompt_tokens": count(raw.get("prompt_tokens")), "completion_tokens": count(raw.get("completion_tokens")),
            "cache_read_tokens": count(cached)}


class PrefixRequest:
    """One same-prefix request for one compaction attempt. The engine gets no client or credential."""

    def __init__(self, agent, messages, commit_fence=None):
        self._agent, self._fence = agent, commit_fence
        self._messages = messages
        self._source = _copy(_source(messages))
        self._capsule = getattr(agent, "_prefix_capsule", None)
        self._route = _route(agent)
        self._lock = threading.Lock()
        self._used = False
        self._deadline = None
        live = getattr(agent, "_session_messages", None)
        self._live = live if type(live) is list else messages
        self._live_digest = _digest(_source(self._live))

    @property
    def capture_age_s(self):
        """Seconds since the captured request completed. None without a capture."""
        completed = (self._capsule or {}).get("completed_at")
        return None if completed is None else max(0.0, time.monotonic() - completed)

    @property
    def cache_read_tokens(self):
        """Cached prompt tokens that the server reported for the captured request. None when unknown."""
        usage = (self._capsule or {}).get("usage") or {}
        return usage.get("cache_read_tokens")

    def _check(self):
        agent = self._agent
        if self._fence is not None and self._fence.is_cancelled:
            raise PrefixRequestError("cancelled")
        event = getattr(agent, "_hard_interrupt_requested", None)
        if event is not None and event.is_set():
            raise PrefixRequestError("cancelled")
        if self._route != _route(agent):
            raise PrefixRequestError("route_changed")
        if _source(self._messages) != self._source or _digest(_source(self._live)) != self._live_digest:
            raise PrefixRequestError("history_changed")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise PrefixRequestError("deadline")

    @staticmethod
    def _text_messages(messages):
        if type(messages) is not list or not messages:
            raise PrefixRequestError("messages_unsupported")
        pending = set()
        for row in messages:
            if type(row) is not dict or row.get("role") not in {"system", "user", "assistant", "tool"}:
                raise PrefixRequestError("messages_unsupported")
            if row.get("content") is not None and type(row["content"]) is not str:
                raise PrefixRequestError("messages_unsupported")
            if row["role"] == "tool":
                if row.get("tool_call_id") not in pending:
                    raise PrefixRequestError("messages_unsupported")
                pending.remove(row["tool_call_id"])
            else:
                if pending:
                    raise PrefixRequestError("messages_unsupported")
                for call in row.get("tool_calls") or []:
                    if type(call) is not dict or not call.get("id") or call["id"] in pending:
                        raise PrefixRequestError("messages_unsupported")
                    pending.add(call["id"])
        if pending:
            raise PrefixRequestError("messages_unsupported")

    def _request(self, instruction):
        agent, capsule = self._agent, self._capsule
        if agent.api_mode != "chat_completions" or agent.provider == "moa":
            raise PrefixRequestError("api_mode_unsupported")
        if capsule is None:
            raise PrefixRequestError("no_capture")
        if capsule["route"] != self._route:
            raise PrefixRequestError("route_changed")
        source = capsule["source"]
        # The session must start with exactly the captured history. The rows after it (reply, tool results,
        # a new user message) go after the captured body, as the next ordinary request would send them.
        if (len(self._source) < len(source)
                or [_key(row) for row in self._source[:len(source)]] != [_key(row) for row in source]):
            raise PrefixRequestError("history_changed")
        suffix = [_wire(row) for row in self._source[len(source):]]
        body = _copy(capsule["body"])
        if (body.get("tool_choice") not in (None, "auto", "none") or body.get("n", 1) != 1
                or any(key in body for key in _UNSUPPORTED_SETTINGS)):
            raise PrefixRequestError("settings_unsupported")
        self._text_messages(body.get("messages"))
        offset = len(body["messages"]) - len(source)
        if offset < 0 or any(row.get("role") != "system" for row in body["messages"][:offset]):
            raise PrefixRequestError("source_transform_unsupported")
        # The body must render exactly these rows (no selection, merge, or extra row).
        rows = body["messages"][offset:]
        if len(rows) != len(source) or not all(_same_row(wire, row) for wire, row in zip(rows, source)):
            raise PrefixRequestError("source_transform_unsupported")
        from agent.model_metadata import estimate_messages_tokens_rough, estimate_request_tokens_rough
        # The server count of the captured request is exact; only the new rows are estimated.
        reported = ((capsule.get("usage") or {}).get("prompt_tokens"))
        prefix_tokens = reported or estimate_request_tokens_rough(body["messages"], tools=body.get("tools"))
        added = [*suffix, {"role": "user", "content": instruction}]
        body["messages"] = [*body["messages"], *added]
        self._text_messages(body["messages"])
        body["stream"] = False
        body.pop("stream_options", None)
        limit = int(getattr(agent.context_compressor, "context_length", 0) or 0)
        reserve = body.get("max_tokens") or body.get("max_completion_tokens") or _DEFAULT_OUTPUT_RESERVE
        if limit <= 0 or prefix_tokens + estimate_messages_tokens_rough(added) + reserve > limit:
            raise PrefixRequestError("capacity")
        return body

    def __call__(self, instruction, *, timeout_s=120.0):
        with self._lock:
            if self._used:
                raise PrefixRequestError("already_used")
            self._used = True
        if type(instruction) is not str or not instruction.strip() or len(instruction.encode()) > _MAX_INSTRUCTION_BYTES:
            raise PrefixRequestError("instruction_invalid")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= _MAX_TIMEOUT_S:
            raise PrefixRequestError("timeout_invalid")
        self._deadline = time.monotonic() + timeout_s
        if self._fence is not None and getattr(self._fence, "deadline_monotonic", None) is not None:
            self._deadline = min(self._deadline, self._fence.deadline_monotonic)
        self._check()
        body = self._request(instruction)
        agent = self._agent
        client = agent._create_request_openai_client(reason="context_prefix_request", api_kwargs=body)
        try:
            from openai import OpenAI
            if not isinstance(client, OpenAI) or client.max_retries != 0:
                raise PrefixRequestError("plain_sdk_required")
            self._check()
            started = time.monotonic()
            try:
                # The final body travels in extra_body; the SDK merges it after its typed fields.
                response = client.chat.completions.create(
                    model=body["model"], messages=[], extra_body=body, timeout=self._deadline - started)
            except Exception as error:
                raise PrefixRequestError("provider_error") from error
            elapsed = time.monotonic() - started
            self._check()
        finally:
            agent._close_request_openai_client(client, reason="context_prefix_request")
        finish, message = _single_reply(response)
        if message is None:
            raise PrefixRequestError("incomplete_response")
        content = getattr(message, "content", None)
        return {"content": content if type(content) is str else None, "finish_reason": finish,
                "tool_calls": bool(getattr(message, "tool_calls", None) or getattr(message, "function_call", None)),
                "refusal": bool(getattr(message, "refusal", None)), "usage": _usage(response),
                "elapsed_s": round(elapsed, 3)}


def build_prefix_request(agent, messages, commit_fence=None):
    """The one-shot same-prefix request for an engine that asks for it (``wants_prefix_request``), else None.
    It must never stop the compaction: a failure here only means no warm request for this attempt."""
    if not enabled(agent):
        return None
    try:
        return PrefixRequest(agent, messages, commit_fence)
    except Exception as exc:  # health: allow BLE001 -- an optional fast path must never stop a compaction
        logger.info("same-prefix request not available for this compaction (%s)", type(exc).__name__)
        return None
