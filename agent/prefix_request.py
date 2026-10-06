"""One same-prefix request for context compaction.

The host keeps the last completed Chat Completions or Anthropic Messages request of the session. A
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

import copy
import hashlib
import json
import re
import logging
import math
import threading
import time

from agent.message_metadata import without_persistence_fields

logger = logging.getLogger(__name__)

_WIRE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")
_UNSUPPORTED_SETTINGS = ("response_format", "grammar", "functions", "function_call", "modalities", "audio")
_DEFAULT_OUTPUT_RESERVE = 4096
# The reply limit of the handoff. The instruction asks for at most 600 words; the upper limit leaves room for a
# thinking model that counts its reasoning in the same limit.
_HANDOFF_MIN_TOKENS, _HANDOFF_MAX_TOKENS = 2048, 8192
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
    """The row fields that the provider sees, with the ``api_content`` sidecar that replaces the content."""
    return {**{k: row.get(k) for k in _WIRE_KEYS}, "api_content": row.get("api_content")}


def _sent(row):
    """The row as the main loop sends it: the ``api_content`` sidecar replaces ``content``."""
    from agent.turn_context import substitute_api_content  # Lazy: an import cycle through conversation_compression.

    row = dict(row)
    substitute_api_content(row)
    return row


def _wire(row, copy_reasoning=None):
    """The Chat Completions form of a history row that came after the captured request. ``copy_reasoning`` is
    the agent's ``_copy_reasoning_content_for_api``: reasoning_content as a main request sends it.
    ``reasoning_details`` stays here; the transport keeps it only on a route that replays it."""
    source, row = row, _sent(row)
    out = {k: row[k] for k in (*_WIRE_KEYS, "reasoning_details") if row.get(k) is not None}
    if copy_reasoning is not None:
        copy_reasoning(source, out)
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


def _text(content):
    """The text of a content value. White space stays: it can change code, tables, or commands."""
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _arguments(value):
    """Canonical tool-call arguments: spacing and key order are not a change, a JSON type is (``true`` and ``1``
    differ, which Python ``==`` does not see). Text that is not JSON stays as it is."""
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return "text", value
    return "json", json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _same_row(wire, row):
    """The sent row carries the stored row: the same shape, the same name (the transport drops ``name`` from
    tool rows only), the stored text (``api_content`` when the row has it) inside the sent text (the host adds
    request-time context), and the same tool-call arguments. Non-text parts are refused before this check
    (``messages_unsupported``). A row that a hook or middleware rewrote would make the handoff summarize text
    that is not in the history it replaces."""
    if _shape(wire) != _shape(row):
        return False
    if wire.get("name") != row.get("name") and not (wire.get("name") is None and row.get("role") == "tool"):
        return False
    # The host stores the text that it sends (api_content): other text is a rewrite, on its own line too.
    if _text(_sent(row).get("content")) != _text(wire.get("content")):
        return False
    calls = lambda r: [_arguments((c.get("function") or {}).get("arguments")) for c in r.get("tool_calls") or []  # noqa: E731
                       if isinstance(c, dict)]
    return calls(wire) == calls(row)


# Fields that the main loop adds to a sent row apart from the transport conversion: the prompt caching marker.
_TRANSPORT_FIELDS = frozenset({"cache_control"})


def _no_extra_fields(wire, expected):
    """The sent row and its tool calls have no field that the transport does not send for the stored row (a
    provider control that a middleware added, for example): the model reads it."""
    if set(wire) - set(expected) - _TRANSPORT_FIELDS:
        return False
    for call, want in zip(wire.get("tool_calls") or [], expected.get("tool_calls") or []):
        if isinstance(call, dict) and isinstance(want, dict) and (set(call) - set(want) or (
                isinstance(call.get("function"), dict) and set(call["function"]) - set(want.get("function") or {}))):
            return False
        # The same keys with another type value: the provider reads another kind of call.
        if isinstance(call, dict) and isinstance(want, dict) and "type" in call and call["type"] != want.get("type"):
            return False
    return True


def _same_replay_fields(wire, expected):
    """The sent row has the reasoning fields and thought signatures that the main loop replays for the stored
    row on this route. The provider reads them: a middleware that changed them made a prefix that the history
    does not have."""
    if any(wire.get(key) != expected.get(key) for key in ("reasoning_content", "reasoning_details")):
        return False
    def signatures(row):
        return [call.get("extra_content") for call in row.get("tool_calls") or [] if isinstance(call, dict)]
    return signatures(wire) == signatures(expected)


def _dump(response):
    """A comparable copy of an SDK response."""
    dump = getattr(response, "model_dump", None)
    return dump() if callable(dump) else repr(response)


def _ends_with_instruction(row, instruction):
    """The row is a user row whose last block is the host instruction (it can join the last user row)."""
    if not isinstance(row, dict) or row.get("role") != "user":
        return False
    content = row.get("content")
    if isinstance(content, str):
        return content == instruction or content.endswith("\n\n" + instruction)
    return (isinstance(content, list) and bool(content) and isinstance(content[-1], dict)
            and content[-1].get("type") == "text" and content[-1].get("text") == instruction)


def _join_user_rows(rows, instruction):
    """Join adjacent user rows of the same author as the main loop joins them (``_merge_user_content``), then
    join the host instruction to the last user row (it keeps its name: the ordinary request also ended with that
    row). The ordinary request has no adjacent user rows, and strict chat templates refuse them. The instruction
    is the last block; it says that it comes from the host."""
    from agent.agent_runtime_helpers import _UNMERGEABLE, _merge_user_content

    out = []
    for row in rows:
        last = out[-1] if out else None
        if (last is not None and last.get("role") == row.get("role") == "user" and last.get("name") == row.get("name")
                and set(last) <= {"role", "content", "name"} and set(row) <= {"role", "content", "name"}):
            joined = _merge_user_content(last.get("content"), row.get("content"))
            if joined is not _UNMERGEABLE:
                out[-1] = {**last, "content": joined}
                continue
        out.append(row)
    last = out[-1] if out else None
    if last is not None and last.get("role") == "user" and set(last) <= {"role", "content", "name"}:
        joined = _merge_user_content(last.get("content"), instruction)
        if joined is not _UNMERGEABLE:
            out[-1] = {**last, "content": joined}
            return out
    return [*out, {"role": "user", "content": instruction}]


def _same_request(base, request, count, instruction):
    """The sent request keeps what the cache and the summary depend on: every setting (model, tools, sampling),
    the first ``count`` messages (the captured request), and the host instruction as the last block of the last
    message. Only the rows after the capture can differ, for example after a redaction."""
    return (isinstance(request, dict) and isinstance(request.get("messages"), list)
            and len(request["messages"]) > count
            and {k: v for k, v in request.items() if k != "messages"}
            == {k: v for k, v in base.items() if k != "messages"}
            and request["messages"][:count] == base["messages"][:count]
            and _ends_with_instruction(request["messages"][-1], instruction))


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


def _merged(kwargs):
    """The request body that the SDK sends: kwargs with extra_body merged. Request-local headers and queries are
    not supported."""
    if kwargs.get("extra_query") or kwargs.get("extra_headers"):
        raise PrefixRequestError("request_options_unsupported")
    extra = kwargs.get("extra_body") or {}
    if type(extra) is not dict:
        raise PrefixRequestError("request_options_unsupported")
    body = {k: v for k, v in kwargs.items() if k not in {"extra_body", "extra_headers", "extra_query", "timeout"}}
    body.update(extra)
    return body


def final_body(kwargs):
    """A JSON copy of the request body that the SDK sends."""
    return _copy(_merged(kwargs))


_ANTHROPIC_MESSAGES = "anthropic_messages"
# Anthropic Messages settings that a replay cannot keep as they are.
_ANTHROPIC_UNSUPPORTED_SETTINGS = ("container", "mcp_servers", "context_management", "compaction")
# A thinking request counts its reasoning in max_tokens: the handoff gets more room there.
_HANDOFF_THINKING_MAX_TOKENS = 32768


def _is_anthropic(agent):
    return getattr(agent, "api_mode", None) == _ANTHROPIC_MESSAGES


def _anthropic_body(kwargs):
    """The Anthropic ``messages.create`` arguments that the SDK sends. A per-request ``anthropic-beta`` header
    (fast mode) is part of the request and is replayed as it is; request-local queries are not supported."""
    if kwargs.get("extra_query"):
        raise PrefixRequestError("request_options_unsupported")
    headers, extra = kwargs.get("extra_headers"), kwargs.get("extra_body")
    if headers is not None and (type(headers) is not dict
                                or not all(type(k) is str and type(v) is str for k, v in headers.items())):
        raise PrefixRequestError("request_options_unsupported")
    if extra is not None and type(extra) is not dict:
        raise PrefixRequestError("request_options_unsupported")
    return {k: v for k, v in kwargs.items() if k not in {"timeout", "extra_query"}}


def _route_body(agent, kwargs):
    """The request body of this route: Chat Completions merges ``extra_body``; Anthropic keeps its arguments."""
    return _anthropic_body(kwargs) if _is_anthropic(agent) else _merged(kwargs)


def _without_cache_control(value):
    """A copy without prompt-cache markers, at any depth (tool_result content carries them too)."""
    if isinstance(value, dict):
        return {k: _without_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_without_cache_control(v) for v in value]
    return value


def _comparable(row):
    """An Anthropic message as the model reads it: no prompt-cache marker, and the shapes that the cache
    decoration makes (a string turned into a text block, a text split before a marker) joined back."""
    clean = {k: _without_cache_control(v) for k, v in row.items() if k != "cache_control"}
    content = clean.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(parts, list):
        return clean
    joined = []
    for part in parts:
        last = joined[-1] if joined else None
        if isinstance(last, dict) and isinstance(part, dict) and _plain_text(part) and _plain_text(last):
            joined[-1] = {"type": "text", "text": str(last.get("text")) + str(part.get("text"))}
        else:
            joined.append(part)
    clean["content"] = joined
    return clean


def _plain_text(part):
    return isinstance(part, dict) and part.get("type") == "text" and set(part) == {"type", "text"}


def _row_digests(messages):
    return [_digest(_key(row)) for row in _source(messages)]


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
        body = _copy(_route_body(agent, kwargs))
        agent._prefix_capture = {"source": _row_digests(source), "route": _route(agent), "body": body,
                                 "body_digest": _digest(body)}
    except (PrefixRequestError, TypeError, ValueError):
        # An unsupported capture must never stop an ordinary request.
        return


def capture_response(agent, kwargs, response):
    """Bind the capture to the response that this exact body produced. Streamed and plain responses."""
    if not enabled(agent):
        return response
    capture = getattr(agent, "_prefix_capture", None)
    try:
        if capture is None or _digest(_route_body(agent, kwargs)) != capture["body_digest"]:
            return response
        object.__setattr__(response, "_hermes_prefix_capture", capture)
    except (PrefixRequestError, TypeError, ValueError, AttributeError):
        pass
    return response


def _anthropic_reply(response):
    """(finish_reason, message) of an Anthropic Messages response, in the Chat Completions form that the rest of
    this module reads: the text blocks joined (thinking blocks are not reply text), ``tool_calls`` for tool_use
    blocks, and the stop reason mapped as the transport maps it for a main request."""
    from types import SimpleNamespace

    from agent.transports.anthropic import AnthropicTransport

    blocks = getattr(response, "content", None)
    if not isinstance(blocks, (list, tuple)):
        return None, None
    kind = lambda block: getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")  # noqa: E731
    texts = [getattr(b, "text", None) if not isinstance(b, dict) else b.get("text") for b in blocks if kind(b) == "text"]
    calls = [b for b in blocks if kind(b) in ("tool_use", "server_tool_use")]
    stop = getattr(response, "stop_reason", None)
    stop_map = dict(getattr(AnthropicTransport, "_STOP_REASON_MAP", None) or {})
    finish = stop_map.get(stop, stop) if isinstance(stop, str) else None
    message = SimpleNamespace(role=getattr(response, "role", None),
                              content="\n".join(t for t in texts if isinstance(t, str)) if texts else None,
                              tool_calls=calls or None, function_call=None, refusal=None)
    return finish, message


def _single_reply(response):
    """Return (finish_reason, message) of a one-choice response, else (None, None). The finish reason is the
    lowercase contract value (``STOP`` is ``stop``), as the transport gives it for a main request. An Anthropic
    Messages response (no ``choices``) is read through ``_anthropic_reply``."""
    from agent.message_sanitization import normalize_finish_reason

    choices = getattr(response, "choices", None)
    if choices is None and getattr(response, "type", None) == "message" and hasattr(response, "stop_reason"):
        return _anthropic_reply(response)
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return None, None
    return normalize_finish_reason(getattr(choices[0], "finish_reason", None)), getattr(choices[0], "message", None)


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
    count = lambda value: value if type(value) is int and value >= 0 else None  # noqa: E731
    if "input_tokens" in raw and "prompt_tokens" not in raw:
        # Anthropic Messages: input_tokens excludes the prompt tokens read from or written to the cache.
        fresh, read, written = (count(raw.get(key)) for key in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        prompt = None if fresh is None else fresh + (read or 0) + (written or 0)
        return {"prompt_tokens": prompt, "completion_tokens": count(raw.get("output_tokens")),
                "cache_read_tokens": read}
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
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

    def _captured_history(self):
        """The capture of this attempt, checked against the route and the history: (row count, capture)."""
        capsule = self._capsule
        if capsule is None:
            raise PrefixRequestError("no_capture")
        if capsule["route"] != self._route:
            raise PrefixRequestError("route_changed")
        # The session must start with exactly the captured history (one digest for each row). The rows after it
        # (reply, tool results, a new user message) go after the captured body, as the next ordinary request
        # would send them.
        count = len(capsule["source"])
        if len(self._source) < count or _row_digests(self._source[:count]) != capsule["source"]:
            raise PrefixRequestError("history_changed")
        return count, capsule

    def _anthropic_render(self, rows):
        """The Anthropic messages that the main loop's request builder makes from these history rows: the same
        converter, the same Claude Code (OAuth) tool naming, the same route settings."""
        agent = self._agent
        rows = [_sent(row) for row in _copy(rows)]
        prepare = getattr(agent, "_prepare_anthropic_messages_for_api", None)
        rows = prepare(rows) if callable(prepare) else rows
        preserve_dots = getattr(agent, "_anthropic_preserve_dots", None)
        compressor = getattr(agent, "context_compressor", None)
        kwargs = agent._get_transport().build_kwargs(
            model=agent.model, messages=rows, tools=agent.tools, max_tokens=None,
            reasoning_config=getattr(agent, "reasoning_config", None),
            is_oauth=bool(getattr(agent, "_is_anthropic_oauth", False)),
            preserve_dots=bool(preserve_dots()) if callable(preserve_dots) else False,
            context_length=getattr(compressor, "context_length", None) or None,
            base_url=getattr(agent, "_anthropic_base_url", None),
            drop_context_1m_beta=bool(getattr(agent, "_oauth_1m_beta_disabled", False)))
        return kwargs["messages"]

    def _anthropic_request(self, instruction):
        """The Anthropic Messages form of the same-prefix request. The captured messages stay byte for byte as
        they were sent, with their ``cache_control`` breakpoints, so the server reads them from its prompt cache.
        The rows after the capture follow without a breakpoint, then the instruction as the last text block."""
        count, capsule = self._captured_history()
        body = _copy(capsule["body"])
        if (body.get("tool_choice") not in (None, {"type": "auto"}, {"type": "none"})
                or any(key in body for key in _ANTHROPIC_UNSUPPORTED_SETTINGS)):
            raise PrefixRequestError("settings_unsupported")
        captured = body.get("messages")
        if (type(captured) is not list or not captured
                or any(type(row) is not dict or row.get("role") not in ("user", "assistant") for row in captured)):
            raise PrefixRequestError("messages_unsupported")
        # The body must be the rendering of exactly the stored rows: a hook or middleware that rewrote a row would
        # make the handoff summarize text that is not in the history it replaces. The rendering of the whole
        # history must start with the same messages (no merge across the capture boundary).
        try:
            prefix, rendered = self._anthropic_render(self._source[:count]), self._anthropic_render(self._source)
        except PrefixRequestError:
            raise
        except Exception as error:  # health: allow BLE001 -- a rendering failure only means no warm request
            raise PrefixRequestError("source_transform_unsupported") from error
        want = [_comparable(row) for row in captured]
        if ([_comparable(row) for row in prefix] != want
                or [_comparable(row) for row in rendered[:len(captured)]] != want):
            raise PrefixRequestError("source_transform_unsupported")
        messages = [*captured, *_without_cache_control(rendered[len(captured):])]
        block = {"type": "text", "text": instruction}
        last = messages[-1]
        if len(messages) > len(captured) and last.get("role") == "user":
            # Join the instruction to the last new user row (tool results, a new message); never to a captured row.
            content = last.get("content")
            content = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
            messages[-1] = {**last, "content": [*content, block]}
        else:
            messages.append({"role": "user", "content": [block]})
        body["messages"] = messages
        body.pop("stream", None)
        # A stop sequence of the main request could cut the handoff after its headings.
        body.pop("stop_sequences", None)
        # The reply limit of the main request is for another task; the handoff gets its own. A thinking request
        # counts its reasoning in the same limit. The limit does not take part in the prompt cache.
        thinking = body.get("thinking")
        limit = _HANDOFF_MAX_TOKENS
        if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive"):
            budget = thinking.get("budget_tokens")
            limit = max(_HANDOFF_THINKING_MAX_TOKENS,
                        (budget if type(budget) is int and budget > 0 else 0) + _HANDOFF_MAX_TOKENS)
        current = body.get("max_tokens")
        body["max_tokens"] = min(current, limit) if type(current) is int and current > 0 else limit
        from agent.model_metadata import estimate_request_tokens_rough
        reported = (capsule.get("usage") or {}).get("prompt_tokens")
        self._prefix_tokens = reported or estimate_request_tokens_rough(
            [*([{"role": "system", "content": json.dumps(body.get("system"))}] if body.get("system") else []),
             *captured], tools=body.get("tools"))
        self._check_capacity(body, len(captured))
        return body

    def _request(self, instruction):
        agent = self._agent
        if agent.provider == "moa":
            raise PrefixRequestError("api_mode_unsupported")
        if _is_anthropic(agent):
            return self._anthropic_request(instruction)
        if agent.api_mode != "chat_completions":
            raise PrefixRequestError("api_mode_unsupported")
        count, capsule = self._captured_history()
        source = self._source[:count]
        copy_reasoning = getattr(agent, "_copy_reasoning_content_for_api", None)
        suffix = [_wire(row, copy_reasoning) for row in self._source[len(source):]]
        # The main loop's transport rule: no call_id or response_item_id, extra_content only for Gemini, no name
        # on tool rows.
        from agent.transports.chat_completions import ChatCompletionsTransport
        suffix = ChatCompletionsTransport().convert_messages(suffix, model=agent.model, base_url=agent.base_url)
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
        expected = ChatCompletionsTransport().convert_messages([_wire(row, copy_reasoning) for row in source],
                                                               model=agent.model, base_url=agent.base_url)
        if not all(_same_replay_fields(wire, want) and _no_extra_fields(wire, want) for wire, want in zip(rows, expected)):
            raise PrefixRequestError("source_transform_unsupported")
        from agent.model_metadata import estimate_request_tokens_rough
        # The server count of the captured request is exact; only the new rows are estimated.
        reported = ((capsule.get("usage") or {}).get("prompt_tokens"))
        self._prefix_tokens = reported or estimate_request_tokens_rough(body["messages"], tools=body.get("tools"))
        added = _join_user_rows(suffix, instruction)
        body["messages"] = [*body["messages"], *added]
        self._text_messages(body["messages"])
        body["stream"] = False
        body.pop("stream_options", None)
        # A stop sequence of the main request could cut the handoff after its headings.
        body.pop("stop", None)
        # A web search costs a search and can bring text that is not in the conversation into the handoff.
        body.pop("web_search_options", None)
        # The reply limit of the main request is for another task: a small one cuts the handoff, a large one
        # reserves space that the handoff does not need. Keep the field that the route uses.
        limits = ("max_tokens", "max_completion_tokens")
        for key in limits:
            value = body.get(key)
            if type(value) is int and value > 0:
                body[key] = min(max(value, _HANDOFF_MIN_TOKENS), _HANDOFF_MAX_TOKENS)
        # Without a limit the server default applies: a small one cuts the handoff, a large one can take more space
        # than the capacity check reserves. The handoff gets its own limit in the field of the route.
        if not any(type(body.get(key)) is int and body.get(key) > 0 for key in limits):
            body.update(self._agent._max_tokens_param(_HANDOFF_MAX_TOKENS))
        self._check_capacity(body, len(capsule["body"]["messages"]))
        return body

    def _remaining_s(self):
        """Seconds left before the deadline of this attempt (set in ``__call__`` before any send)."""
        return (self._deadline if self._deadline is not None else time.monotonic()) - time.monotonic()

    def _check_capacity(self, body, count):
        """The measured (or estimated) captured prefix, the estimated rows after it, and the reply reserve must
        fit in the context window."""
        from agent.model_metadata import estimate_messages_tokens_rough
        limit = int(getattr(self._agent.context_compressor, "context_length", 0) or 0)
        # A server can honor either reply limit: reserve the larger one.
        limits = [body.get(key) for key in ("max_tokens", "max_completion_tokens")]
        reserve = max([value for value in limits if type(value) is int and value > 0] or [_DEFAULT_OUTPUT_RESERVE])
        if limit <= 0 or self._prefix_tokens + estimate_messages_tokens_rough(body["messages"][count:]) + reserve > limit:
            raise PrefixRequestError("capacity")

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
        from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        context = {"purpose": "context_prefix_request", "api_request_id": None, "session_id": agent.session_id or "",
                   "model": agent.model, "provider": agent.provider, "base_url": agent.base_url,
                   "api_mode": agent.api_mode}
        # Like a main request, the rows after the capture go through llm_request middleware (redaction, policy).
        try:
            changed = apply_llm_request_middleware(body, **context).payload
        except Exception as error:
            raise PrefixRequestError("middleware_refused") from error
        if not isinstance(changed, dict) or not isinstance(changed.get("messages"), list):
            raise PrefixRequestError("middleware_refused")
        # The captured request went through llm_request middleware already: a change to it (for example, an added
        # system row) would apply twice and change the cached prefix. The rows after it can change.
        count = len(self._capsule["body"]["messages"])
        if not _same_request(body, changed, count, instruction):
            raise PrefixRequestError("middleware_rewrite")
        body = changed
        # A request middleware can add text to the rows after the capture: check the size again.
        self._check_capacity(body, count)
        anthropic = _is_anthropic(agent)
        if anthropic:
            client = agent._create_request_anthropic_client(reason="context_prefix_request")
        else:
            client = agent._create_request_openai_client(reason="context_prefix_request", api_kwargs=body)
        try:
            if anthropic:
                # The SDK must not retry on its own: one attempt, cancellable through the deadline.
                if getattr(client, "max_retries", None) != 0:
                    raise PrefixRequestError("plain_sdk_required")
            else:
                from openai import OpenAI
                if not isinstance(client, OpenAI) or client.max_retries != 0:
                    raise PrefixRequestError("plain_sdk_required")
            self._check()
            started = time.monotonic()

            sent = []
            # A copy that no middleware can change in place: the request identity check below compares with it.
            base = copy.deepcopy(body)

            def _send(request):
                # The last seam before the provider: an execution middleware can have changed the request. The
                # captured part, the settings, and the instruction must be the ones that the checks above accepted.
                if not _same_request(base, request, count, instruction):
                    raise PrefixRequestError("middleware_rewrite")
                # An execution middleware can run after a cancel, a route switch, or the deadline: check again.
                self._check()
                if anthropic:
                    # The main loop's Messages call (streamed when the route streams); one final Message.
                    from agent.anthropic_adapter import create_anthropic_message
                    kwargs = {**copy.deepcopy(request), "timeout": self._remaining_s()}
                    response = create_anthropic_message(
                        client, kwargs, log_prefix=getattr(agent, "log_prefix", ""),
                        prefer_stream=not bool(getattr(agent, "_disable_streaming", False)))
                else:
                    # The final body travels in extra_body; the SDK merges it after its typed fields.
                    response = client.chat.completions.create(
                        model=request["model"], messages=[], extra_body=request, timeout=self._remaining_s())
                sent.append((response, _dump(response)))
                return response
            try:
                # Like a main request, this request goes through llm_execution middleware (audit, policy).
                response = run_llm_execution_middleware(body, _send, original_request=base, **context)
            except PrefixRequestError:
                raise
            except Exception as error:
                raise PrefixRequestError("provider_error") from error
            # Only the server's own reply: a middleware that skipped the request or changed its reply would make
            # a text that the model did not write replace the history. A block (None) is incomplete_response.
            if response is not None and (not sent or response is not sent[0][0] or _dump(response) != sent[0][1]):
                raise PrefixRequestError("middleware_changed_reply")
            elapsed = time.monotonic() - started
            self._check()
        finally:
            if anthropic:
                agent._close_request_anthropic_client(client, reason="context_prefix_request")
            else:
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
