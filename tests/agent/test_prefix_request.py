"""One same-prefix request for compaction: capture, request shape, and refusals, with a real SDK sink."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent.conversation_compression import CompressionCommitFence
from agent.prefix_request import (
    PrefixRequest, PrefixRequestError, begin_capture, capture_response, publish_response,
)

REPLY = "Completed synthetic reply."


def make_agent(usage=None, reply_content="Synthetic handoff."):
    calls = []

    def sink(request):
        calls.append(json.loads(request.content))
        content = REPLY if len(calls) == 1 else reply_content
        return httpx.Response(200, request=request, json={
            "id": "synthetic", "object": "chat.completion", "created": 0, "model": "synthetic",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": usage or {"prompt_tokens": 5000, "completion_tokens": 5, "total_tokens": 5005},
        })

    client = OpenAI(api_key="synthetic-no-secret", base_url="https://sink.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False))
    history = []
    for index in range(4):
        history.extend([{"role": "user", "content": f"{index} data" * 50},
                        {"role": "assistant", "content": " result" * 50}])
    history.append({"role": "user", "content": "Continue."})
    tools = [{"type": "function", "function": {"name": "synthetic_tool", "parameters": {"type": "object"}}}]
    agent = SimpleNamespace(
        client=client, context_compressor=SimpleNamespace(wants_prefix_request=True, context_length=65536),
        session_id="synthetic", model="synthetic", provider="custom", base_url="https://sink.invalid/v1",
        api_mode="chat_completions", tools=tools, reasoning_config={"effort": "high"},
        _cached_system_prompt="Keep the synthetic task.",
        _client_kwargs={"api_key": "synthetic-no-secret", "base_url": "https://sink.invalid/v1"},
        _prefix_source_messages=history, _session_messages=history,
        _create_request_openai_client=lambda **kwargs: client,
        _close_request_openai_client=lambda *args, **kwargs: None)
    ordinary = {"model": "synthetic", "messages": [
        {"role": "system", "content": agent._cached_system_prompt}, *copy.deepcopy(history)],
        "tools": tools, "tool_choice": "auto", "max_tokens": 2048, "stream": False,
        "extra_body": {"draft": True, "chat_template_kwargs": {"enable_thinking": True}, "seed": 7}}
    return agent, calls, ordinary, client, history


def ordinary_turn(agent, ordinary, history):
    begin_capture(agent, ordinary)
    response = _dispatch_nonstreaming_api_request(agent, ordinary, make_client=lambda *a, **k: agent.client)
    response = capture_response(agent, ordinary, response)
    publish_response(agent, response)
    history.append({"role": "assistant", "content": response.choices[0].message.content})
    return response


def test_request_keeps_the_prefix_and_settings_and_appends_reply_and_instruction():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        before = copy.deepcopy(history)
        result = PrefixRequest(agent, history, CompressionCommitFence())("Write the handoff.", timeout_s=30)
        assert len(calls) == 2
        first, second = calls
        assert second["messages"][:len(first["messages"])] == first["messages"]
        assert second["messages"][len(first["messages"]):] == [
            {"role": "assistant", "content": REPLY}, {"role": "user", "content": "Write the handoff."}]
        changed = {"messages", "stream", "stream_options"}
        assert {k: v for k, v in second.items() if k not in changed} == {
            k: v for k, v in first.items() if k not in changed}
        assert second["draft"] is True and second["chat_template_kwargs"] == {"enable_thinking": True}
        assert second["stream"] is False
        assert history == before
        assert result["content"] == "Synthetic handoff."
        assert result["finish_reason"] == "stop"
        assert result["tool_calls"] is False and result["refusal"] is False
        assert result["usage"] == {"prompt_tokens": 5000, "completion_tokens": 5, "cache_read_tokens": None}
        assert result["elapsed_s"] >= 0
    finally:
        client.close()


def test_cached_tokens_are_reported_when_the_server_sends_them():
    usage = {"prompt_tokens": 5000, "completion_tokens": 5, "total_tokens": 5005,
             "prompt_tokens_details": {"cached_tokens": 4864}}
    agent, calls, ordinary, client, history = make_agent(usage=usage)
    try:
        ordinary_turn(agent, ordinary, history)
        result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert result["usage"]["cache_read_tokens"] == 4864
    finally:
        client.close()


def test_a_streamed_turn_is_captured():
    agent, calls, ordinary, client, history = make_agent()
    try:
        streamed = {**ordinary, "stream": True, "stream_options": {"include_usage": True}}
        begin_capture(agent, streamed)
        message = SimpleNamespace(role="assistant", content=REPLY, tool_calls=None, refusal=None,
                                  reasoning_content="hidden reasoning")
        response = SimpleNamespace(id="stream-1", model="synthetic", usage=None,
                                   choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")])
        publish_response(agent, capture_response(agent, streamed, response))
        history.append({"role": "assistant", "content": REPLY})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
        assert calls[0]["stream"] is False and "stream_options" not in calls[0]
        assert calls[0]["messages"][-2:] == [{"role": "assistant", "content": REPLY},
                                             {"role": "user", "content": "Write the handoff."}]
    finally:
        client.close()


def test_only_one_attempt():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        request = PrefixRequest(agent, history)
        request("Write the handoff.", timeout_s=30)
        with pytest.raises(PrefixRequestError, match="already_used"):
            request("Again.", timeout_s=30)
        assert len(calls) == 2
    finally:
        client.close()


@pytest.mark.parametrize("change, reason", [
    (lambda agent, history: history.__setitem__(0, {"role": "user", "content": "Edited."}), "history_changed"),
    (lambda agent, history: history.__setitem__(slice(-3, -1), []), "history_changed"),
    (lambda agent, history: setattr(agent, "model", "other"), "route_changed"),
    (lambda agent, history: setattr(agent, "api_mode", "codex_responses"), "api_mode_unsupported"),
    (lambda agent, history: setattr(agent.context_compressor, "context_length", 1000), "capacity"),
])
def test_refuses_before_any_request(change, reason):
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        change(agent, history)
        with pytest.raises(PrefixRequestError, match=reason):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_refuses_without_a_successful_capture():
    agent, calls, ordinary, client, history = make_agent()
    try:
        history.append({"role": "assistant", "content": REPLY})
        with pytest.raises(PrefixRequestError, match="no_capture"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls == []
    finally:
        client.close()


def test_rows_after_the_captured_request_are_appended_in_order():
    """Automatic compaction before a turn: the reply and the new user message follow the captured prefix."""
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "user", "content": "Next question.", "timestamp": 5})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        first, second = calls
        n = len(first["messages"])
        assert second["messages"][:n] == first["messages"]
        assert second["messages"][n:] == [{"role": "assistant", "content": REPLY},
                                          {"role": "user", "content": "Next question."},
                                          {"role": "user", "content": "Write the handoff."}]
    finally:
        client.close()


def test_a_tool_call_reply_is_published_and_its_tool_rows_follow():
    """Automatic compaction inside a turn: the tool call and its result follow the captured prefix."""
    agent, calls, ordinary, client, history = make_agent()
    try:
        begin_capture(agent, ordinary)
        call = SimpleNamespace(id="call-1", type="function", function=SimpleNamespace(name="synthetic_tool",
                                                                                      arguments="{}"))
        message = SimpleNamespace(role="assistant", content=None, tool_calls=[call], refusal=None)
        response = SimpleNamespace(usage={"prompt_tokens": 5000, "prompt_tokens_details": {"cached_tokens": 4000}},
                                   choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls")])
        publish_response(agent, capture_response(agent, ordinary, response))
        assert agent._prefix_capsule["usage"]["cache_read_tokens"] == 4000
        wire_call = {"id": "call-1", "type": "function", "function": {"name": "synthetic_tool", "arguments": "{}"}}
        history.extend([{"role": "assistant", "content": None, "reasoning": "hidden", "tool_calls": [wire_call]},
                        {"role": "tool", "tool_call_id": "call-1", "name": "synthetic_tool", "content": "result"}])
        request = PrefixRequest(agent, history)
        assert request.cache_read_tokens == 4000
        request("Write the handoff.", timeout_s=30)
        assert calls[0]["messages"][-3:] == [
            {"role": "assistant", "content": None, "tool_calls": [wire_call]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},  # The transport drops a tool name.
            {"role": "user", "content": "Write the handoff."}]
    finally:
        client.close()


def test_the_capacity_check_uses_the_reported_prompt_count():
    agent, calls, ordinary, client, history = make_agent(
        usage={"prompt_tokens": 64000, "completion_tokens": 5, "total_tokens": 64005})
    try:
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="capacity"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_capture_age_is_known_after_a_capture():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        assert 0 <= PrefixRequest(agent, history).capture_age_s < 60
    finally:
        client.close()


def test_cache_read_tokens_is_unknown_without_a_capture():
    agent, calls, ordinary, client, history = make_agent()
    try:
        assert PrefixRequest(agent, history).cache_read_tokens is None
        assert PrefixRequest(agent, history).capture_age_s is None
    finally:
        client.close()


def test_a_cancelled_fence_refuses():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        fence = CompressionCommitFence()
        fence.cancel_before_commit()
        with pytest.raises(PrefixRequestError, match="cancelled"):
            PrefixRequest(agent, history, fence)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_no_capture_unless_the_engine_asks_for_it():
    agent, calls, ordinary, client, history = make_agent()
    try:
        agent.context_compressor.wants_prefix_request = False
        begin_capture(agent, ordinary)
        assert getattr(agent, "_prefix_capture", None) is None
    finally:
        client.close()


def test_request_local_headers_are_not_captured():
    agent, calls, ordinary, client, history = make_agent()
    try:
        begin_capture(agent, {**ordinary, "extra_headers": {"x-initiator": "user"}})
        assert getattr(agent, "_prefix_capture", None) is None
    finally:
        client.close()


def _tool_round(history, ordinary):
    """One tool round, stored the way Hermes stores it, and the wire copy that Hermes sends for it."""
    history[:] = [
        {"role": "user", "content": "Read the file."},
        {"role": "assistant", "content": "", "finish_reason": "tool_calls", "reasoning": "Look first.",
         "tool_calls": [{"id": "call_1", "call_id": "call_1", "response_item_id": "fc_1", "type": "function",
                         "function": {"name": "synthetic_tool", "arguments": '{"path": "a.txt"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "synthetic_tool", "content": "file text " * 50},
        {"role": "user", "content": "Continue."},
    ]
    ordinary["messages"] = [ordinary["messages"][0],
        {"role": "user", "content": "Read the file.\n\n[recalled context: the user wants short answers]"},
        {"role": "assistant", "content": "", "reasoning_content": "Look first.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "synthetic_tool", "arguments": '{"path":"a.txt"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file text " * 50},
        {"role": "user", "content": "Continue."}]


def test_host_wire_transforms_of_the_same_rows_are_accepted():
    agent, calls, ordinary, client, history = make_agent()
    try:
        _tool_round(history, ordinary)
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 2
        assert calls[1]["messages"][:len(calls[0]["messages"])] == calls[0]["messages"]
    finally:
        client.close()


@pytest.mark.parametrize("change", [
    lambda messages: messages[1].__setitem__("content", "A different request."),
    lambda messages: messages[2]["tool_calls"][0]["function"].__setitem__("arguments", '{"path":"b.txt"}'),
    lambda messages: messages.__delitem__(slice(1, 4)),
    lambda messages: messages.append({"role": "user", "content": "Request-time note."}),
    lambda messages: messages[1].__setitem__("name", "another_user"),
    lambda messages: messages[2]["tool_calls"][0].__setitem__("id", "call_9") or messages[3].__setitem__(
        "tool_call_id", "call_9"),
])
def test_a_request_with_other_rows_is_refused(change):
    agent, calls, ordinary, client, history = make_agent()
    try:
        _tool_round(history, ordinary)
        change(ordinary["messages"])
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def _execution_middleware(monkeypatch, *callbacks):
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(_middleware={"llm_execution": list(callbacks)},
                              _report_hook_failure=lambda *args, **kwargs: None)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)


def test_the_request_runs_through_the_execution_middleware(monkeypatch):
    seen = []

    def audit(request=None, next_call=None, **context):
        seen.append((len(request["messages"]), context.get("purpose"), context.get("session_id")))
        return next_call()
    _execution_middleware(monkeypatch, audit)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert result["content"] == "Synthetic handoff."
        assert seen == [(len(calls[1]["messages"]), "context_prefix_request", "synthetic")]
    finally:
        client.close()


def test_a_blocking_execution_middleware_stops_the_request(monkeypatch):
    def block(request=None, next_call=None, **context):
        return None  # A policy middleware blocks by not calling next_call.
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, block)
        with pytest.raises(PrefixRequestError, match="incomplete_response"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_media_part_in_the_sent_rows_is_refused():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary["messages"][1]["content"] = [
            {"type": "text", "text": ordinary["messages"][1]["content"]},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="messages_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_named_user_row_sent_without_its_name_is_refused():
    # The transport drops ``name`` from tool rows only; a user row without its name lost its speaker.
    agent, calls, ordinary, client, history = make_agent()
    try:
        _tool_round(history, ordinary)
        history[0]["name"] = "alice"
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_rows_with_an_api_content_sidecar_are_sent_as_the_main_loop_sends_them():
    agent, calls, ordinary, client, history = make_agent()
    try:
        history[0]["api_content"] = "[recalled: short answers]\n\n" + history[0]["content"]
        ordinary["messages"][1]["content"] = history[0]["api_content"]
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "user", "content": "Next.", "api_content": "[note]\n\nNext."})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-2] == {"role": "user", "content": "[note]\n\nNext."}
    finally:
        client.close()


def test_a_sent_row_that_is_not_the_stored_api_content_is_refused():
    agent, calls, ordinary, client, history = make_agent()
    try:
        history[1]["api_content"] = "The answer is 4."
        ordinary["messages"][2]["content"] = "The answer is 5." + history[1]["content"]
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_the_request_runs_through_the_request_middleware(monkeypatch):
    seen = []

    def redact(request=None, **context):
        seen.append(context.get("purpose"))
        return {"request": json.loads(json.dumps(request).replace("SECRET", "[redacted]"))}
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(
        _middleware={"llm_request": [redact]}, has_middleware=lambda kind: kind == "llm_request",
        invoke_middleware=lambda kind, **kwargs: [redact(**kwargs)] if kind == "llm_request" else [],
        _report_hook_failure=lambda *args, **kwargs: None)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
        history.append({"role": "user", "content": "The key is SECRET."})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert seen == ["context_prefix_request"]
        assert "SECRET" not in json.dumps(calls[1]) and "The key is [redacted]." in json.dumps(calls[1])
    finally:
        client.close()


def test_appended_assistant_rows_get_reasoning_content_like_a_main_request():
    # A thinking-mode route (DeepSeek, Kimi) rejects an assistant row without reasoning_content.
    from agent.message_sanitization import apply_reasoning_content_policy
    agent, calls, ordinary, client, history = make_agent()
    agent._copy_reasoning_content_for_api = lambda source, target: apply_reasoning_content_policy(source, target, True)
    try:
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-2] == {"role": "assistant", "content": REPLY, "reasoning_content": " "}
    finally:
        client.close()


def test_changed_white_space_inside_a_sent_row_is_refused():
    # White space is API-visible: it can change code, tables, or commands.
    agent, calls, ordinary, client, history = make_agent()
    try:
        history[0]["content"] = "def f():\n    return 1"
        ordinary["messages"][1]["content"] = "def f():\n  return 1"
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_a_replacement_reply_from_execution_middleware_is_refused(monkeypatch):
    agent, calls, ordinary, client, history = make_agent()

    def replace(request=None, next_call=None, **context):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content="## Goal\nA handoff that the server did not write.", tool_calls=None, refusal=None))], usage=None)

    def change(request=None, next_call=None, **context):
        response = next_call()
        response.choices[0].message.content = "## Goal\nChanged."
        return response
    try:
        ordinary_turn(agent, ordinary, history)
        for middleware in (replace, change):
            _execution_middleware(monkeypatch, middleware)
            with pytest.raises(PrefixRequestError, match="middleware_changed_reply"):
                PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_appended_tool_calls_are_sanitized_like_a_main_request():
    # Strict providers reject call_id and response_item_id; only a Gemini model reads extra_content.
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "call_id": "call_1", "response_item_id": "fc_1", "type": "function",
             "extra_content": {"google": {"thought_signature": "sig"}},
             "function": {"name": "synthetic_tool", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": "call_1", "content": "ok"})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-3]["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "synthetic_tool", "arguments": "{}"}}]
    finally:
        client.close()


def test_a_request_middleware_that_changes_the_captured_part_is_refused(monkeypatch):
    # The captured request already went through llm_request middleware; a second pass would apply it twice.
    def prepend(request=None, **context):
        return {"request": {**request, "messages": [{"role": "system", "content": "policy"}, *request["messages"]]}}
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(
        _middleware={"llm_request": [prepend]}, has_middleware=lambda kind: kind == "llm_request",
        invoke_middleware=lambda kind, **kwargs: [prepend(**kwargs)] if kind == "llm_request" else [],
        _report_hook_failure=lambda *args, **kwargs: None)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
        with pytest.raises(PrefixRequestError, match="middleware_rewrite"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()

