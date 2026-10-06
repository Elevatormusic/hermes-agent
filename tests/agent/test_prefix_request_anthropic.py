"""The same-prefix request on the Anthropic Messages route: capture, request shape, refusals, and the reply,
with the real Anthropic SDK against a recording HTTP sink and the main loop's own request builder."""

import copy
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from agent.anthropic_adapter import create_anthropic_message
from agent.prefix_request import (
    PrefixRequest, PrefixRequestError, begin_capture, capture_response, publish_response,
)
from agent.prompt_caching import apply_anthropic_cache_control
from agent.transports.anthropic import AnthropicTransport

REPLY = "Completed synthetic reply."
HANDOFF = "## Goal\nSynthetic goal."
INSTRUCTION = "Write the handoff."
TOOLS = [{"type": "function", "function": {"name": "synthetic_tool", "description": "A synthetic tool.",
                                           "parameters": {"type": "object", "properties": {}}}}]


def _message(content, stop="end_turn", usage=None):
    return {"id": "msg_synthetic", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": usage or {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 4800,
                               "cache_creation_input_tokens": 0}}


def make_agent(handoff_blocks=None, handoff_stop="end_turn", is_oauth=False):
    calls = []

    def sink(request):
        calls.append({"body": json.loads(request.content), "beta": request.headers.get("anthropic-beta")})
        if len(calls) == 1:
            return httpx.Response(200, request=request, json=_message([{"type": "text", "text": REPLY}]))
        blocks = handoff_blocks if handoff_blocks is not None else [{"type": "text", "text": HANDOFF}]
        return httpx.Response(200, request=request, json=_message(blocks, stop=handoff_stop))

    client = anthropic.Anthropic(api_key="synthetic-no-secret", base_url="https://api.anthropic.com", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False))
    history = []
    for index in range(4):
        history.extend([{"role": "user", "content": f"{index} data " * 50},
                        {"role": "assistant", "content": "result " * 50}])
    history.append({"role": "user", "content": "Continue."})
    transport = AnthropicTransport()
    agent = SimpleNamespace(
        client=None, context_compressor=SimpleNamespace(wants_prefix_request=True, context_length=200000),
        session_id="synthetic", model="claude-opus-5-5", provider="anthropic", base_url="https://api.anthropic.com",
        api_mode="anthropic_messages", tools=TOOLS, reasoning_config=None,
        _cached_system_prompt="Keep the synthetic task.", _client_kwargs={},
        _anthropic_base_url="https://api.anthropic.com", _is_anthropic_oauth=is_oauth,
        _anthropic_preserve_dots=lambda: False, _oauth_1m_beta_disabled=False, _disable_streaming=True,
        _prefix_source_messages=history, _session_messages=history, _get_transport=lambda: transport,
        _prepare_anthropic_messages_for_api=lambda messages: messages,
        _create_request_anthropic_client=lambda **kwargs: client,
        _close_request_anthropic_client=lambda *args, **kwargs: None)
    return agent, calls, client, history


def ordinary_kwargs(agent, history, **overrides):
    """The main loop's request: cache decoration on the OpenAI-form rows, then the Anthropic builder."""
    api_messages = apply_anthropic_cache_control(
        [{"role": "system", "content": agent._cached_system_prompt}, *copy.deepcopy(history)], native_anthropic=True)
    params = dict(max_tokens=4096, reasoning_config=agent.reasoning_config, is_oauth=agent._is_anthropic_oauth,
                  context_length=200000, base_url=agent._anthropic_base_url)
    params.update(overrides)
    return agent._get_transport().build_kwargs(model=agent.model, messages=api_messages, tools=agent.tools, **params)


def ordinary_turn(agent, history, kwargs=None):
    kwargs = kwargs if kwargs is not None else ordinary_kwargs(agent, history)
    begin_capture(agent, kwargs)
    response = create_anthropic_message(agent._create_request_anthropic_client(reason="test"),
                                        {**copy.deepcopy(kwargs), "timeout": 30}, prefer_stream=False)
    publish_response(agent, capture_response(agent, kwargs, response))
    history.append({"role": "assistant", "content": REPLY})
    return kwargs


def text_of(row):
    content = row["content"]
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


def test_request_keeps_the_captured_prefix_with_its_breakpoints_and_appends_reply_and_instruction():
    agent, calls, client, history = make_agent()
    try:
        ordinary_turn(agent, history)
        before = copy.deepcopy(history)
        request = PrefixRequest(agent, history)
        assert request.cache_read_tokens == 4800
        result = request(INSTRUCTION, timeout_s=30)
        assert len(calls) == 2
        first, second = calls[0]["body"], calls[1]["body"]
        count = len(first["messages"])
        # The captured messages go back byte for byte, cache_control breakpoints included: the server reads them
        # from its prompt cache.
        assert second["messages"][:count] == first["messages"]
        assert any("cache_control" in json.dumps(row) for row in second["messages"][:count])
        added = second["messages"][count:]
        assert [row["role"] for row in added] == ["assistant", "user"]
        assert text_of(added[0]) == REPLY and "cache_control" not in json.dumps(added)
        assert added[-1]["content"][-1] == {"type": "text", "text": INSTRUCTION}
        assert {k: v for k, v in second.items() if k not in {"messages", "max_tokens"}} == {
            k: v for k, v in first.items() if k not in {"messages", "max_tokens"}}
        assert second["max_tokens"] == 4096 and "stream" not in second
        assert history == before
        assert result["content"] == HANDOFF and result["finish_reason"] == "stop"
        assert result["tool_calls"] is False and result["refusal"] is False
        # input_tokens excludes the cached prompt: the prompt count adds the cache read and write.
        assert result["usage"] == {"prompt_tokens": 4812, "completion_tokens": 5, "cache_read_tokens": 4800}
    finally:
        client.close()


def test_tool_results_after_the_capture_carry_the_instruction_as_their_last_block():
    agent, calls, client, history = make_agent()
    try:
        ordinary_turn(agent, history)
        history[-1] = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "toolu_1", "type": "function", "function": {"name": "synthetic_tool", "arguments": "{}"}}]}
        history.append({"role": "tool", "tool_call_id": "toolu_1", "content": "tool output"})
        PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
        added = calls[1]["body"]["messages"][len(calls[0]["body"]["messages"]):]
        assert [row["role"] for row in added] == ["assistant", "user"]
        assert [part["type"] for part in added[0]["content"]] == ["tool_use"]
        assert [part["type"] for part in added[1]["content"]] == ["tool_result", "text"]
        assert added[1]["content"][-1] == {"type": "text", "text": INSTRUCTION}
    finally:
        client.close()


def test_claude_code_oauth_naming_is_rendered_the_same_way():
    agent, calls, client, history = make_agent(is_oauth=True)
    try:
        history[-2] = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "toolu_0", "type": "function", "function": {"name": "synthetic_tool", "arguments": "{}"}}]}
        history.insert(len(history) - 1, {"role": "tool", "tool_call_id": "toolu_0", "content": "earlier output"})
        ordinary_turn(agent, history)
        PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
        first, second = calls[0]["body"], calls[1]["body"]
        assert second["messages"][:len(first["messages"])] == first["messages"]
        assert second["system"] == first["system"] and second["tools"] == first["tools"]
    finally:
        client.close()


def test_thinking_settings_stay_and_the_handoff_gets_the_thinking_limit():
    agent, calls, client, history = make_agent()
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    try:
        kwargs = ordinary_kwargs(agent, history, max_tokens=128000)
        assert kwargs.get("thinking"), "the builder must enable thinking for this test to mean anything"
        ordinary_turn(agent, history, kwargs)
        PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
        first, second = calls[0]["body"], calls[1]["body"]
        assert second["thinking"] == first["thinking"]
        assert second.get("output_config") == first.get("output_config")
        assert second["max_tokens"] == 32768
    finally:
        client.close()


def test_a_fast_mode_beta_header_is_replayed():
    agent, calls, client, history = make_agent()
    try:
        kwargs = ordinary_kwargs(agent, history)
        kwargs["extra_headers"] = {"anthropic-beta": "fast-mode-2026-02-01"}
        kwargs["extra_body"] = {"speed": "fast"}
        ordinary_turn(agent, history, kwargs)
        PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
        assert calls[1]["beta"] == calls[0]["beta"] == "fast-mode-2026-02-01"
        assert calls[1]["body"]["speed"] == "fast"
    finally:
        client.close()


def test_a_tool_use_reply_is_reported_as_a_tool_call():
    blocks = [{"type": "tool_use", "id": "toolu_9", "name": "synthetic_tool", "input": {}}]
    agent, calls, client, history = make_agent(handoff_blocks=blocks, handoff_stop="tool_use")
    try:
        ordinary_turn(agent, history)
        result = PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
        assert result["tool_calls"] is True and result["finish_reason"] == "tool_calls"
        assert result["content"] is None
    finally:
        client.close()


def test_thinking_blocks_are_not_reply_text():
    blocks = [{"type": "thinking", "thinking": "hidden", "signature": "sig"}, {"type": "text", "text": HANDOFF}]
    agent, calls, client, history = make_agent(handoff_blocks=blocks)
    try:
        ordinary_turn(agent, history)
        assert PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)["content"] == HANDOFF
    finally:
        client.close()


@pytest.mark.parametrize("change, reason", [
    # A hook or middleware rewrote a captured row: the handoff would summarize text that is not in the history.
    (lambda body: body["messages"][0]["content"].__setitem__(0, {"type": "text", "text": "Edited."}),
     "source_transform_unsupported"),
    (lambda body: body.__setitem__("tool_choice", {"type": "any"}), "settings_unsupported"),
    (lambda body: body.__setitem__("container", "synthetic"), "settings_unsupported"),
    (lambda body: body.__setitem__("messages", []), "messages_unsupported"),
])
def test_refusals_send_nothing(change, reason):
    agent, calls, client, history = make_agent()
    try:
        ordinary_turn(agent, history)
        first = agent._prefix_capsule["body"]["messages"][0]
        if isinstance(first.get("content"), str):
            first["content"] = [{"type": "text", "text": first["content"]}]
        change(agent._prefix_capsule["body"])
        with pytest.raises(PrefixRequestError, match=reason):
            PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_history_that_no_longer_starts_with_the_capture_is_refused():
    agent, calls, client, history = make_agent()
    try:
        ordinary_turn(agent, history)
        history[0] = {"role": "user", "content": "Edited."}
        with pytest.raises(PrefixRequestError, match="history_changed"):
            PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_reply_that_did_not_finish_is_not_published():
    agent, calls, client, history = make_agent()
    try:
        kwargs = ordinary_kwargs(agent, history)
        begin_capture(agent, kwargs)
        response = anthropic.types.Message.model_validate(_message([{"type": "text", "text": "cut"}], stop="max_tokens"))
        publish_response(agent, capture_response(agent, kwargs, response))
        assert agent._prefix_capsule is None
        with pytest.raises(PrefixRequestError, match="no_capture"):
            PrefixRequest(agent, history)(INSTRUCTION, timeout_s=30)
    finally:
        client.close()
