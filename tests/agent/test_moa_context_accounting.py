"""MoA context pressure uses one request; billing includes every model attempt."""

import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("finish_reason", ["stop", "length", "content_filter"])
def test_moa_response_accounts_spend_without_inflating_context(finish_reason, monkeypatch):
    from agent.moa_loop import MoAClient
    from agent.turn_response_check import check_api_response
    from agent.turn_retry_state import TurnRetryState
    from agent.usage_pricing import CanonicalUsage
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", provider="custom", model="test/model",
        base_url="http://localhost:12345/v1", api_mode="chat_completions",
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False,
    )
    agent.provider = "moa"
    agent.client = MoAClient("test")
    advisor_usage = CanonicalUsage(input_tokens=230_000, output_tokens=2_000)
    agent.client.chat.completions._pending_reference_usage = advisor_usage
    compressor = agent.context_compressor
    compressor.context_length = 262_144
    compressor.threshold_tokens = 180_000
    compressor._verify_compaction_cleared_threshold = True
    messages = [{"role": "user", "content": "Continue the task."}]
    response = SimpleNamespace(
        id="test-response", model="test/model",
        choices=[SimpleNamespace(
            index=0, finish_reason=finish_reason,
            message=SimpleNamespace(content="Partial answer.", tool_calls=[], refusal=None),
        )],
        usage=SimpleNamespace(prompt_tokens=8_000, completion_tokens=100, total_tokens=8_100),
    )
    monkeypatch.setattr(agent, "_persist_session", lambda *args: None)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *args: None)
    try:
        verdict = check_api_response(
            agent, response=response, _retry=TurnRetryState(), thinking_spinner=None,
            messages=messages, api_messages=list(messages), api_kwargs={},
            active_system_prompt="", conversation_history=[], finish_reason=None,
            retry_count=0, max_retries=3, compression_attempts=3,
            max_compression_attempts=3, length_continue_retries=0,
            truncated_response_parts=[], truncated_tool_call_retries=0,
            current_turn_user_idx=0, api_call_count=1, api_request_id="test-request",
            api_start_time=time.time(), effective_task_id="test-task", turn_id="test-turn",
            _preflight_compression_blocked=True, _last_preflight_pressure=238_000,
        )
        assert compressor.last_prompt_tokens == 8_000
        assert compressor.last_completion_tokens == 100
        assert agent._last_turn_usage["total_tokens"] == 8_100
        assert agent.session_prompt_tokens == 8_000 + advisor_usage.prompt_tokens
        assert agent.session_output_tokens == 100 + advisor_usage.output_tokens
        assert agent.session_api_calls == 1
        assert agent.client.consume_reference_usage()[0].total_tokens == 0
        assert verdict.compression_attempts == 0
        assert verdict._preflight_compression_blocked is False
        assert verdict._last_preflight_pressure is None
    finally:
        agent.close()


@pytest.mark.parametrize("task", ["moa_reference", "moa_aggregator"])
@pytest.mark.parametrize("model, cap_key", [
    ("custom-model", "max_tokens"), ("gpt-5", "max_completion_tokens"),
])
def test_moa_request_builder_preserves_explicit_output_budget(task, model, cap_key):
    from agent.auxiliary_client import _build_call_kwargs

    messages = [{"role": "user", "content": "Continue the answer."}]
    for cap in (None, 8_192, 16_384, 32_768):
        request = _build_call_kwargs(
            "custom", model, messages, max_tokens=cap,
            base_url="https://model.example/v1", task=task,
        )
        assert request.get(cap_key) == cap
        other_key = "max_completion_tokens" if cap_key == "max_tokens" else "max_tokens"
        assert other_key not in request
