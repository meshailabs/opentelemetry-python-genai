# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock InvokeModelWithResponseStream API instrumentation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.genai.bedrock.patch import (
    _handle_invoke_model,
)
from opentelemetry.instrumentation.genai.bedrock.stream import (
    BedrockInvokeModelStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


def test_stream_wrapper_anthropic_messages(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="anthropic.claude-3-sonnet-20240229-v1:0",
        operation_name="chat",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "role": "assistant",
                            "usage": {"input_tokens": 14},
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": "Let me think...",
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {
                            "type": "text_delta",
                            "text": "Hello world!",
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 20},
                    }
                ).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    stream_events = list(wrapper)
    assert len(stream_events) == 4

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat anthropic.claude-3-sonnet-20240229-v1:0"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 14
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    parts = output_msgs[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["type"] == "reasoning"
    assert parts[0]["content"] == "Let me think..."
    assert parts[1]["type"] == "text"
    assert parts[1]["content"] == "Hello world!"


def test_stream_wrapper_titan_with_metrics(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "Here is ",
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "the response.",
                        "completionReason": "FINISH",
                        "amazon-bedrock-invocationMetrics": {
                            "inputTokenCount": 6,
                            "outputTokenCount": 12,
                        },
                    }
                ).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    stream_events = list(wrapper)
    assert len(stream_events) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.titan-text-lite-v1"
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 6
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 12
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["parts"][0]["content"] == "Here is the response."


def test_stream_wrapper_llama_and_mistral(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="meta.llama3-8b-instruct-v1:0",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {"generation": "Llama says hi", "stop_reason": "stop"}
                ).encode("utf-8")
            }
        }
    ]
    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert output_msgs[0]["parts"][0]["content"] == "Llama says hi"


def test_stream_wrapper_no_content(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "secret text",
                        "completionReason": "FINISH",
                        "amazon-bedrock-invocationMetrics": {
                            "inputTokenCount": 5,
                            "outputTokenCount": 10,
                        },
                    }
                ).encode("utf-8")
            }
        }
    ]
    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=False,
    )
    list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_stream_wrapper_caller_side_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps({"outputText": "chunk 1"}).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
    )

    with pytest.raises(RuntimeError, match="caller exploded"):
        with wrapper as stream:
            for _ in stream:
                raise RuntimeError("caller exploded")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


def test_stream_wrapper_stream_side_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )

    class FailingEventStream:
        def __iter__(self):
            yield {"chunk": {"bytes": b'{"outputText": "chunk"}'}}
            raise ConnectionError("connection reset")

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=FailingEventStream(),  # type: ignore[arg-type]
        invocation=invocation,
    )

    with pytest.raises(ConnectionError, match="connection reset"):
        list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


def test_handle_invoke_model_streaming_integration(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "Streamed text.",
                        "completionReason": "FINISH",
                    }
                ).encode("utf-8")
            }
        }
    ]
    mock_instance = MagicMock()
    mock_instance.meta.endpoint_url = (
        "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    mock_wrapped = MagicMock(return_value={"body": events})

    api_params = {
        "modelId": "amazon.titan-text-lite-v1",
        "body": json.dumps({"inputText": "hello"}),
    }

    response = _handle_invoke_model(
        mock_wrapped,
        mock_instance,
        ("InvokeModelWithResponseStream", api_params),
        {},
        api_params,
        handler,
        is_stream=True,
    )

    assert isinstance(response["body"], BedrockInvokeModelStreamWrapper)
    list(response["body"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat amazon.titan-text-lite-v1"
