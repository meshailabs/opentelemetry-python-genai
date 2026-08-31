# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock InvokeModel API instrumentation."""

from __future__ import annotations

import io
import json

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from botocore.stub import Stubber

from opentelemetry.instrumentation.genai.bedrock.extractors import (
    determine_invoke_model_operation_name,
    extract_invoke_model_request,
    extract_invoke_model_response,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


def test_invoke_model_anthropic_messages(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello!"}],
    }
    response_body = {
        "id": "msg_12345",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 4,
        },
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(request_body),
        )

    # Verify StreamingBody is preserved and readable
    assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat anthropic.claude-3-sonnet-20240229-v1:0"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "anthropic.claude-3-sonnet-20240229-v1:0"
    )
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "msg_12345"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40.0
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert (
        span.attributes[
            GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
        ]
        == 2
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == 4
    )
    assert (
        span.attributes[ServerAttributes.SERVER_ADDRESS]
        == "bedrock-runtime.us-east-1.amazonaws.com"
    )

    # Verify captured content
    input_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["content"] == "Hello!"

    sys_instruction = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    )
    assert sys_instruction[0]["content"] == "You are a helpful assistant."

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["parts"][0]["content"] == "Hi there!"


def test_invoke_model_anthropic_legacy_completion(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "prompt": "\n\nHuman: Tell me a joke\n\nAssistant:",
        "max_tokens_to_sample": 50,
        "temperature": 0.5,
        "stop_sequences": ["\n\nHuman:"],
    }
    response_body = {
        "completion": " Why did the chicken cross the road?",
        "stop_reason": "stop_sequence",
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-v2",
            "body": json.dumps(request_body).encode("utf-8"),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(request_body).encode("utf-8"),
        )

    assert response["body"].read() == raw_response_bytes
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "text_completion anthropic.claude-v2"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 50
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == (
        "\n\nHuman:",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_invoke_model_titan_text(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "inputText": "Write a poem",
        "textGenerationConfig": {
            "maxTokenCount": 200,
            "temperature": 0.6,
            "topP": 0.8,
            "stopSequences": ["User:"],
        },
    }
    response_body = {
        "inputTextTokenCount": 3,
        "results": [
            {
                "tokenCount": 25,
                "outputText": "Roses are red, violets are blue...",
                "completionReason": "FINISH",
            }
        ],
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "amazon.titan-text-express-v1",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        bedrock_client.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=json.dumps(request_body),
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "text_completion amazon.titan-text-express-v1"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.6
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 25
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_invoke_model_llama(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "prompt": "Explain quantum computing",
        "max_gen_len": 150,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    response_body = {
        "generation": "Quantum computing uses qubits...",
        "prompt_token_count": 5,
        "generation_token_count": 30,
        "stop_reason": "stop",
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "meta.llama3-8b-instruct-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        bedrock_client.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",
            body=json.dumps(request_body),
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "text_completion meta.llama3-8b-instruct-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 150
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 30
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_invoke_model_no_content(
    bedrock_client,
    instrument_no_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "messages": [{"role": "user", "content": "Sensitive secret"}],
        "max_tokens": 50,
    }
    response_body = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Sensitive answer"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(request_body),
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 10


def test_invoke_model_error(
    bedrock_client,
    instrument_bedrock,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    stubber.add_client_error(
        "invoke_model",
        service_error_code="ValidationException",
        service_message="Model identifier is invalid",
        expected_params={"modelId": "invalid-model-id", "body": b'{"prompt": "hi"}'},
    )

    with stubber:
        with pytest.raises(ClientError):
            bedrock_client.invoke_model(
                modelId="invalid-model-id",
                body=b'{"prompt": "hi"}',
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] in (
        "ValidationException",
        "botocore.errorfactory.ValidationException",
    )


def test_extract_invoke_model_response_headers(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    extract_invoke_model_response(
        {
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "X-Amzn-Bedrock-Input-Token-Count": "15",
                    "X-Amzn-Bedrock-Output-Token-Count": "22",
                }
            }
        },
        b'{"completion": "hello"}',
        invocation,
    )
    assert invocation.input_tokens == 15
    assert invocation.output_tokens == 22


def test_determine_invoke_model_operation_name() -> None:
    assert (
        determine_invoke_model_operation_name(
            {"body": json.dumps({"messages": [{"role": "user"}]})}
        )
        == "chat"
    )
    assert (
        determine_invoke_model_operation_name(
            {"body": json.dumps({"prompt": "Hello"})}
        )
        == "text_completion"
    )
    assert (
        determine_invoke_model_operation_name(
            {"body": json.dumps({"inputText": "Hello"})}
        )
        == "text_completion"
    )
    assert (
        determine_invoke_model_operation_name(
            {"body": json.dumps({"message": "Hello"})}
        )
        == "text_completion"
    )
