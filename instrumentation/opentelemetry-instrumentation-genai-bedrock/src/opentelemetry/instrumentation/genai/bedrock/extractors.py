# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any, TypeGuard
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
)
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    BlobPart,
    FunctionToolDefinition,
    GenericPart,
    GenericToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
)


def _is_dict(val: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(val, dict)


def _is_list(val: object) -> TypeGuard[list[Any]]:
    return isinstance(val, list)


_FINISH_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "finish": "stop",
    "complete": "stop",
    "endoftext": "stop",
    "length": "length",
    "stop": "stop",
    "tool_calls": "tool_calls",
}

_DOC_MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "csv": "text/csv",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "txt": "text/plain",
    "md": "text/markdown",
}


def _safe_int(val: Any) -> int | None:
    """Safely convert a value to int or return None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def map_finish_reason(stop_reason: str | None) -> str | None:
    """Map Bedrock stopReason to GenAI semantic convention finish_reason."""
    if stop_reason is None:
        return None
    return _FINISH_REASON_MAP.get(stop_reason.lower(), stop_reason.lower())


def extract_server_address_and_port(
    endpoint_url: str | None,
) -> tuple[str | None, int | None]:
    """Parse server address and port from a client endpoint URL."""
    if not endpoint_url:
        return None, None
    parsed = urlparse(endpoint_url)
    server_address = parsed.hostname
    server_port = parsed.port
    if server_port is None and parsed.scheme in ("http", "https"):
        server_port = 443 if parsed.scheme == "https" else 80
    return server_address, server_port


def extract_content_block(block: dict[str, Any]) -> MessagePart | None:
    """Map a single Bedrock content block to an OpenTelemetry MessagePart."""
    if "text" in block:
        return TextPart(content=block["text"])

    reasoning = block.get("reasoningContent")
    if _is_dict(reasoning):
        reasoning_text = reasoning.get("reasoningText")
        if _is_dict(reasoning_text) and "text" in reasoning_text:
            return ReasoningPart(content=reasoning_text["text"])
        if "redactedContent" in reasoning:
            return ReasoningPart(content="")

    image = block.get("image")
    if _is_dict(image):
        fmt = image.get("format", "jpeg")
        source = image.get("source")
        content_bytes = source.get("bytes", b"") if _is_dict(source) else b""
        return BlobPart(
            content=content_bytes,
            mime_type=f"image/{fmt}",
            modality="image",
        )

    document = block.get("document")
    if _is_dict(document):
        fmt = document.get("format", "pdf")
        source = document.get("source")
        content_bytes = source.get("bytes", b"") if _is_dict(source) else b""
        mime_type = _DOC_MIME_TYPES.get(fmt, f"application/{fmt}")
        return BlobPart(
            content=content_bytes,
            mime_type=mime_type,
            modality="document",
        )

    tool_use = block.get("toolUse")
    if _is_dict(tool_use):
        return ToolCallRequestPart(
            id=tool_use.get("toolUseId"),
            name=tool_use.get("name", ""),
            arguments=tool_use.get("input"),
        )

    tool_result = block.get("toolResult")
    if _is_dict(tool_result):
        return ToolCallResponsePart(
            id=tool_result.get("toolUseId"),
            response=tool_result.get("content"),
        )

    for key in (
        "video",
        "audio",
        "guardContent",
        "cachePoint",
        "citationsContent",
        "searchResult",
        "toolAddition",
        "toolRemoval",
    ):
        if key in block:
            return GenericPart(type=key, value=None)

    return None


def extract_converse_request(
    kwargs: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    """Populate request attributes from converse kwargs onto the invocation."""
    inf_config = kwargs.get("inferenceConfig")
    if _is_dict(inf_config):
        invocation.temperature = inf_config.get("temperature")
        invocation.top_p = inf_config.get("topP")
        invocation.max_tokens = inf_config.get("maxTokens")
        invocation.stop_sequences = inf_config.get("stopSequences")
        invocation.top_k = inf_config.get("topK") or inf_config.get("top_k")
        invocation.seed = inf_config.get("seed")

    add_fields = kwargs.get("additionalModelRequestFields")
    if _is_dict(add_fields):
        add_inf = add_fields.get("inferenceConfig")
        top_k = (
            add_fields.get("topK")
            or add_fields.get("top_k")
            or (
                (add_inf.get("topK") or add_inf.get("top_k"))
                if _is_dict(add_inf)
                else None
            )
        )
        if top_k is not None:
            invocation.top_k = top_k
        if "seed" in add_fields:
            invocation.seed = add_fields["seed"]

    # system instruction
    raw_system = kwargs.get("system")
    if capture_content and _is_list(raw_system):
        system_parts: list[MessagePart] = []
        for item in raw_system:
            if _is_dict(item):
                part = extract_content_block(item)
                if part is not None:
                    system_parts.append(part)
        if system_parts:
            invocation.system_instruction = system_parts

    # input messages
    raw_messages = kwargs.get("messages")
    if capture_content and _is_list(raw_messages):
        input_messages: list[InputMessage] = []
        for msg in raw_messages:
            if not _is_dict(msg):
                continue
            role = msg.get("role", "user")
            parts: list[MessagePart] = []
            content = msg.get("content")
            if _is_list(content):
                for block in content:
                    if _is_dict(block):
                        part = extract_content_block(block)
                        if part is not None:
                            parts.append(part)
            input_messages.append(InputMessage(role=role, parts=parts))
        invocation.input_messages = input_messages

    # tool definitions
    tool_config = kwargs.get("toolConfig")
    if _is_dict(tool_config):
        tools = tool_config.get("tools")
        if _is_list(tools):
            tool_defs: list[ToolDefinition] = []
            for tool in tools:
                if not _is_dict(tool):
                    continue
                tool_spec = tool.get("toolSpec")
                if _is_dict(tool_spec):
                    name = tool_spec.get("name", "")
                    description = tool_spec.get("description")
                    raw_schema = tool_spec.get("inputSchema", {})
                    params = (
                        raw_schema.get("json", raw_schema)
                        if _is_dict(raw_schema)
                        else raw_schema
                    )
                    tool_defs.append(
                        FunctionToolDefinition(
                            name=name,
                            description=description,
                            parameters=params,
                        )
                    )
                elif "name" in tool and "type" in tool:
                    tool_defs.append(
                        GenericToolDefinition(
                            name=tool["name"],
                            type=tool["type"],
                        )
                    )
            invocation.tool_definitions = tool_defs


def extract_converse_response(
    response: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    stop_reason = response.get("stopReason")
    finish_reason = map_finish_reason(stop_reason)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]

    output = response.get("output")
    if capture_content and _is_dict(output):
        msg = output.get("message")
        if _is_dict(msg):
            role = msg.get("role", "assistant")
            parts: list[MessagePart] = []
            content = msg.get("content")
            if _is_list(content):
                for block in content:
                    if _is_dict(block):
                        part = extract_content_block(block)
                        if part is not None:
                            parts.append(part)
            invocation.output_messages = [
                OutputMessage(
                    role=role,
                    parts=parts,
                    finish_reason=finish_reason or "stop",
                )
            ]

    usage = response.get("usage")
    if _is_dict(usage):
        invocation.input_tokens = usage.get("inputTokens")
        invocation.output_tokens = usage.get("outputTokens")
        invocation.cache_read_input_tokens = usage.get("cacheReadInputTokens")
        invocation.cache_creation_input_tokens = usage.get(
            "cacheWriteInputTokens"
        )


def _parse_body(body: Any) -> dict[str, Any] | None:
    """Safely parse body as a dictionary."""
    if _is_dict(body):
        return body
    if isinstance(body, (bytes, bytearray)):
        try:
            parsed: object = json.loads(body.decode("utf-8"))
            if _is_dict(parsed):
                return parsed
        except Exception:
            return None
    elif isinstance(body, str):
        try:
            parsed = json.loads(body)
            if _is_dict(parsed):
                return parsed
        except Exception:
            return None
    elif hasattr(body, "read"):
        try:
            content: object = body.read()
            if hasattr(body, "seek"):
                body.seek(0)
            if isinstance(content, (bytes, bytearray)):
                parsed = json.loads(content.decode("utf-8"))
                if _is_dict(parsed):
                    return parsed
            elif isinstance(content, str):
                parsed = json.loads(content)
                if _is_dict(parsed):
                    return parsed
        except Exception:
            return None
    return None


def determine_invoke_model_operation_name(api_params: dict[str, Any]) -> str:
    """Determine whether InvokeModel call represents chat or text_completion."""
    body = _parse_body(api_params.get("body"))
    if _is_dict(body):
        if "messages" in body:
            return GenAiOperationNameValues.CHAT.value
        if "prompt" in body or "inputText" in body or "message" in body:
            return GenAiOperationNameValues.TEXT_COMPLETION.value
    return GenAiOperationNameValues.TEXT_COMPLETION.value


def extract_invoke_model_request(
    api_params: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    """Populate request attributes from InvokeModel api_params onto the invocation."""
    body = _parse_body(api_params.get("body"))
    if not _is_dict(body):
        return

    # Temperature
    temp = (
        body.get("temperature")
        or (
            body.get("textGenerationConfig", {}).get("temperature")
            if _is_dict(body.get("textGenerationConfig"))
            else None
        )
        or (
            body.get("inferenceConfig", {}).get("temperature")
            if _is_dict(body.get("inferenceConfig"))
            else None
        )
    )
    if temp is not None:
        try:
            invocation.temperature = float(temp)
        except (ValueError, TypeError):
            pass

    # Top P
    top_p = (
        body.get("top_p")
        or body.get("topP")
        or body.get("p")
        or (
            body.get("textGenerationConfig", {}).get("topP")
            if _is_dict(body.get("textGenerationConfig"))
            else None
        )
        or (
            body.get("inferenceConfig", {}).get("top_p")
            if _is_dict(body.get("inferenceConfig"))
            else None
        )
    )
    if top_p is not None:
        try:
            invocation.top_p = float(top_p)
        except (ValueError, TypeError):
            pass

    # Top K
    top_k = (
        body.get("top_k")
        or body.get("topK")
        or body.get("k")
        or (
            body.get("inferenceConfig", {}).get("top_k")
            if _is_dict(body.get("inferenceConfig"))
            else None
        )
    )
    if top_k is not None:
        try:
            invocation.top_k = float(top_k)
        except (ValueError, TypeError):
            pass

    # Max tokens
    max_tokens = _safe_int(
        body.get("max_tokens")
        or body.get("max_tokens_to_sample")
        or body.get("max_gen_len")
        or body.get("maxTokens")
        or (
            body.get("textGenerationConfig", {}).get("maxTokenCount")
            if _is_dict(body.get("textGenerationConfig"))
            else None
        )
        or (
            body.get("inferenceConfig", {}).get("max_new_tokens")
            if _is_dict(body.get("inferenceConfig"))
            else None
        )
    )
    if max_tokens is not None:
        invocation.max_tokens = max_tokens

    # Stop sequences
    stop_seqs = (
        body.get("stop_sequences")
        or body.get("stopSequences")
        or (
            body.get("textGenerationConfig", {}).get("stopSequences")
            if _is_dict(body.get("textGenerationConfig"))
            else None
        )
    )
    if _is_list(stop_seqs):
        invocation.stop_sequences = [str(s) for s in stop_seqs]

    # Seed
    seed = _safe_int(body.get("seed"))
    if seed is not None:
        invocation.seed = seed

    # Tool definitions (e.g. Anthropic format)
    raw_tools = body.get("tools")
    if _is_list(raw_tools):
        tool_defs: list[ToolDefinition] = []
        for tool in raw_tools:
            if not _is_dict(tool):
                continue
            name = tool.get("name", "")
            description = tool.get("description")
            params = tool.get("input_schema") or tool.get("parameters")
            if params is not None:
                tool_defs.append(
                    FunctionToolDefinition(
                        name=name,
                        description=description,
                        parameters=params if _is_dict(params) else {},
                    )
                )
            elif name and "type" in tool:
                tool_defs.append(
                    GenericToolDefinition(name=name, type=tool["type"])
                )
        if tool_defs:
            invocation.tool_definitions = tool_defs

    if not capture_content:
        return

    # System instruction (e.g. Anthropic / Nova)
    raw_system = body.get("system")
    if raw_system:
        if isinstance(raw_system, str):
            invocation.system_instruction = [Text(content=raw_system)]
        elif _is_list(raw_system):
            system_parts: list[MessagePart] = []
            for item in raw_system:
                if isinstance(item, str):
                    system_parts.append(Text(content=item))
                elif _is_dict(item):
                    part = extract_content_block(item)
                    if part is not None:
                        system_parts.append(part)
            if system_parts:
                invocation.system_instruction = system_parts

    # Input messages / prompt
    if "messages" in body and _is_list(body["messages"]):
        input_messages: list[InputMessage] = []
        for msg in body["messages"]:
            if not _is_dict(msg):
                continue
            role = msg.get("role", "user")
            content = msg.get("content")
            parts: list[MessagePart] = []
            if isinstance(content, str):
                parts.append(Text(content=content))
            elif _is_list(content):
                for block in content:
                    if isinstance(block, str):
                        parts.append(Text(content=block))
                    elif _is_dict(block):
                        part = extract_content_block(block)
                        if part is not None:
                            parts.append(part)
            input_messages.append(InputMessage(role=role, parts=parts))
        if input_messages:
            invocation.input_messages = input_messages
    elif "prompt" in body and isinstance(body["prompt"], str):
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[Text(content=body["prompt"])],
            )
        ]
    elif "inputText" in body and isinstance(body["inputText"], str):
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[Text(content=body["inputText"])],
            )
        ]
    elif "message" in body and isinstance(body["message"], str):
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[Text(content=body["message"])],
            )
        ]


def extract_invoke_model_response(
    response: dict[str, Any],
    raw_body_bytes: bytes,
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    """Populate response attributes from InvokeModel response."""
    # 1. Token counts from response headers (case-insensitive)
    resp_meta = response.get("ResponseMetadata")
    http_headers = (
        resp_meta.get("HTTPHeaders") if _is_dict(resp_meta) else None
    )
    if _is_dict(http_headers):
        headers_lower: dict[str, str] = {
            str(k).lower(): str(v) for k, v in http_headers.items()
        }
        invocation.input_tokens = _safe_int(
            headers_lower.get("x-amzn-bedrock-input-token-count")
        )
        invocation.output_tokens = _safe_int(
            headers_lower.get("x-amzn-bedrock-output-token-count")
        )

    body = _parse_body(raw_body_bytes)
    if not _is_dict(body):
        return

    # 2. Token counts from payload if not in headers
    usage = body.get("usage")
    if _is_dict(usage):
        if invocation.input_tokens is None:
            invocation.input_tokens = _safe_int(
                usage.get("input_tokens") or usage.get("inputTokens")
            )
        if invocation.output_tokens is None:
            invocation.output_tokens = _safe_int(
                usage.get("output_tokens") or usage.get("outputTokens")
            )
        invocation.cache_read_input_tokens = _safe_int(
            usage.get("cache_read_input_tokens")
            or usage.get("cacheReadInputTokens")
        )
        invocation.cache_creation_input_tokens = _safe_int(
            usage.get("cache_creation_input_tokens")
            or usage.get("cacheWriteInputTokens")
        )

    if invocation.input_tokens is None and "inputTextTokenCount" in body:
        invocation.input_tokens = _safe_int(body.get("inputTextTokenCount"))

    results = body.get("results")
    if _is_list(results) and results and _is_dict(results[0]):
        if invocation.output_tokens is None:
            invocation.output_tokens = _safe_int(results[0].get("tokenCount"))

    if invocation.input_tokens is None and "prompt_token_count" in body:
        invocation.input_tokens = _safe_int(body.get("prompt_token_count"))
    if invocation.output_tokens is None and "generation_token_count" in body:
        invocation.output_tokens = _safe_int(
            body.get("generation_token_count")
        )

    # 3. Finish reasons
    raw_finish_reason: str | None = None
    if "stop_reason" in body and isinstance(body["stop_reason"], str):
        raw_finish_reason = body["stop_reason"]
    elif "stopReason" in body and isinstance(body["stopReason"], str):
        raw_finish_reason = body["stopReason"]
    elif _is_list(results) and results and _is_dict(results[0]):
        raw_finish_reason = results[0].get("completionReason")
    elif (
        "outputs" in body
        and _is_list(body["outputs"])
        and body["outputs"]
        and _is_dict(body["outputs"][0])
    ):
        raw_finish_reason = body["outputs"][0].get("stop_reason")
    elif (
        "generations" in body
        and _is_list(body["generations"])
        and body["generations"]
        and _is_dict(body["generations"][0])
    ):
        raw_finish_reason = body["generations"][0].get("finish_reason")
    elif (
        "completions" in body
        and _is_list(body["completions"])
        and body["completions"]
        and _is_dict(body["completions"][0])
    ):
        finish_obj = body["completions"][0].get("finishReason")
        if _is_dict(finish_obj):
            raw_finish_reason = finish_obj.get("reason")

    finish_reason = map_finish_reason(raw_finish_reason)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]

    # Response ID (e.g. Anthropic msg_...)
    if "id" in body and isinstance(body["id"], str):
        invocation.response_id = body["id"]

    # 4. Content capture
    if not capture_content:
        return

    # Anthropic Messages format
    if "content" in body and _is_list(body["content"]):
        parts: list[MessagePart] = []
        for block in body["content"]:
            if isinstance(block, str):
                parts.append(Text(content=block))
            elif _is_dict(block):
                part = extract_content_block(block)
                if part is not None:
                    parts.append(part)
        role = body.get("role", "assistant")
        invocation.output_messages = [
            OutputMessage(
                role=role,
                parts=parts,
                finish_reason=finish_reason or "stop",
            )
        ]
    # Amazon Nova format
    elif (
        "output" in body
        and _is_dict(body["output"])
        and _is_dict(body["output"].get("message"))
    ):
        msg = body["output"]["message"]
        role = msg.get("role", "assistant")
        content = msg.get("content")
        nova_parts: list[MessagePart] = []
        if _is_list(content):
            for block in content:
                if _is_dict(block):
                    part = extract_content_block(block)
                    if part is not None:
                        nova_parts.append(part)
        invocation.output_messages = [
            OutputMessage(
                role=role,
                parts=nova_parts,
                finish_reason=finish_reason or "stop",
            )
        ]
    # Anthropic Legacy completion
    elif "completion" in body and isinstance(body["completion"], str):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=body["completion"])],
                finish_reason=finish_reason or "stop",
            )
        ]
    # Titan outputText
    elif (
        _is_list(results)
        and results
        and _is_dict(results[0])
        and "outputText" in results[0]
    ):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=str(results[0]["outputText"]))],
                finish_reason=finish_reason or "stop",
            )
        ]
    # Llama generation
    elif "generation" in body and isinstance(body["generation"], str):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=body["generation"])],
                finish_reason=finish_reason or "stop",
            )
        ]
    # Mistral outputs
    elif (
        "outputs" in body
        and _is_list(body["outputs"])
        and body["outputs"]
        and _is_dict(body["outputs"][0])
        and "text" in body["outputs"][0]
    ):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=str(body["outputs"][0]["text"]))],
                finish_reason=finish_reason or "stop",
            )
        ]
    # Cohere generations
    elif (
        "generations" in body
        and _is_list(body["generations"])
        and body["generations"]
        and _is_dict(body["generations"][0])
        and "text" in body["generations"][0]
    ):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=str(body["generations"][0]["text"]))],
                finish_reason=finish_reason or "stop",
            )
        ]
    # AI21 completions
    elif (
        "completions" in body
        and _is_list(body["completions"])
        and body["completions"]
        and _is_dict(body["completions"][0])
    ):
        data = body["completions"][0].get("data")
        if _is_dict(data) and "text" in data:
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=str(data["text"]))],
                    finish_reason=finish_reason or "stop",
                )
            ]
