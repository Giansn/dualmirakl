"""Proof-run: Authority chat completion via gateway proxy.

Verifies that POST /v1/chat/completions with model="authority" returns a
well-formed OpenAI-compatible response, then repeats the call with
stream=true to confirm SSE chunks arrive correctly.

Usage:
    python tests/proof_run/test_proof_authority_chat.py
"""

import json
import ssl
import sys
import time
import urllib.request

BASE_URL = "https://48acx0kqem74jt-9000.proxy.runpod.net"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"

# Accept any TLS certificate (RunPod proxy may use self-signed)
SSL_CTX = ssl._create_unverified_context()

results: list[bool] = []
def check(label: str, condition: bool, detail: str = "") -> bool:
    results.append(condition)
    if condition:
        print(f"  [PASS] {label}")
    else:
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
    return condition
def _post(payload: dict) -> urllib.request.Request:
    return urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "dualmirakl-proof-run"},
        method="POST",
    )
CHAT_PAYLOAD = {
    "model": "authority",
    "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
    "max_tokens": 50,
    "temperature": 0.1,
}

# ------------------------------------------------------------------
# 1. Non-streaming chat completion
# ------------------------------------------------------------------
print("=== Authority chat completion (non-streaming) ===")

t0 = time.perf_counter()
try:
    with urllib.request.urlopen(_post(CHAT_PAYLOAD), context=SSL_CTX, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    latency_ms = (time.perf_counter() - t0) * 1000
except Exception as exc:
    print(f"  [FAIL] Request failed: {exc}")
    sys.exit(1)

choices = body.get("choices")
check(
    "Response has 'choices' array with >= 1 entry",
    isinstance(choices, list) and len(choices) >= 1,
)

choice = (choices or [{}])[0]
message = choice.get("message", {})
content = message.get("content")

check(
    "choices[0].message.content is a non-empty string",
    isinstance(content, str) and len(content.strip()) > 0,
    detail=f"content={content!r}",
)

check(
    "choices[0].message.role == 'assistant'",
    message.get("role") == "assistant",
    detail=f"role={message.get('role')!r}",
)

check(
    "Response has 'model' field",
    "model" in body and isinstance(body["model"], str),
    detail=f"model={body.get('model')!r}",
)

check(
    "Response has 'usage' field",
    "usage" in body and isinstance(body["usage"], dict),
    detail=f"usage={body.get('usage')}",
)

print(f"  Latency: {latency_ms:.0f} ms")
print(f"  Content: {content!r}")

# ------------------------------------------------------------------
# 2. Streaming chat completion
# ------------------------------------------------------------------
print("\n=== Authority chat completion (streaming) ===")

sse_count = 0
chunks_with_content = 0

t0 = time.perf_counter()
try:
    with urllib.request.urlopen(
        _post({**CHAT_PAYLOAD, "stream": True}), context=SSL_CTX, timeout=120
    ) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
            if not line.startswith("data: "):
                continue
            sse_count += 1
            payload_str = line[len("data: "):]
            if payload_str.strip() == "[DONE]":
                continue
            try:
                delta = json.loads(payload_str)["choices"][0].get("delta", {})
                if delta.get("content"):
                    chunks_with_content += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    stream_latency_ms = (time.perf_counter() - t0) * 1000
except Exception as exc:
    print(f"  [FAIL] Streaming request failed: {exc}")
    sys.exit(1)

check(
    "Received SSE 'data:' lines",
    sse_count > 0,
    detail=f"got {sse_count} data lines",
)

check(
    "At least one chunk has delta.content",
    chunks_with_content > 0,
    detail=f"{chunks_with_content} content chunks",
)

print(f"  Stream latency (full): {stream_latency_ms:.0f} ms")
print(f"  SSE data lines: {sse_count}, content chunks: {chunks_with_content}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
passed = sum(results)
total = len(results)
print(f"\n{'='*40}")
print(f"Total: {passed}/{total} passed")
sys.exit(0 if all(results) else 1)
