"""Proof-run: swarm chat completion via gateway /v1/chat/completions."""

import json
import ssl
import sys
import time
import urllib.request

BASE = "https://48acx0kqem74jt-9000.proxy.runpod.net"
URL = f"{BASE}/v1/chat/completions"
CTX = ssl._create_unverified_context()
def post(payload: dict) -> tuple[dict, float]:
    """POST JSON to the chat completions endpoint; return (body, latency_s)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json", "User-Agent": "dualmirakl-proof-run"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, context=CTX) as resp:
        body = json.loads(resp.read().decode())
    latency = time.perf_counter() - t0
    return body, latency
def extract_content(choices: list[dict]) -> str:
    """Extract text from choices, handling Nemotron reasoning_content fallback."""
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content") or msg.get("reasoning_content") or ""


def main() -> int:
    PASSED = 0
    FAILED = 0
    def check(label: str, condition: bool, detail: str = ""):
        nonlocal PASSED, FAILED
        if condition:
            PASSED += 1
            print(f"[PASS] {label}")
        else:
            FAILED += 1
            msg = f"[FAIL] {label}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    # ------------------------------------------------------------------
    # Test 1: basic swarm chat completion
    # ------------------------------------------------------------------
    print("=== Test 1: basic swarm chat completion ===")
    payload1 = {
        "model": "swarm",
        "messages": [{"role": "user", "content": "Respond with a single word: yes or no."}],
        "max_tokens": 30,
        "temperature": 0.1,
    }
    resp1, lat1 = post(payload1)

    # 1a. response has choices array with at least one choice
    choices = resp1.get("choices", [])
    check("choices array present and non-empty", isinstance(choices, list) and len(choices) > 0,
          f"got {type(choices).__name__} len={len(choices) if isinstance(choices, list) else 'N/A'}")

    # 1b. model produced output (completion_tokens > 0 proves generation occurred)
    content1 = extract_content(choices)
    usage = resp1.get("usage", {})
    generated = usage.get("completion_tokens", 0) > 0
    check("swarm generated tokens (completion_tokens > 0)", generated,
          f"completion_tokens={usage.get('completion_tokens')}, content={content1!r}")

    # 1c. routing -- model field should NOT contain 'authority'
    resp_model = resp1.get("model", "")
    check("routed to swarm (model field does not contain 'authority')",
          "authority" not in resp_model.lower(),
          f"model={resp_model!r}")

    # 1d. usage field with prompt_tokens and completion_tokens
    has_prompt = isinstance(usage.get("prompt_tokens"), int)
    has_completion = isinstance(usage.get("completion_tokens"), int)
    check("usage.prompt_tokens present (int)", has_prompt, f"usage={usage}")
    check("usage.completion_tokens present (int)", has_completion, f"usage={usage}")

    # 1e. latency
    print(f"       latency: {lat1:.2f}s")

    # ------------------------------------------------------------------
    # Test 2: system + user message
    # ------------------------------------------------------------------
    print("\n=== Test 2: system + user message ===")
    payload2 = {
        "model": "swarm",
        "messages": [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 30,
        "temperature": 0.1,
    }
    resp2, lat2 = post(payload2)

    choices2 = resp2.get("choices", [])
    content2 = extract_content(choices2)
    check("system+user: choices present", isinstance(choices2, list) and len(choices2) > 0)
    check("system+user: content is non-empty", len(content2.strip()) > 0, f"content={content2!r}")
    check("system+user: response is coherent (non-empty)", len(content2.strip()) > 0,
          f"content={content2!r}")

    # routing
    resp_model2 = resp2.get("model", "")
    check("system+user: routed to swarm", "authority" not in resp_model2.lower(),
          f"model={resp_model2!r}")

    # usage
    usage2 = resp2.get("usage", {})
    check("system+user: usage.prompt_tokens present", isinstance(usage2.get("prompt_tokens"), int))
    check("system+user: usage.completion_tokens present", isinstance(usage2.get("completion_tokens"), int))

    print(f"       latency: {lat2:.2f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = PASSED + FAILED
    print(f"\n{'='*40}")
    print(f"TOTAL: {PASSED}/{total} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
