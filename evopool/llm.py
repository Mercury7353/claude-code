"""
LLM backend abstraction for EvoPool.
Supports Anthropic (Claude) and OpenAI (GPT) via their respective APIs.
Also supports local vLLM/Ollama servers.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any


def load_server_info() -> dict | None:
    """Load vLLM server info written by serve_vllm.sh."""
    import json as _json
    server_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "vllm_server.json"
    )
    if os.path.exists(server_file):
        try:
            with open(server_file) as f:
                return _json.load(f)
        except Exception:
            pass
    return None


def _get_local_url() -> str:
    """
    Return a vLLM server URL for load balancing.
    Checks EVOPOOL_LOCAL_LLM_URLS (comma-separated) first, then EVOPOOL_LOCAL_LLM_URL.
    Picks randomly among available URLs for simple load balancing.
    """
    multi = os.environ.get("EVOPOOL_LOCAL_LLM_URLS", "")
    if multi:
        urls = [u.strip() for u in multi.split(",") if u.strip()]
        if urls:
            return random.choice(urls)
    return os.environ.get("EVOPOOL_LOCAL_LLM_URL", "")


def llm_call(
    model: str,
    user: str,
    system: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    retry: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Call an LLM and return the response text.
    Supports claude-* models via Anthropic API and gpt-* / o* models via OpenAI API.
    Falls back to a local vLLM endpoint if EVOPOOL_LOCAL_LLM_URL is set.
    """
    # Auto-detect vLLM server if no explicit URL set and model looks like a local one
    if not os.environ.get("EVOPOOL_LOCAL_LLM_URL") and not os.environ.get("EVOPOOL_LOCAL_LLM_URLS") and not model.startswith(("claude", "gpt-", "o1", "o3", "o4")):
        server = load_server_info()
        if server:
            os.environ["EVOPOOL_LOCAL_LLM_URL"] = server["url"]

    for attempt in range(retry):
        try:
            local_url = _get_local_url()
            if local_url and not model.startswith(("claude", "gpt-", "o1", "o3", "o4")):
                return _call_local(model, user, system, max_tokens, temperature, local_url)
            elif model.startswith("claude"):
                return _call_anthropic(model, user, system, max_tokens, temperature)
            elif model.startswith(("gpt-", "o1", "o3", "o4")):
                return _call_openai(model, user, system, max_tokens, temperature)
            elif local_url:
                return _call_local(model, user, system, max_tokens, temperature, local_url)
            else:
                return _call_anthropic(model, user, system, max_tokens, temperature)
        except Exception as e:
            if attempt == retry - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
    return ""


def _call_anthropic(model: str, user: str, system: str, max_tokens: int, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _call_openai(model: str, user: str, system: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_local(model: str, user: str, system: str, max_tokens: int, temperature: float, url: str = "") -> str:
    """Call a local vLLM/Ollama endpoint. Strips Qwen3 <think>...</think> tokens."""
    import re
    import requests

    if not url:
        url = _get_local_url() or os.environ.get("EVOPOOL_LOCAL_LLM_URL", "")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens + 512,  # extra headroom for thinking tokens
        "temperature": temperature,
    }
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # Strip Qwen3 thinking tokens <think>...</think>
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Handle truncated thinking: if <think> was opened but never closed (token limit hit),
    # strip only the opening tag and keep the content — the response may still contain
    # a \boxed{answer} (MATH) or code block inside the thinking we want to evaluate.
    if "<think>" in content:
        content = content[content.index("<think>") + len("<think>"):].strip()
    return content
