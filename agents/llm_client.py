"""
LLM Client
----------
Calls HuggingFace's current OpenAI-compatible router first, then falls
back to the legacy model endpoint for older local setups.

The agents require strict JSON. Generic completion models such as GPT-2,
BLOOM, or base FLAN often return plain text, so they are intentionally not
used as defaults for agent decisions.

Your HuggingFace token must have Inference Providers permission.
"""

import ast
import json
import os
import re
import time

import requests
from dotenv import load_dotenv


def _safe_load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            load_dotenv(dotenv_path=env_path, encoding=enc, override=False)
            return
        except Exception:
            continue


_safe_load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "").strip()
HF_MODEL_ENV = os.getenv("HF_MODEL", "").strip()
HF_ROUTER_URL = os.getenv(
    "HF_ROUTER_URL",
    "https://router.huggingface.co/v1/chat/completions",
).strip()

# HuggingFace router chat models. These are instruction/chat models because
# the AutoML agents need structured JSON, not open-ended completions.
DEFAULT_MODEL_CHAIN = [
    "openai/gpt-oss-120b:fireworks-ai",
    "Qwen/Qwen2.5-7B-Instruct-1M:preferred",
    "google/gemma-2-2b-it:preferred",
    "Qwen/Qwen2.5-0.5B-Instruct:preferred",
]

GATED_OR_LEGACY_MODELS = (
    "google/flan-t5",
    "bigscience/bloom",
    "gpt2",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/",
)


def _looks_gated_or_legacy(model: str) -> bool:
    return any(marker in model for marker in GATED_OR_LEGACY_MODELS)


env_models = [m.strip() for m in HF_MODEL_ENV.split(",") if m.strip()]
preferred_env_models = [m for m in env_models if not _looks_gated_or_legacy(m)]
last_chance_env_models = [m for m in env_models if _looks_gated_or_legacy(m)]

MODEL_CHAIN = preferred_env_models + DEFAULT_MODEL_CHAIN + last_chance_env_models
MODEL_CHAIN = list(dict.fromkeys(m for m in MODEL_CHAIN if m))

_working_model: str | None = None

SEP = "=" * 64

HTTP_HINTS = {
    401: "Invalid/expired key, or token missing Inference Providers permission",
    402: "Billing or provider quota issue on HuggingFace",
    403: "Forbidden: gated model, missing provider permission, or account access issue",
    404: "Model/provider not found - check the model name",
    429: "Rate limited - wait and try again",
    500: "HuggingFace server error - try again in a minute",
    503: "Model/provider is loading or temporarily unavailable",
}


def _legacy_url(model: str) -> str:
    return f"https://api-inference.huggingface.co/models/{model}"


def _extract_json(text: str, agent_name: str) -> dict | None:
    """Robustly extract JSON from LLM output."""
    if not text or not text.strip():
        print("[LLMClient] X Empty response from model")
        return None

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        print(f"[LLMClient] X No JSON found in: {text[:200]}")
        return None

    candidate = text[start : end + 1]
    try:
        result = json.loads(candidate)
        print(f"[LLMClient] OK JSON extracted successfully for {agent_name}")
        return result
    except json.JSONDecodeError as e:
        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, dict):
                print(f"[LLMClient] OK JSON extracted via ast.literal_eval for {agent_name}")
                return result
        except Exception:
            pass

        print(f"[LLMClient] X JSON parse failed: {e}")
        print(f"[LLMClient]   Attempted: {candidate[:200]}")
        return None


def _try_router_chat(model: str, prompt: str, max_new_tokens: int) -> str | None:
    """Try HuggingFace router chat completions. Returns raw text or None."""
    is_reasoning_model = "gpt-oss" in model or "DeepSeek-R1" in model
    router_max_tokens = max(max_new_tokens, 900) if is_reasoning_model else max_new_tokens

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise AutoML agent. Return only valid JSON. "
                    "Do not include markdown fences or explanatory text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": router_max_tokens,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    if is_reasoning_model:
        payload["reasoning_effort"] = "low"

    try:
        resp = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            hint = HTTP_HINTS.get(resp.status_code, "Unexpected router error")
            print(f"[LLMClient]   Router HTTP {resp.status_code}: {hint}")
            print(f"[LLMClient]   Body: {resp.text[:300]}")
            return None

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            print("[LLMClient]   Router returned no choices")
            return None

        message = choices[0].get("message") or {}
        raw = message.get("content", "")
        if isinstance(raw, list):
            raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
        if raw:
            return raw

        finish_reason = choices[0].get("finish_reason", "unknown")
        print(f"[LLMClient]   Router returned empty content (finish_reason={finish_reason})")
        print(f"[LLMClient]   Router message keys: {list(message.keys())}")
        return None

    except requests.exceptions.ConnectionError:
        print("[LLMClient]   Router connection failed - check internet")
        return None
    except requests.exceptions.Timeout:
        print("[LLMClient]   Router timeout after 60s")
        return None
    except Exception as e:
        print(f"[LLMClient]   Router error: {type(e).__name__}: {e}")
        return None


def _try_legacy_model(model: str, prompt: str, max_new_tokens: int) -> str | None:
    """Try the legacy Inference API. Returns raw text or None."""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.05,
            "do_sample": False,
            "return_full_text": False,
        },
    }

    for attempt in range(2):
        try:
            resp = requests.post(_legacy_url(model), headers=headers, json=payload, timeout=60)

            if resp.status_code == 503:
                body = resp.json() if resp.content else {}
                wait = body.get("estimated_time", 20)
                if attempt == 0:
                    print(f"[LLMClient]   Legacy 503 - waiting {wait:.0f}s...")
                    time.sleep(min(wait, 25))
                    continue
                print("[LLMClient]   Legacy 503 after retry")
                return None

            if not resp.ok:
                hint = HTTP_HINTS.get(resp.status_code, "Unexpected legacy error")
                print(f"[LLMClient]   Legacy HTTP {resp.status_code}: {hint}")
                print(f"[LLMClient]   Body: {resp.text[:300]}")
                return None

            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            if isinstance(data, dict):
                if "error" in data:
                    print(f"[LLMClient]   Legacy API error: {data['error']}")
                    return None
                return data.get("generated_text", "")

            print(f"[LLMClient]   Legacy unexpected format: {type(data)}")
            return None

        except requests.exceptions.ConnectionError:
            print("[LLMClient]   Legacy connection failed - check internet")
            return None
        except requests.exceptions.Timeout:
            print("[LLMClient]   Legacy timeout after 60s")
            return None
        except Exception as e:
            print(f"[LLMClient]   Legacy error: {type(e).__name__}: {e}")
            return None

    return None


def _try_model(model: str, prompt: str, max_new_tokens: int) -> str | None:
    """Try the router first, then the legacy endpoint."""
    raw = _try_router_chat(model, prompt, max_new_tokens)
    if raw is not None:
        return raw

    legacy_model = model.split(":", 1)[0]
    if legacy_model != model:
        print(f"[LLMClient]   Trying legacy endpoint as {legacy_model}")
    return _try_legacy_model(legacy_model, prompt, max_new_tokens)


def call_llm(prompt: str, max_new_tokens: int = 400, agent_name: str = "Agent") -> dict | None:
    """
    Call HuggingFace and return a parsed JSON dict, or None if all models fail.
    """
    global _working_model

    print(f"\n{SEP}")
    print(f"[LLMClient] >>> {agent_name} - LLM call starting")

    if not HF_API_KEY:
        print("[LLMClient] X No HUGGINGFACE_API_KEY in .env")
        print("[LLMClient]   Create a token with Inference Providers permission")
        print(f"[LLMClient]   -> {agent_name} will use rule-based fallback")
        print(SEP)
        return None

    print(f"[LLMClient]   Key: {HF_API_KEY[:8]}...{HF_API_KEY[-4:]} (len={len(HF_API_KEY)})")

    chain = MODEL_CHAIN.copy()
    if _working_model and _working_model in chain:
        chain.remove(_working_model)
        chain.insert(0, _working_model)

    for model in chain:
        print(f"[LLMClient]   Trying model: {model}")
        raw = _try_model(model, prompt, max_new_tokens)

        if raw is None:
            print(f"[LLMClient]   Model {model} failed - trying next...")
            continue

        print(f"[LLMClient] <<< Response ({len(raw)} chars):")
        print(f"  {raw[:600]}")

        parsed = _extract_json(raw, agent_name)
        if parsed is not None:
            _working_model = model
            print(f"[LLMClient] OK SUCCESS with model: {model}")
            print(f"[LLMClient] OK Parsed: {json.dumps(parsed)}")
            print(SEP)
            return parsed

        print(f"[LLMClient]   Model {model} responded but JSON extraction failed")
        print("[LLMClient]   Trying next model...")

    print(f"[LLMClient] X ALL models in chain failed for {agent_name}")
    print(f"[LLMClient]   Models tried: {chain}")
    print(f"[LLMClient]   -> {agent_name} will use rule-based fallback")
    print(SEP)
    return None


def get_current_model() -> str:
    """Return the currently working model name for UI display."""
    return _working_model or "None (rule-based)"
