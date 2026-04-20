import os
import json
import re
import requests
from dotenv import load_dotenv

def _safe_load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            load_dotenv(dotenv_path=env_path, encoding=encoding, override=False)
            return
        except (UnicodeDecodeError, Exception):
            continue

_safe_load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HF_MODEL   = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
API_URL    = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


def _extract_json(text: str) -> dict | None:
    """
    Robustly extract the first valid JSON object from a raw LLM string.
    Handles markdown fences, leading prose, trailing text, etc.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try to find a JSON block between the first { and last }
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def call_llm(prompt: str, max_new_tokens: int = 512) -> dict | None:
    """
    Call the HuggingFace Inference API with the given prompt.
    Returns a parsed dict if successful, else None.
    """
    if not HF_API_KEY:
        return None

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,          # keep it deterministic
            "do_sample": False,
            "return_full_text": False,
        },
    }

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # HF returns a list of generated texts
        if isinstance(data, list) and data:
            raw_text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            raw_text = data.get("generated_text", "")
        else:
            return None

        return _extract_json(raw_text)

    except requests.exceptions.RequestException as exc:
        print(f"[LLMClient] HTTP error: {exc}")
        return None
    except Exception as exc:
        print(f"[LLMClient] Unexpected error: {exc}")
        return None