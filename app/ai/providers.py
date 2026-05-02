from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import Settings


DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "minimax": "https://api.minimaxi.com/v1",
    "mimo": "https://api.mimo-v2.com/v1",
}


class AIClient:
    def __init__(self, settings: Settings) -> None:
        self.provider = settings.ai_provider
        self.model = _normalize_model(settings.ai_provider, settings.ai_model)
        self.api_key = settings.ai_api_key
        self.base_url = (settings.ai_base_url or DEFAULT_BASE_URLS.get(settings.ai_provider, "")).rstrip("/")

    async def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.1) -> dict[str, Any]:
        if not self.api_key:
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": "AI_API_KEY is empty; using safety hold.",
                "risk_notes": ["AI provider was not called."],
            }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://local.trading-with-ai"
            headers["X-Title"] = "Trading with AI"

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(self._format_http_error(exc)) from exc
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"action": "hold", "confidence": 0.0, "reason": content[:500], "risk_notes": ["Model did not return JSON."]}
        parsed["_provider_response"] = data
        return parsed

    def _format_http_error(self, exc: httpx.HTTPStatusError) -> str:
        status = exc.response.status_code
        body = exc.response.text[:500]
        message = f"{self.provider} API returned HTTP {status} at {self.base_url}/chat/completions."
        if self.provider == "minimax" and status == 401:
            message += (
                " MiniMax 鉴权失败：请检查 AI_API_KEY 是否来自 MiniMax 控制台；"
                "中国区账号通常需要把 AI_BASE_URL 设置为 https://api.minimaxi.com/v1，"
                "国际区账号使用 https://api.minimax.io/v1；"
                "模型建议使用 MiniMax-M2.7 或 MiniMax-M2.7-highspeed。"
            )
        elif status == 401:
            message += " 请检查 API Key、Base URL 和账号权限。"
        if body:
            message += f" Provider response: {body}"
        return message


def _normalize_model(provider: str, model: str) -> str:
    if provider != "minimax":
        return model
    compact = model.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "minimax-m2": "MiniMax-M2",
        "minimax-m2.5": "MiniMax-M2.5",
        "minimax-m2.7": "MiniMax-M2.7",
        "minimax-m2.7-highspeed": "MiniMax-M2.7-highspeed",
        "m2": "MiniMax-M2",
        "m2.5": "MiniMax-M2.5",
        "m2.7": "MiniMax-M2.7",
        "m2.7-highspeed": "MiniMax-M2.7-highspeed",
    }
    return aliases.get(compact, model.strip())
