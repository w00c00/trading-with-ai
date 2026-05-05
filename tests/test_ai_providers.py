from app.ai.providers import _normalize_model, _parse_model_json


def test_normalize_minimax_model_alias() -> None:
    assert _normalize_model("minimax", "minimax m2.7") == "MiniMax-M2.7"
    assert _normalize_model("minimax", "m2.7-highspeed") == "MiniMax-M2.7-highspeed"


def test_normalize_model_leaves_other_providers_alone() -> None:
    assert _normalize_model("openrouter", "openai/gpt-4o-mini") == "openai/gpt-4o-mini"


def test_parse_model_json_from_fenced_thinking_response() -> None:
    content = """
<think>先分析市场。</think>

```json
{
  "action": "hold",
  "confidence": 0.52,
  "reason": "信号冲突",
  "risk_notes": ["等待更清晰"]
}
```
"""
    parsed = _parse_model_json(content)
    assert parsed is not None
    assert parsed["action"] == "hold"
    assert parsed["confidence"] == 0.52
    assert parsed["risk_notes"] == ["等待更清晰"]


def test_parse_model_json_from_plain_text_prefix() -> None:
    parsed = _parse_model_json('结论如下：{"action":"buy","confidence":0.7,"reason":"breakout","risk_notes":[]}')
    assert parsed is not None
    assert parsed["action"] == "buy"
