from app.ai.providers import _normalize_model


def test_normalize_minimax_model_alias() -> None:
    assert _normalize_model("minimax", "minimax m2.7") == "MiniMax-M2.7"
    assert _normalize_model("minimax", "m2.7-highspeed") == "MiniMax-M2.7-highspeed"


def test_normalize_model_leaves_other_providers_alone() -> None:
    assert _normalize_model("openrouter", "openai/gpt-4o-mini") == "openai/gpt-4o-mini"
