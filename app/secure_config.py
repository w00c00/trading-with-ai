from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

CONFIG_PATH = Path(os.getenv("TRADAI_CONFIG_PATH", ".config/settings.enc"))
KEY_PATH = Path(os.getenv("TRADAI_CONFIG_KEY_PATH", ".config/secret.key"))

SECRET_FIELDS = {
    "ai_api_key",
    "exchange_api_key",
    "exchange_secret",
    "exchange_password",
    "serverchan_sendkey",
}


def load_encrypted_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = _fernet().decrypt(CONFIG_PATH.read_bytes())
        payload = json.loads(data.decode("utf-8"))
    except (InvalidToken, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def save_encrypted_config(values: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_values = {key: value for key, value in values.items() if value is not None}
    encrypted = _fernet().encrypt(json.dumps(clean_values, ensure_ascii=False).encode("utf-8"))
    CONFIG_PATH.write_bytes(encrypted)


def public_config_snapshot(settings_dict: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in settings_dict.items():
        if key in SECRET_FIELDS:
            snapshot[key] = {"configured": bool(value), "masked": _mask_secret(str(value)) if value else ""}
        else:
            snapshot[key] = value
    return snapshot


def merge_config_update(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in update.items():
        if value is None:
            continue
        if key in SECRET_FIELDS and value == "":
            continue
        merged[key] = value
    return merged


def _fernet() -> Fernet:
    key = os.getenv("TRADAI_CONFIG_KEY")
    if key:
        return Fernet(key.encode("utf-8"))
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(Fernet.generate_key())
        KEY_PATH.chmod(0o600)
    return Fernet(KEY_PATH.read_bytes())


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"
