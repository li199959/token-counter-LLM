"""Utilities for loading model configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import ModelSpec, Pricing, TokenizerSpec

_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "resources" / "model_registry.json"


def _parse_pricing(data):
    if data is None:
        return None
    currency = data.get("currency", "USD")
    input_price = data.get("input_per_1k")
    output_price = data.get("output_per_1k")
    return Pricing(currency=currency, input_per_1k=input_price, output_per_1k=output_price)


def _parse_tokenizer(spec):
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError("Tokenizer spec must define a 'type'.")
    type_name = spec["type"].strip().lower()
    options = spec.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("Tokenizer options must be a mapping.")
    return TokenizerSpec(type=type_name, options=options)


def load_registry(path: Path | None = None) -> List[ModelSpec]:
    """Load model specifications from a JSON file."""

    target = Path(path) if path else _DEFAULT_REGISTRY_PATH
    if not target.exists():
        raise FileNotFoundError(f"Model registry file not found: {target}")

    with target.open("r", encoding="utf-8") as stream:
        raw_data = json.load(stream)

    if not isinstance(raw_data, Iterable):
        raise ValueError("Model registry must be a list of models.")

    models: List[ModelSpec] = []
    for item in raw_data:
        model_id = item.get("id")
        if not model_id:
            raise ValueError("Model entry must include an 'id'.")
        tokenizer_spec = _parse_tokenizer(item.get("tokenizer"))
        pricing = _parse_pricing(item.get("pricing"))
        models.append(
            ModelSpec(
                model_id=model_id,
                display_name=item.get("display_name", model_id),
                family=item.get("family", "unknown"),
                provider=item.get("provider", "unknown"),
                max_context=int(item.get("max_context", 0)),
                tokenizer=tokenizer_spec,
                description=item.get("description"),
                pricing=pricing,
            )
        )
    return models
