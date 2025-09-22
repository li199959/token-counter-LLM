"""Dataclasses describing model and tokenizer configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Pricing:
    """Pricing information for a model."""

    currency: str = "USD"
    input_per_1k: Optional[float] = None
    output_per_1k: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation."""

        return {
            "currency": self.currency,
            "input_per_1k": self.input_per_1k,
            "output_per_1k": self.output_per_1k,
        }


@dataclass(frozen=True)
class TokenizerSpec:
    """Description of a tokenizer implementation."""

    type: str
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "options": dict(self.options)}


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a supported large language model."""

    model_id: str
    display_name: str
    family: str
    provider: str
    max_context: int
    tokenizer: TokenizerSpec
    description: Optional[str] = None
    pricing: Optional[Pricing] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.model_id,
            "display_name": self.display_name,
            "family": self.family,
            "provider": self.provider,
            "max_context": self.max_context,
            "tokenizer": self.tokenizer.to_dict(),
        }
        if self.description:
            data["description"] = self.description
        if self.pricing:
            data["pricing"] = self.pricing.to_dict()
        return data
