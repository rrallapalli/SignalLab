"""Single-pass extraction: one LLM call -> all four signals' features."""
from .extractor import (
    Completer, OpenAICompleter, FeatureBundle,
    SYSTEM_PROMPT, USER_TEMPLATE,
    extract, attach_prior, extract_and_score, build_features,
)
