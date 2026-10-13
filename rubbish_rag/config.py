from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    unicode_form: str = "NFC"
    digits_to_ascii: bool = True
    remove_tatweel: bool = True
    strip_diacritics_for_match: bool = True
    zwnj_to_space_for_match: bool = True


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    normalization: NormalizationConfig = NormalizationConfig()
    pipeline_version: str = "rubbishrag-v2-core-1"

    def stable_hash(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


DEFAULT_CONFIG = PipelineConfig()
