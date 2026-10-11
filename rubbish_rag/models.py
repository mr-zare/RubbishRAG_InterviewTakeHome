"""Shared typed domain models for RubbishRAG.

The public decorator API continues to use dictionaries for compatibility with the
challenge. Internally, these dataclasses give us stable contracts across P1-P12.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EntityType(StrEnum):
    BANK = "bank"
    RANK = "rank"
    PERSONA = "persona"
    DEBT = "debt"


class Persona(StrEnum):
    INDIVIDUAL = "حقیقی"
    LEGAL = "حقوقی"


class DebtType(StrEnum):
    CHECK = "چک"
    TAX = "مالیاتی"
    LOAN = "تسهیلاتی"
    LEGAL = "محکومیت"
    SUPPLIER = "تامین‌کنندگان"


class QuantityType(StrEnum):
    CREDIT_SCORE = "credit_score"
    DURATION = "duration"
    MONEY = "money"
    PERCENTAGE = "percentage"
    DATE = "date"
    YEAR = "year"
    COUNT = "count"
    GENERIC_NUMBER = "generic_number"


class Comparator(StrEnum):
    EQ = "eq"
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"
    BETWEEN = "between"
    UNKNOWN = "unknown"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class Intent(StrEnum):
    DEFINE_RANK = "define_rank"
    EXPLAIN_SCORE = "explain_score"
    FACTOR_EFFECT = "factor_effect"
    DEBT_EFFECT = "debt_effect"
    DEBT_REMOVAL = "debt_removal"
    LOAN_ELIGIBILITY = "loan_eligibility"
    REPORT_CONTENT = "report_content"
    DATA_CORRECTION = "data_correction"
    TIMING = "timing"
    COMPARISON = "comparison"
    GENERAL_INFO = "general_info"


class Relation(StrEnum):
    SINGLE = "single"
    LIST = "list"
    COMPARISON = "comparison"
    ALTERNATIVE = "alternative"


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class NormalizationChange:
    code: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    original_text: str
    canonical_text: str
    match_text: str
    transformations: tuple[NormalizationChange, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntityMention:
    entity_type: EntityType
    canonical_id: str
    canonical_value: str
    raw_text: str
    start: int
    end: int
    match_type: str = "alias"


@dataclass(frozen=True, slots=True)
class Quantity:
    quantity_type: QuantityType
    value: float | int | None
    unit: str | None
    comparator: Comparator = Comparator.EQ
    start: int = -1
    end: int = -1
    raw_text: str = ""
    value_max: float | int | None = None


@dataclass(frozen=True, slots=True)
class Proposition:
    concept: str
    polarity: Polarity
    cue: str | None = None
    start: int = -1
    end: int = -1


@dataclass(frozen=True, slots=True)
class AmbiguityInfo:
    missing_required_slots: tuple[str, ...] = ()
    conflicting_slots: tuple[str, ...] = ()
    underspecified: bool = False


@dataclass(slots=True)
class QuerySemanticFrame:
    original_text: str
    canonical_text: str
    match_text: str
    intent: Intent = Intent.GENERAL_INFO
    relation: Relation = Relation.SINGLE
    banks: list[EntityMention] = field(default_factory=list)
    ranks: list[EntityMention] = field(default_factory=list)
    personas: list[EntityMention] = field(default_factory=list)
    debt_types: list[EntityMention] = field(default_factory=list)
    quantities: list[Quantity] = field(default_factory=list)
    propositions: list[Proposition] = field(default_factory=list)
    ambiguity: AmbiguityInfo = field(default_factory=AmbiguityInfo)
    normalization_changes: tuple[NormalizationChange, ...] = ()

    def values(self, entity_type: EntityType) -> list[str]:
        mapping = {
            EntityType.BANK: self.banks,
            EntityType.RANK: self.ranks,
            EntityType.PERSONA: self.personas,
            EntityType.DEBT: self.debt_types,
        }
        return [m.canonical_value for m in mapping[entity_type]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
