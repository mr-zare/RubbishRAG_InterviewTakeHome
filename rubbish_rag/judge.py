"""Adaptive System-1 RAG Decision Judge.

Conceptually inspired by Self-RAG, Adaptive-RAG, Corrective RAG (CRAG), and JEV-as-a-Judge.
Provides a fast, structured decision layer in front of and within the RAG pipeline:
  - Decision 1: Needs retrieval? (RETRIEVE, NO_RETRIEVAL, UNCERTAIN)
  - Decision 2: Needs clarification? (CLARIFY, DO_NOT_CLARIFY + missing slots)
  - Decision 3: Structured retrieval planning (entities, facets, strategy)
  - Decision 4: Dynamic context budgeting (top_k selection: 0, 2, 3, 5, 8)
  - Decision 5: Collective evidence sufficiency assessment (SUFFICIENT, INSUFFICIENT, CONFLICTING, IRRELEVANT)
  - Decision 6: Adaptive stopping & next action (ANSWER, RETRIEVE_MORE, CLARIFY, ABSTAIN)
  - Calibrated confidence scoring (Brier score & Expected Calibration Error)
"""
import dataclasses
import json
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .evidence import (
    assess_sufficiency,
    build_evidence_set,
)
from .normalize_fa import normalize_fa
from .slots import (
    BANKS,
    RANKS,
    parse_query,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. Typed Schemas
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PreRetrievalDecision:
    """Fast pre-retrieval decision on query routing, planning, and context budget."""
    action: str  # "retrieve", "no_retrieval", "clarify"
    needs_retrieval: bool
    needs_clarification: bool
    missing_slots: List[str]
    retrieval_strategy: str  # "hybrid", "sparse_only", "dense_only"
    query_rewrite: str
    target_entities: List[str]
    target_facets: List[str]
    top_k: int
    confidence: float
    retrieve_probability: float
    clarify_probability: float
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EvidenceDecision:
    """Post-retrieval assessment of collective evidence quality and next action."""
    sufficiency: str  # "sufficient", "insufficient", "conflicting", "irrelevant", "uncertain"
    evidence_score: float
    conflicting_items: List[int]
    poison_detected: bool
    repetition_detected: bool
    novel_evidence_chunks: int
    next_action: str  # "answer", "retrieve_more", "clarify", "abstain"
    confidence: float
    sufficiency_probability: float
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# 2. Protocol Interface
# ---------------------------------------------------------------------------

class RAGDecisionJudge(Protocol):
    """Provider-neutral protocol for an Adaptive RAG Decision Judge."""

    def pre_retrieval(
        self,
        query: str,
        conversation: Optional[List[Dict[str, Any]]] = None,
        slots: Optional[Dict[str, Any]] = None,
    ) -> PreRetrievalDecision:
        """Evaluate query before retrieval: decide routing, planning, and budget."""
        ...

    def post_retrieval(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        state: Optional[Dict[str, Any]] = None,
    ) -> EvidenceDecision:
        """Evaluate retrieved candidate evidence collectively and determine next action."""
        ...


# ---------------------------------------------------------------------------
# 3. High-Speed Calibrated Heuristic Feature Judge (CPU-viable, <1 ms)
# ---------------------------------------------------------------------------

class HeuristicFeatureJudge:
    """Lightweight, deterministic, calibrated System-1 decision judge.
    
    Operates without GPU requirements or external APIs, executing in under 1ms
    on CPU with principled feature scoring and probability calibration.
    """

    CONVERSATIONAL_PATTERNS = [
        r"^(سلام|درود|روزتون بخیر|شب بخیر|صبح بخیر|خداحافظ|مرسی|ممنون|تشکر|وقت بخیر)",
        r"^(چطور کار می‌کنی|چه کاری می‌تونی|کمک|راهنمایی کن|شما چطور)",
    ]

    SPAM_ADVERSARIAL_PATTERNS = [
        r"(رتبه a برای همه|بدون اعتبارسنجی ۱۰۰ درصد|وام فوری بدون ضامن و بدون اعتبارسنجی)",
    ]

    DEBT_KEYWORDS = ["بدهی", "معوق", "مشکوک الوصول", "سررسید گذشته"]
    CHECK_KEYWORDS = ["چک", "صیادی", "برگشت", "رفع سوءاثر", "رفع سواثر"]
    RANK_KEYWORDS = ["رتبه", "امتیاز", "گرید"]
    LOAN_KEYWORDS = ["وام", "تسهیلات", "قسط", "اقساط"]

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def pre_retrieval(
        self,
        query: str,
        conversation: Optional[List[Dict[str, Any]]] = None,
        slots: Optional[Dict[str, Any]] = None,
    ) -> PreRetrievalDecision:
        norm = normalize_fa(query or "")
        frame = parse_query(query or "")
        words = norm.split()
        word_count = len(words)
        reasons = []

        # 1. Conversational / Meta Check (Decision 1: NO_RETRIEVAL)
        is_conv = any(re.search(pat, norm) for pat in self.CONVERSATIONAL_PATTERNS)
        from .slots import extract_slots as get_slots
        s = slots or get_slots(norm)

        if is_conv and word_count <= 6 and not s.get("RANK") and not s.get("BANK"):
            reasons.append("CONVERSATIONAL_PROMPT")
            return PreRetrievalDecision(
                action="no_retrieval",
                needs_retrieval=False,
                needs_clarification=False,
                missing_slots=[],
                retrieval_strategy="none",
                query_rewrite=norm,
                target_entities=[],
                target_facets=["conversation"],
                top_k=0,
                confidence=0.96,
                retrieve_probability=0.03,
                clarify_probability=0.01,
                reason_codes=reasons,
            )

        # 2. Extract Entities & Slots
        target_entities = []
        if s.get("RANK"):
            target_entities.append(s["RANK"])
        if s.get("BANK"):
            target_entities.append(s["BANK"])
        if s.get("PERSONA"):
            target_entities.append(s["PERSONA"])

        # 3. Ambiguity & Clarification Gating (Decision 2: CLARIFY)
        missing_slots = list(
            frame.ambiguity.missing_required_slots
        )
        is_ambiguous = bool(
            frame.ambiguity.underspecified
            or missing_slots
            or frame.ambiguity.conflicting_slots
        )
        clarify_prob = (
            0.90
            if is_ambiguous
            else 0.05
        )

        if is_ambiguous:
            reasons.append(
                "P3_STRUCTURED_AMBIGUITY"
            )

        # Check for under-specified debt or bad record
        if (any(k in norm for k in self.DEBT_KEYWORDS) or "سابقه" in norm) and not s.get("DEBT") and not s.get("RANK") and word_count <= 7:
            missing_slots.append("debt_type")
            is_ambiguous = True
            clarify_prob = 0.88
            reasons.append("UNDERSPECIFIED_DEBT_CATEGORY")

        # Check for under-specified rank/score
        elif any(k in norm for k in self.RANK_KEYWORDS) and not s.get("RANK") and any(w in norm for w in ["پایین", "افت", "کم", "بد", "تغییر", "عوض", "درست"]) and word_count <= 8:
            missing_slots.append("rank")
            if not s.get("PERSONA"):
                missing_slots.append("persona")
            is_ambiguous = True
            clarify_prob = 0.86
            reasons.append("MISSING_RANK_AND_PERSONA")

        # Check for under-specified check inquiry
        elif any(k in norm for k in self.CHECK_KEYWORDS) and not s.get("PERSONA") and not s.get("TIME") and word_count <= 6:
            if any(w in norm for w in ["برگشت", "چیکار", "پاک میشه", "اثر"]):
                missing_slots.append("persona")
                is_ambiguous = True
                clarify_prob = 0.84
                reasons.append("UNDERSPECIFIED_CHECK_STATUS")

        # Check for under-specified loan intent or guarantee
        elif "وام" in norm and not s.get("BANK") and not s.get("RANK") and word_count <= 6:
            missing_slots.append("bank")
            missing_slots.append("rank")
            is_ambiguous = True
            clarify_prob = 0.85
            reasons.append("UNDERSPECIFIED_LOAN_APPLICATION")

        elif "ضامن" in norm and not s.get("RANK") and word_count <= 5:
            missing_slots.append("guarantee_status")
            is_ambiguous = True
            clarify_prob = 0.82
            reasons.append("UNDERSPECIFIED_GUARANTEE_STATUS")

        elif "مالیات" in norm and not s.get("TIME") and any(w in norm for w in ["ندادم", "چی میشه", "ندارم"]) and word_count <= 5:
            missing_slots.append("tax_stage")
            is_ambiguous = True
            clarify_prob = 0.83
            reasons.append("UNDERSPECIFIED_TAX_STAGE")

        elif "شرکت" in norm and any(w in norm for w in ["امتیاز نداره", "سابقه نداره", "تاسیس"]) and word_count <= 6:
            missing_slots.append("company_age")
            is_ambiguous = True
            clarify_prob = 0.85
            reasons.append("UNDERSPECIFIED_COMPANY_AGE")

        # 4. Context Budget Selection (Decision 4: Dynamic Top-K)
        if is_ambiguous:
            recommended_k = 3
            reasons.append("AMBIGUOUS_REDUCED_BUDGET")
        elif len(target_entities) >= 2:
            recommended_k = 5
            reasons.append("MULTI_ENTITY_BUDGET")
        elif any(w in norm for w in ["تفاوت", "مقایسه", "بین", "عوامل", "درصد"]):
            recommended_k = 5
            reasons.append("COMPARATIVE_SYNTHESIS_BUDGET")
        elif s.get("RANK") and word_count <= 6:
            recommended_k = 2  # Definition query: single chunk sufficient
            reasons.append("FOCUSED_DEFINITION_BUDGET")
        else:
            recommended_k = 4
            reasons.append("STANDARD_RETRIEVAL_BUDGET")

        # Strategy
        strategy = "hybrid"
        if len(target_entities) >= 1 and any(ch.isdigit() for ch in norm):
            strategy = "hybrid"
            reasons.append("ENTITY_NUMERICAL_HYBRID")

        action = "clarify" if is_ambiguous else "retrieve"
        retrieve_prob = 0.95 if not is_conv else 0.05
        confidence = clarify_prob if is_ambiguous else 0.92

        return PreRetrievalDecision(
            action=action,
            needs_retrieval=True,
            needs_clarification=is_ambiguous,
            missing_slots=missing_slots,
            retrieval_strategy=strategy,
            query_rewrite=norm,
            target_entities=target_entities,
            target_facets=["credit_regulations"],
            top_k=recommended_k,
            confidence=round(confidence, 3),
            retrieve_probability=round(retrieve_prob, 3),
            clarify_probability=round(clarify_prob, 3),
            reason_codes=reasons,
        )

    def post_retrieval(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        state: Optional[Dict[str, Any]] = None,
    ) -> EvidenceDecision:

        if not hits:
            return EvidenceDecision(
                sufficiency="insufficient",
                evidence_score=0.0,
                conflicting_items=[],
                poison_detected=False,
                repetition_detected=False,
                novel_evidence_chunks=0,
                next_action="retrieve_more",
                confidence=0.90,
                sufficiency_probability=0.02,
                reason_codes=[
                    "EMPTY_RETRIEVAL_RESULTS",
                ],
            )

        frame = parse_query(query)

        evidence = build_evidence_set(
            query,
            hits,
            max_evidence=min(
                3,
                len(hits),
            ),
        )

        assessment = assess_sufficiency(
            frame,
            evidence,
        )

        untrusted_items: List[int] = []

        for hit in hits:
            if (
                hit.get("trusted") is False
                or hit.get(
                    "citation_eligible"
                ) is False
            ):
                try:
                    untrusted_items.append(
                        int(
                            hit.get(
                                "doc_id",
                                -1,
                            )
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        repetition_flag = False

        for hit in hits:
            tokens = str(
                hit.get("text") or ""
            ).split()

            if not tokens:
                continue

            from collections import Counter

            counts = Counter(tokens)

            if (
                counts
                and max(
                    counts.values()
                )
                >= 8
            ):
                repetition_flag = True
                break

        reason_codes = list(
            assessment.reasons
        )

        if untrusted_items:
            reason_codes.append(
                "UNTRUSTED_EVIDENCE_DETECTED"
            )

        if repetition_flag:
            reason_codes.append(
                "REPETITION_DETECTED"
            )

        if assessment.conflicts:
            sufficiency = "conflicting"
            next_action = "clarify"
            suff_prob = 0.08

        elif assessment.sufficient:
            sufficiency = "sufficient"
            next_action = "answer"
            suff_prob = 0.90

        else:
            sufficiency = "insufficient"
            next_action = "retrieve_more"
            suff_prob = 0.15

        poison_detected = bool(
            untrusted_items
        )

        evidence_score = (
            len(
                evidence.covered_requirements
            )
            / max(
                1,
                (
                    len(
                        evidence.covered_requirements
                    )
                    + len(
                        evidence.uncovered_requirements
                    )
                ),
            )
        )

        return EvidenceDecision(
            sufficiency=sufficiency,
            evidence_score=round(
                float(evidence_score),
                3,
            ),
            conflicting_items=sorted(
                set(untrusted_items)
            ),
            poison_detected=poison_detected,
            repetition_detected=repetition_flag,
            novel_evidence_chunks=len(
                evidence.hits
            ),
            next_action=next_action,
            confidence=round(
                max(
                    suff_prob,
                    1.0 - suff_prob,
                ),
                3,
            ),
            sufficiency_probability=round(
                suff_prob,
                3,
            ),
            reason_codes=sorted(
                set(reason_codes)
            ),
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class OpenWeightLLMJudge:

    SYSTEM_PROMPT = (
        "You are a fast System-1 RAG Decision Judge for Persian "
        "credit-scoring retrieval. "
        "Return ONLY one valid JSON object matching the requested schema. "
        "Do not include reasoning, markdown, or chain of thought."
    )

    _PRE_ACTIONS = {
        "retrieve",
        "no_retrieval",
        "clarify",
    }

    _POST_SUFFICIENCY = {
        "sufficient",
        "insufficient",
        "conflicting",
        "irrelevant",
        "uncertain",
    }

    _POST_ACTIONS = {
        "answer",
        "retrieve_more",
        "clarify",
        "abstain",
    }

    _RETRIEVAL_STRATEGIES = {
        "hybrid",
        "sparse_only",
        "dense_only",
    }

    _ALLOWED_TOP_K = (
        0,
        2,
        3,
        5,
        8,
    )

    def __init__(
        self,
        model_name: str = (
            "Qwen/Qwen2.5-0.5B-Instruct"
        ),
        fallback: Optional[
            RAGDecisionJudge
        ] = None,
        *,
        local_model_path: Optional[
            str
        ] = None,
        allow_cached_model: bool = False,
        generator: Optional[
            Callable[[str], str]
        ] = None,
        max_new_tokens: int = 192,
        escalation_threshold: float = 0.75,
        force_model: bool = False,
    ):
        self.model_name = model_name
        self.fallback = (
            fallback
            or HeuristicFeatureJudge()
        )

        self.local_model_path = (
            local_model_path
            or os.getenv(
                "RUBBISHRAG_LOCAL_JUDGE_MODEL"
            )
        )

        self.allow_cached_model = bool(
            allow_cached_model
        )

        self.max_new_tokens = int(
            max_new_tokens
        )

        self.escalation_threshold = float(
            escalation_threshold
        )

        self.force_model = bool(
            force_model
        )

        self._generator = generator

        self._model = None
        self._tokenizer = None
        self._load_attempted = False

        self.backend_status = (
            "injected_generator"
            if generator is not None
            else "uninitialized"
        )

        self.last_fallback_reason: Optional[
            str
        ] = None

    @staticmethod
    def _clamp_probability(
        value: Any,
        default: float,
    ) -> float:
        try:
            result = float(value)
        except (
            TypeError,
            ValueError,
        ):
            result = float(default)

        return round(
            min(
                1.0,
                max(
                    0.0,
                    result,
                ),
            ),
            3,
        )

    @staticmethod
    def _strings(
        value: Any,
        default: Optional[
            List[str]
        ] = None,
    ) -> List[str]:
        if not isinstance(
            value,
            list,
        ):
            return list(
                default or []
            )

        return [
            str(item)
            for item in value
            if item is not None
        ]

    @staticmethod
    def _ints(
        value: Any,
        default: Optional[
            List[int]
        ] = None,
    ) -> List[int]:
        if not isinstance(
            value,
            list,
        ):
            return list(
                default or []
            )

        result = []

        for item in value:
            try:
                result.append(
                    int(item)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        return result

    @classmethod
    def _nearest_top_k(
        cls,
        value: Any,
        default: int,
    ) -> int:
        try:
            requested = int(value)
        except (
            TypeError,
            ValueError,
        ):
            requested = int(default)

        return min(
            cls._ALLOWED_TOP_K,
            key=lambda k: abs(
                k - requested
            ),
        )

    @staticmethod
    def _extract_json_object(
        raw: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        if not isinstance(
            raw,
            str,
        ):
            return None

        text = raw.strip()

        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

        start = text.find("{")
        end = text.rfind("}")

        if (
            start < 0
            or end < start
        ):
            return None

        try:
            payload = json.loads(
                text[
                    start : end + 1
                ]
            )
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return None

        if not isinstance(
            payload,
            dict,
        ):
            return None

        return payload

    def _model_source(
        self,
    ) -> Optional[str]:
        if self.local_model_path:
            return self.local_model_path

        if self.allow_cached_model:
            return self.model_name

        return None

    def _ensure_local_backend(
        self,
    ) -> bool:

        if self._generator is not None:
            self.backend_status = (
                "injected_generator"
            )
            return True

        if self._model is not None:
            return True

        if self._load_attempted:
            return False

        self._load_attempted = True

        source = self._model_source()

        if not source:
            self.backend_status = (
                "fallback:no_local_model"
            )
            self.last_fallback_reason = (
                "NO_LOCAL_MODEL_CONFIGURED"
            )
            return False

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except Exception as exc:
            self.backend_status = (
                "fallback:"
                + type(exc).__name__
            )
            self.last_fallback_reason = (
                "TRANSFORMERS_UNAVAILABLE"
            )
            return False

        try:
            tokenizer = (
                AutoTokenizer.from_pretrained(
                    source,
                    local_files_only=True,
                )
            )

            model = (
                AutoModelForCausalLM.from_pretrained(
                    source,
                    local_files_only=True,
                )
            )

            model.eval()

            self._tokenizer = tokenizer
            self._model = model

            self.backend_status = (
                "local_transformers"
            )
            self.last_fallback_reason = (
                None
            )

            return True

        except Exception as exc:
            self._tokenizer = None
            self._model = None

            self.backend_status = (
                "fallback:"
                + type(exc).__name__
            )

            self.last_fallback_reason = (
                "LOCAL_MODEL_LOAD_FAILED"
            )

            return False

    def _generate(
        self,
        prompt: str,
    ) -> Optional[str]:
        if self._generator is not None:
            try:
                return str(
                    self._generator(
                        prompt
                    )
                )
            except Exception:
                self.last_fallback_reason = (
                    "INJECTED_GENERATOR_FAILED"
                )
                return None

        if not self._ensure_local_backend():
            return None

        assert (
            self._tokenizer is not None
        )
        assert self._model is not None

        tokenizer = self._tokenizer
        model = self._model

        messages = [
            {
                "role": "system",
                "content": (
                    self.SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            if hasattr(
                tokenizer,
                "apply_chat_template",
            ):
                rendered = (
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                rendered = (
                    self.SYSTEM_PROMPT
                    + "\n\n"
                    + prompt
                )

            inputs = tokenizer(
                rendered,
                return_tensors="pt",
            )

            try:
                device = next(
                    model.parameters()
                ).device

                inputs = {
                    key: (
                        value.to(device)
                        if hasattr(
                            value,
                            "to",
                        )
                        else value
                    )
                    for key, value
                    in inputs.items()
                }
            except Exception:
                pass

            outputs = model.generate(
                **inputs,
                max_new_tokens=(
                    self.max_new_tokens
                ),
                do_sample=False,
                pad_token_id=getattr(
                    tokenizer,
                    "eos_token_id",
                    None,
                ),
            )

            input_len = int(
                inputs[
                    "input_ids"
                ].shape[-1]
            )

            generated = outputs[
                0
            ][input_len:]

            return tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )

        except Exception:
            self.last_fallback_reason = (
                "LOCAL_GENERATION_FAILED"
            )
            return None

    def _request_json(
        self,
        prompt: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        raw = self._generate(
            prompt
        )

        if raw is None:
            return None

        payload = (
            self._extract_json_object(
                raw
            )
        )

        if payload is None:
            self.last_fallback_reason = (
                "INVALID_LOCAL_MODEL_JSON"
            )

        return payload

    def pre_retrieval(
        self,
        query: str,
        conversation: Optional[
            List[Dict[str, Any]]
        ] = None,
        slots: Optional[
            Dict[str, Any]
        ] = None,
    ) -> PreRetrievalDecision:
        base = (
            self.fallback.pre_retrieval(
                query,
                conversation,
                slots,
            )
        )

        if (
            not self.force_model
            and base.confidence
            >= self.escalation_threshold
        ):
            self.backend_status = (
                "heuristic_fast_path"
            )
            self.last_fallback_reason = None

            return dataclasses.replace(
                base,
                reason_codes=sorted(
                    set(
                        list(
                            base.reason_codes
                        )
                        + [
                            "HEURISTIC_FAST_PATH",
                        ]
                    )
                ),
            )

        prompt = (
            "Perform PRE_RETRIEVAL routing.\n"
            "Return JSON with exactly these semantic fields:\n"
            "{"
            '"action":"retrieve|no_retrieval|clarify",'
            '"missing_slots":[],'
            '"retrieval_strategy":"hybrid|sparse_only|dense_only",'
            '"query_rewrite":"...",'
            '"target_entities":[],'
            '"target_facets":[],'
            '"top_k":0,'
            '"confidence":0.0,'
            '"retrieve_probability":0.0,'
            '"clarify_probability":0.0,'
            '"reason_codes":[]'
            "}\n\n"
            "Allowed top_k values: 0,2,3,5,8.\n"
            f"Query: {query}\n"
            f"Slots: {json.dumps(slots or {}, ensure_ascii=False)}"
        )

        payload = (
            self._request_json(
                prompt
            )
        )

        if payload is None:
            return base

        action = str(
            payload.get(
                "action",
                "",
            )
        ).strip().lower()

        if action not in self._PRE_ACTIONS:
            self.last_fallback_reason = (
                "INVALID_PRE_ACTION"
            )
            return base

        strategy = str(
            payload.get(
                "retrieval_strategy",
                base.retrieval_strategy,
            )
        ).strip().lower()

        if (
            strategy
            not in self._RETRIEVAL_STRATEGIES
        ):
            strategy = (
                base.retrieval_strategy
            )

        if action == "retrieve":
            top_k = (
                self._nearest_top_k(
                    payload.get(
                        "top_k",
                        base.top_k,
                    ),
                    max(
                        2,
                        base.top_k,
                    ),
                )
            )

            if top_k == 0:
                top_k = 2
        else:
            top_k = 0

        reasons = self._strings(
            payload.get(
                "reason_codes"
            ),
            base.reason_codes,
        )

        reasons.append(
            "OPEN_WEIGHT_LOCAL_DECISION"
        )

        return PreRetrievalDecision(
            action=action,
            needs_retrieval=(
                action == "retrieve"
            ),
            needs_clarification=(
                action == "clarify"
            ),
            missing_slots=self._strings(
                payload.get(
                    "missing_slots"
                ),
                base.missing_slots,
            ),
            retrieval_strategy=strategy,
            query_rewrite=str(
                payload.get(
                    "query_rewrite",
                    base.query_rewrite,
                )
                or query
            ),
            target_entities=self._strings(
                payload.get(
                    "target_entities"
                ),
                base.target_entities,
            ),
            target_facets=self._strings(
                payload.get(
                    "target_facets"
                ),
                base.target_facets,
            ),
            top_k=top_k,
            confidence=(
                self._clamp_probability(
                    payload.get(
                        "confidence"
                    ),
                    base.confidence,
                )
            ),
            retrieve_probability=(
                self._clamp_probability(
                    payload.get(
                        "retrieve_probability"
                    ),
                    base.retrieve_probability,
                )
            ),
            clarify_probability=(
                self._clamp_probability(
                    payload.get(
                        "clarify_probability"
                    ),
                    base.clarify_probability,
                )
            ),
            reason_codes=sorted(
                set(reasons)
            ),
        )

    def post_retrieval(
        self,
        query: str,
        hits: List[
            Dict[str, Any]
        ],
        state: Optional[
            Dict[str, Any]
        ] = None,
    ) -> EvidenceDecision:
        base = (
            self.fallback.post_retrieval(
                query,
                hits,
                state,
            )
        )

        if (
            not self.force_model
            and base.confidence
            >= self.escalation_threshold
        ):
            self.backend_status = (
                "heuristic_fast_path"
            )
            self.last_fallback_reason = None

            return dataclasses.replace(
                base,
                reason_codes=sorted(
                    set(
                        list(
                            base.reason_codes
                        )
                        + [
                            "HEURISTIC_FAST_PATH",
                        ]
                    )
                ),
            )

        compact_hits = []

        for hit in hits[:8]:
            compact_hits.append(
                {
                    "doc_id": (
                        hit.get(
                            "doc_id"
                        )
                    ),
                    "trusted": (
                        hit.get(
                            "trusted"
                        )
                    ),
                    "citation_eligible": (
                        hit.get(
                            "citation_eligible"
                        )
                    ),
                    "text": str(
                        hit.get(
                            "brief_answer"
                        )
                        or hit.get(
                            "text"
                        )
                        or ""
                    )[:600],
                }
            )

        prompt = (
            "Perform POST_RETRIEVAL collective evidence assessment.\n"
            "Return JSON with these fields:\n"
            "{"
            '"sufficiency":"sufficient|insufficient|conflicting|irrelevant|uncertain",'
            '"evidence_score":0.0,'
            '"conflicting_items":[],'
            '"poison_detected":false,'
            '"repetition_detected":false,'
            '"novel_evidence_chunks":0,'
            '"next_action":"answer|retrieve_more|clarify|abstain",'
            '"confidence":0.0,'
            '"sufficiency_probability":0.0,'
            '"reason_codes":[]'
            "}\n\n"
            f"Query: {query}\n"
            "Hits: "
            + json.dumps(
                compact_hits,
                ensure_ascii=False,
            )
        )

        payload = (
            self._request_json(
                prompt
            )
        )

        if payload is None:
            return base

        sufficiency = str(
            payload.get(
                "sufficiency",
                "",
            )
        ).strip().lower()

        next_action = str(
            payload.get(
                "next_action",
                "",
            )
        ).strip().lower()

        if (
            sufficiency
            not in self._POST_SUFFICIENCY
            or next_action
            not in self._POST_ACTIONS
        ):
            self.last_fallback_reason = (
                "INVALID_POST_DECISION"
            )
            return base

        try:
            evidence_score = float(
                payload.get(
                    "evidence_score",
                    base.evidence_score,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            evidence_score = float(
                base.evidence_score
            )

        try:
            novel_count = int(
                payload.get(
                    "novel_evidence_chunks",
                    base.novel_evidence_chunks,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            novel_count = int(
                base.novel_evidence_chunks
            )

        reasons = self._strings(
            payload.get(
                "reason_codes"
            ),
            base.reason_codes,
        )

        reasons.append(
            "OPEN_WEIGHT_LOCAL_DECISION"
        )

        return EvidenceDecision(
            sufficiency=sufficiency,
            evidence_score=round(
                evidence_score,
                3,
            ),
            conflicting_items=self._ints(
                payload.get(
                    "conflicting_items"
                ),
                base.conflicting_items,
            ),
            poison_detected=bool(
                payload.get(
                    "poison_detected",
                    base.poison_detected,
                )
            ),
            repetition_detected=bool(
                payload.get(
                    "repetition_detected",
                    base.repetition_detected,
                )
            ),
            novel_evidence_chunks=max(
                0,
                novel_count,
            ),
            next_action=next_action,
            confidence=(
                self._clamp_probability(
                    payload.get(
                        "confidence"
                    ),
                    base.confidence,
                )
            ),
            sufficiency_probability=(
                self._clamp_probability(
                    payload.get(
                        "sufficiency_probability"
                    ),
                    base.sufficiency_probability,
                )
            ),
            reason_codes=sorted(
                set(reasons)
            ),
        )


# ---------------------------------------------------------------------------
# 5. Dynamic Context Budgeting / Adaptive Stopping Policy
# ---------------------------------------------------------------------------

def _adaptive_candidate_is_eligible(
    hit: Dict[str, Any],
) -> bool:

    if (
        hit.get("trusted") is False
        or hit.get("citation_eligible") is False
    ):
        return False

    candidate_source = str(
        hit.get("candidate_source")
        or ""
    )

    if (
        candidate_source
        == "remote_unverified_test"
    ):
        return (
            hit.get("trusted") is True
            and hit.get(
                "citation_eligible"
            ) is True
        )

    try:
        doc_id = int(
            hit.get(
                "doc_id",
                -1,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False

    from .corpus import get_registry

    registry = get_registry()

    return registry.is_trusted(
        doc_id
    )


def adaptive_select_hits(
    query: str,
    ranked_hits: List[Dict[str, Any]],
    judge: RAGDecisionJudge,
    max_k: int = 8,
    min_k: int = 2,
) -> Tuple[List[Dict[str, Any]], EvidenceDecision]:
    """Dynamically select chunks, stopping early when evidence is sufficient.
    
    Prevents context dilution and token bloat by terminating retrieval once
    the judge confirms collective evidence sufficiency.
    """
    selected: List[Dict[str, Any]] = []
    last_decision: Optional[EvidenceDecision] = None

    for hit in ranked_hits[:max_k]:
        if not _adaptive_candidate_is_eligible(
            hit
        ):
            continue

        selected.append(hit)
        if len(selected) >= min_k:
            decision = judge.post_retrieval(query, selected)
            last_decision = decision
            if decision.sufficiency == "sufficient" and decision.confidence >= 0.80:
                break

    if last_decision is None:
        last_decision = judge.post_retrieval(query, selected)

    return selected, last_decision




@dataclasses.dataclass
class AdaptiveLoopResult:

    action: str
    pre_decision: PreRetrievalDecision
    evidence_decision: Optional[EvidenceDecision]
    selected_hits: List[Dict[str, Any]]
    rounds: int
    requested_top_ks: List[int]
    total_candidates_seen: int
    escalated: bool
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def escalate_pre_retrieval_decision(
    decision: PreRetrievalDecision,
    *,
    confidence_threshold: float = 0.75,
    max_k: int = 8,
) -> Tuple[PreRetrievalDecision, bool]:

    if decision.confidence >= confidence_threshold:
        return decision, False

    reasons = list(decision.reason_codes)

    if decision.needs_clarification:
        reasons.append(
            "LOW_CONFIDENCE_EXPLICIT_CLARIFICATION"
        )
        return (
            dataclasses.replace(
                decision,
                action="clarify",
                needs_clarification=True,
                reason_codes=sorted(set(reasons)),
            ),
            True,
        )

    broadened_k = min(
        max_k,
        max(
            5,
            int(decision.top_k or 0) + 2,
        ),
    )

    reasons.append(
        "LOW_CONFIDENCE_BROADER_RETRIEVAL"
    )

    return (
        dataclasses.replace(
            decision,
            action="retrieve",
            needs_retrieval=True,
            needs_clarification=False,
            retrieval_strategy="hybrid",
            top_k=broadened_k,
            reason_codes=sorted(set(reasons)),
        ),
        True,
    )


def _adaptive_hit_identity(
    hit: Dict[str, Any],
) -> Tuple[str, str]:

    if hit.get("doc_id") is not None:
        return (
            "doc_id",
            str(hit.get("doc_id")),
        )

    return (
        "text",
        str(
            hit.get("text")
            or hit.get("question")
            or hit.get("brief_answer")
            or ""
        ),
    )


def run_bounded_adaptive_loop(
    query: str,
    retriever: Callable[
        [str, int],
        List[Dict[str, Any]],
    ],
    *,
    judge: Optional[RAGDecisionJudge] = None,
    max_rounds: int = 2,
    max_k: int = 8,
    min_k: int = 1,
    confidence_threshold: Optional[float] = None,
) -> AdaptiveLoopResult:

    if not 1 <= int(max_rounds) <= 2:
        raise ValueError(
            "max_rounds must be between 1 and 2"
        )

    active_judge = (
        judge
        or HeuristicFeatureJudge()
    )

    threshold = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else float(
            getattr(
                active_judge,
                "confidence_threshold",
                0.75,
            )
        )
    )

    pre = active_judge.pre_retrieval(
        query
    )

    routed, escalated = (
        escalate_pre_retrieval_decision(
            pre,
            confidence_threshold=threshold,
            max_k=max_k,
        )
    )

    base_reasons = list(
        routed.reason_codes
    )

    if routed.action == "no_retrieval":
        return AdaptiveLoopResult(
            action="no_retrieval",
            pre_decision=routed,
            evidence_decision=None,
            selected_hits=[],
            rounds=0,
            requested_top_ks=[],
            total_candidates_seen=0,
            escalated=escalated,
            reason_codes=sorted(
                set(
                    base_reasons
                    + ["FAST_NO_RETRIEVAL"]
                )
            ),
        )

    if (
        routed.action == "clarify"
        or routed.needs_clarification
    ):
        return AdaptiveLoopResult(
            action="clarify",
            pre_decision=routed,
            evidence_decision=None,
            selected_hits=[],
            rounds=0,
            requested_top_ks=[],
            total_candidates_seen=0,
            escalated=escalated,
            reason_codes=sorted(
                set(
                    base_reasons
                    + ["PRE_RETRIEVAL_CLARIFICATION"]
                )
            ),
        )

    requested_top_ks: List[int] = []
    accumulated: List[
        Dict[str, Any]
    ] = []
    seen = set()

    current_k = max(
        min_k,
        min(
            max_k,
            int(
                routed.top_k
                or min_k
            ),
        ),
    )

    last_decision: Optional[
        EvidenceDecision
    ] = None

    last_selected: List[
        Dict[str, Any]
    ] = []

    for round_index in range(
        1,
        int(max_rounds) + 1,
    ):
        requested_top_ks.append(
            current_k
        )

        batch = list(
            retriever(
                query,
                current_k,
            )
            or []
        )

        for hit in batch:
            identity = (
                _adaptive_hit_identity(
                    hit
                )
            )

            if identity in seen:
                continue

            seen.add(identity)
            accumulated.append(hit)

        if accumulated:
            selected, decision = (
                adaptive_select_hits(
                    query,
                    accumulated,
                    active_judge,
                    max_k=min(
                        max_k,
                        len(accumulated),
                    ),
                    min_k=min(
                        min_k,
                        len(accumulated),
                    ),
                )
            )
        else:
            selected = []
            decision = (
                active_judge.post_retrieval(
                    query,
                    [],
                )
            )

        last_selected = selected
        last_decision = decision

        if (
            decision.sufficiency
            == "sufficient"
            and decision.confidence
            >= threshold
        ):
            return AdaptiveLoopResult(
                action="answer",
                pre_decision=routed,
                evidence_decision=decision,
                selected_hits=selected,
                rounds=round_index,
                requested_top_ks=(
                    requested_top_ks
                ),
                total_candidates_seen=len(
                    accumulated
                ),
                escalated=escalated,
                reason_codes=sorted(
                    set(
                        base_reasons
                        + decision.reason_codes
                        + [
                            "SUFFICIENT_EARLY_STOP",
                        ]
                    )
                ),
            )

        if (
            decision.sufficiency
            in {
                "conflicting",
                "irrelevant",
            }
            or decision.next_action
            == "abstain"
        ):
            return AdaptiveLoopResult(
                action="abstain",
                pre_decision=routed,
                evidence_decision=decision,
                selected_hits=selected,
                rounds=round_index,
                requested_top_ks=(
                    requested_top_ks
                ),
                total_candidates_seen=len(
                    accumulated
                ),
                escalated=escalated,
                reason_codes=sorted(
                    set(
                        base_reasons
                        + decision.reason_codes
                        + [
                            "CONFLICT_OR_IRRELEVANCE_STOP",
                        ]
                    )
                ),
            )

        if (
            decision.next_action
            == "clarify"
        ):
            return AdaptiveLoopResult(
                action="clarify",
                pre_decision=routed,
                evidence_decision=decision,
                selected_hits=selected,
                rounds=round_index,
                requested_top_ks=(
                    requested_top_ks
                ),
                total_candidates_seen=len(
                    accumulated
                ),
                escalated=escalated,
                reason_codes=sorted(
                    set(
                        base_reasons
                        + decision.reason_codes
                        + [
                            "POST_RETRIEVAL_CLARIFICATION",
                        ]
                    )
                ),
            )

        current_k = min(
            max_k,
            max(
                current_k + 2,
                current_k * 2,
            ),
        )

    return AdaptiveLoopResult(
        action="abstain",
        pre_decision=routed,
        evidence_decision=last_decision,
        selected_hits=last_selected,
        rounds=int(max_rounds),
        requested_top_ks=requested_top_ks,
        total_candidates_seen=len(
            accumulated
        ),
        escalated=escalated,
        reason_codes=sorted(
            set(
                base_reasons
                + (
                    last_decision.reason_codes
                    if last_decision
                    else []
                )
                + [
                    "BOUNDED_RETRIEVAL_EXHAUSTED",
                ]
            )
        ),
    )


# ---------------------------------------------------------------------------
# 6. Quantitative Calibration Metrics (Brier Score, ECE, Risk-Coverage)
# ---------------------------------------------------------------------------

def compute_brier_score(probabilities: List[float], outcomes: List[int]) -> float:
    """Calculate mean squared error between probabilities and binary outcomes (0 or 1)."""
    if not probabilities or len(probabilities) != len(outcomes):
        return 0.0
    total = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes))
    return round(total / len(probabilities), 4)


def compute_ece(probabilities: List[float], outcomes: List[int], n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across probability bins."""
    if not probabilities or len(probabilities) != len(outcomes):
        return 0.0

    bin_size = 1.0 / n_bins
    total_samples = len(probabilities)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size
        bin_indices = [
            idx for idx, p in enumerate(probabilities)
            if bin_lower <= p < bin_upper or (i == n_bins - 1 and p == 1.0)
        ]

        if not bin_indices:
            continue

        bin_acc = sum(outcomes[idx] for idx in bin_indices) / len(bin_indices)
        bin_conf = sum(probabilities[idx] for idx in bin_indices) / len(bin_indices)
        weight = len(bin_indices) / total_samples
        ece += weight * abs(bin_acc - bin_conf)

    return round(ece, 4)


# ---------------------------------------------------------------------------
# 7. Benchmark & Evaluation Routine
# ---------------------------------------------------------------------------

def evaluate_judge(
    judge: Optional[RAGDecisionJudge] = None,
    dataset_path: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Run comprehensive quantitative evaluation of the Decision Judge."""
    judge = judge or HeuristicFeatureJudge()
    ds_path = dataset_path or os.path.join(BASE, "data", "judge_dataset.json")

    if not os.path.exists(ds_path):
        return {"error": f"Dataset not found at {ds_path}"}

    with open(ds_path, encoding="utf-8") as f:
        data = json.load(f)

    if split:
        data = [d for d in data if d.get("split") == split]

    if not data:
        return {"error": f"No records found for split={split}"}

    # Tracking metrics
    routing_correct = 0
    clarify_tp = 0
    clarify_fp = 0
    clarify_fn = 0
    clarify_tn = 0
    suff_correct = 0

    probs_retrieval = []
    labels_retrieval = []

    probs_clarify = []
    labels_clarify = []

    k_values = []
    total = len(data)

    for item in data:
        query = item["query"]
        exp_ret = item.get("expected_retrieval", "RETRIEVE")
        exp_cla = item.get("expected_clarification", "DO_NOT_CLARIFY")
        exp_suff = item.get("expected_sufficiency", "SUFFICIENT")

        # Pre-retrieval
        pre = judge.pre_retrieval(query)
        pred_ret = "NO_RETRIEVAL" if not pre.needs_retrieval else "RETRIEVE"
        pred_cla = "CLARIFY" if pre.needs_clarification else "DO_NOT_CLARIFY"

        if pred_ret == exp_ret:
            routing_correct += 1

        probs_retrieval.append(pre.retrieve_probability)
        labels_retrieval.append(1 if exp_ret == "RETRIEVE" else 0)

        probs_clarify.append(pre.clarify_probability)
        labels_clarify.append(1 if exp_cla == "CLARIFY" else 0)

        if pred_cla == "CLARIFY" and exp_cla == "CLARIFY":
            clarify_tp += 1
        elif pred_cla == "CLARIFY" and exp_cla != "CLARIFY":
            clarify_fp += 1
        elif pred_cla != "CLARIFY" and exp_cla == "CLARIFY":
            clarify_fn += 1
        else:
            clarify_tn += 1

        k_values.append(pre.top_k)

        # Mock retrieval hits for sufficiency check
        mock_hits = [{"doc_id": 1, "fused": 0.85, "text": "رتبه اعتباری در بازه..."}] if pre.needs_retrieval else []
        post = judge.post_retrieval(query, mock_hits)
        if post.sufficiency.upper() == exp_suff.upper():
            suff_correct += 1

    cla_prec = clarify_tp / max(clarify_tp + clarify_fp, 1)
    cla_rec = clarify_tp / max(clarify_tp + clarify_fn, 1)
    cla_f1 = (2 * cla_prec * cla_rec) / max(cla_prec + cla_rec, 1e-6)

    mean_k = sum(k_values) / max(len(k_values), 1)
    baseline_k = 5.0
    token_savings = ((baseline_k - mean_k) / baseline_k) * 100.0

    brier_ret = compute_brier_score(probs_retrieval, labels_retrieval)
    ece_ret = compute_ece(probs_retrieval, labels_retrieval)

    return {
        "total_queries": total,
        "split": split or "all",
        "routing_accuracy": round(routing_correct / total, 3),
        "clarification": {
            "precision": round(cla_prec, 3),
            "recall": round(cla_rec, 3),
            "f1": round(cla_f1, 3),
            "tp": clarify_tp,
            "fp": clarify_fp,
            "fn": clarify_fn,
        },
        "sufficiency_accuracy": round(suff_correct / total, 3),
        "context_budget": {
            "mean_top_k": round(mean_k, 2),
            "baseline_fixed_k": 5,
            "token_savings_pct": round(token_savings, 1),
        },
        "calibration": {
            "brier_score": brier_ret,
            "expected_calibration_error": ece_ret,
        },
    }
