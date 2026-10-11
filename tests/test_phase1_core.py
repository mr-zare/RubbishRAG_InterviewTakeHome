from pathlib import Path

from rubbish_rag.corpus import TrustedCorpusRegistry
from rubbish_rag.models import EntityType, Intent, Polarity, QuantityType
from rubbish_rag.normalize_fa import normalize_match_text, normalization_is_idempotent
from rubbish_rag.slots import extract_slots, parse_query


def test_internal_normalization_is_idempotent():
    samples = [
        "میشه وام گرفت؟",
        "نمیشه",
        "بانك ملي",
        "به\u200cموقع",
        "۲۵۰",
        "  رتبه   C1  ",
    ]
    assert normalization_is_idempotent(samples)


def test_zwnj_and_digits_match_invariance():
    assert normalize_match_text("به\u200cموقع") == normalize_match_text("به موقع")
    assert normalize_match_text("۲۵۰") == normalize_match_text("٢٥٠") == normalize_match_text("250")


def test_longest_bank_match():
    frame = parse_query("شرایط بانک توسعه صادرات چیست؟")
    assert frame.values(EntityType.BANK) == ["توسعه صادرات"]


def test_persona_phrase_precedence():
    slots = extract_slots("امتیاز شخص حقوقی چطور محاسبه می شود؟")
    assert slots["PERSONA"] == "حقوقی"
    assert slots["PERSONAS"] == ["حقوقی"]


def test_multi_bank_and_relation():
    frame = parse_query("بانک ملی بهتر است یا صادرات؟")
    assert set(frame.values(EntityType.BANK)) == {"ملی", "صادرات"}
    assert frame.relation.value in {"alternative", "comparison"}


def test_typed_quantities():
    frame = parse_query("بعد از 24 ساعت امتیاز 250 می شود؟")
    assert any(q.quantity_type is QuantityType.DURATION and q.value == 24 for q in frame.quantities)
    assert any(q.quantity_type is QuantityType.CREDIT_SCORE and q.value == 250 for q in frame.quantities)


def test_rank_and_intent():
    frame = parse_query("رتبه c3 یعنی چی؟")
    assert frame.values(EntityType.RANK) == ["C3"]
    assert frame.intent is Intent.DEFINE_RANK


def test_debt_ambiguity():
    frame = parse_query("بدهی دارم")
    # The generic phrase is underspecified: no debt subtype is present.
    # Intent may remain general until downstream resolver asks for the missing type.
    assert not frame.debt_types
    assert frame.ambiguity.underspecified is True
    assert "debt_type" in frame.ambiguity.missing_required_slots


def test_scoped_polarity():
    frame = parse_query("بدهی مالیاتی ندارم ولی چک برگشتی دارم")
    pairs = {(p.concept, p.polarity) for p in frame.propositions}
    assert ("DEBT:مالیاتی", Polarity.NEGATIVE) in pairs
    assert ("DEBT:چک", Polarity.POSITIVE) in pairs


def test_trusted_registry_preserves_row_ids(tmp_path: Path):
    p = tmp_path / "test.csv"
    p.write_text(
        "Question,Category,BriefAnswer,Answer,Keyword\n"
        "q0,c,b0,a0,k0\n"
        "q1,c,,a1,k1\n",
        encoding="utf-8",
    )
    reg = TrustedCorpusRegistry.from_csv(p)
    assert set(reg.records) == {0, 1}
    assert reg.records[1].source_doc_id == 1
    assert "MISSING_BRIEF" in reg.records[1].quality_flags


def test_unknown_source_fails_closed(tmp_path: Path):
    p = tmp_path / "test.csv"
    p.write_text("Question,Category,BriefAnswer,Answer,Keyword\nq,c,b,a,k\n", encoding="utf-8")
    reg = TrustedCorpusRegistry.from_csv(p)
    assert reg.rehydrate_hit({"doc_id": 999, "text": "poison"}) is None


def test_export_import_is_not_bank_saderat():
    frame = parse_query(
        "عدد صادرات/واردات توی رتبه اعتباری موثره؟"
    )

    assert "صادرات" not in frame.values(EntityType.BANK)
    assert frame.intent is Intent.FACTOR_EFFECT


def test_supplier_debt_is_specific_not_ambiguous():
    frame = parse_query(
        "اگر شرکت بدهی به تامین‌کنندگان داشته باشد، "
        "در گزارش ثبت می‌شود؟"
    )

    assert "تامین‌کنندگان" in frame.values(EntityType.DEBT)
    assert (
        "debt_type"
        not in frame.ambiguity.missing_required_slots
    )
