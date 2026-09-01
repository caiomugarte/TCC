from __future__ import annotations

from typing import Mapping

from app.schemas.profile import (
    QUESTION_DIMENSIONS,
    QUESTION_OPTIONS,
    QUESTION_WEIGHTS,
    ComputedProfile,
    ProfileSubmission,
)

ANCHOR_SCORES = {
    "conservador": 0.0,
    "moderado": 0.5,
    "arrojado": 1.0,
}


def _rounded(value: float) -> float:
    return round(value, 4)


def _dimension_score(
    answers: Mapping[str, str | list[str]],
    dimension: str,
) -> float:
    total = 0.0
    weight_total = 0.0
    for question, question_dimension in QUESTION_DIMENSIONS.items():
        if question_dimension != dimension:
            continue
        value = answers[question]
        assert isinstance(value, str)
        total += QUESTION_OPTIONS[question][value] * QUESTION_WEIGHTS[question]
        weight_total += QUESTION_WEIGHTS[question]
    return _rounded(total / weight_total)


def _nearest_anchor(score: float) -> str:
    return min(ANCHOR_SCORES, key=lambda name: abs(ANCHOR_SCORES[name] - score))


def compute_profile(submission: ProfileSubmission) -> ComputedProfile:
    answers = submission.answers
    dimensions = {
        "apetite": _dimension_score(answers, "apetite"),
        "capacidade": _dimension_score(answers, "capacidade"),
        "liquidez": _dimension_score(answers, "liquidez"),
        "conhecimento": _dimension_score(answers, "conhecimento"),
    }
    raw_score = dimensions["apetite"]
    score = min(
        dimensions["apetite"],
        dimensions["capacidade"],
        dimensions["liquidez"],
    )
    rules: list[str] = []
    warnings: list[str] = []

    def cap_score(value: float, message: str) -> None:
        nonlocal score
        score = min(score, value)
        rules.append(message)

    if answers["horizonte"] == "ate_2_anos":
        cap_score(0.25, "horizonte de até 2 anos: score limitado a 0,25")
    if QUESTION_OPTIONS["liquidez"][answers["liquidez"]] <= 0.2:
        cap_score(0.35, "necessidade de liquidez em até 1 ano: score limitado a 0,35")
    if answers["capacidade"] == "nenhuma":
        cap_score(0.25, "sem margem declarada para manter a parcela: score limitado a 0,25")
    if answers["concentracao"] == "mais_de_80":
        cap_score(0.25, "mais de 80% do patrimônio nesta parcela: score limitado a 0,25")
    if answers["necessidade_futura"] in {"30_a_60", "mais_de_60"}:
        cap_score(
            0.25,
            "mais de 30% do patrimônio tem uso planejado em até 3 anos: score limitado a 0,25",
        )
    if dimensions["conhecimento"] < 0.5:
        warnings.append(
            "conhecimento abaixo da faixa intermediária: revisar produtos elegíveis antes de operar ações"
        )

    restrictions = answers["restricoes"]
    assert isinstance(restrictions, list)
    if "nenhuma" in restrictions and len(restrictions) > 1:
        warnings.append("restrição 'nenhuma' foi selecionada junto com outras preferências")
    elif "nenhuma" not in restrictions:
        warnings.append(
            "restrições registradas: aplicar manualmente na alocação e nos produtos elegíveis"
        )

    rounded_score = _rounded(score)
    return ComputedProfile(
        answers=answers,
        investable_capital_brl=submission.investable_capital_brl,
        consented=submission.consented,
        raw_score=_rounded(raw_score),
        score=rounded_score,
        dimensions=dimensions,
        generic_profile=_nearest_anchor(rounded_score),
        rules=rules,
        warnings=warnings,
    )
