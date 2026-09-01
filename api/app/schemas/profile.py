from __future__ import annotations

from datetime import datetime
import math
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GenericProfile: TypeAlias = Literal["conservador", "moderado", "arrojado"]
AnswerValue: TypeAlias = str | list[str]

PROFILE_VERSION = 1


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ProfileSchema(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


QUESTION_OPTIONS: dict[str, dict[str, float]] = {
    "objetivo": {
        "preservacao": 0.0,
        "renda": 0.25,
        "equilibrio": 0.5,
        "crescimento": 0.8,
        "crescimento_agressivo": 1.0,
    },
    "horizonte": {
        "ate_2_anos": 0.0,
        "2_a_5_anos": 0.33,
        "5_a_10_anos": 0.67,
        "mais_de_10_anos": 1.0,
    },
    "capacidade": {
        "nenhuma": 0.0,
        "menos_de_10": 0.25,
        "10_a_30": 0.5,
        "30_a_60": 0.75,
        "mais_de_60": 1.0,
    },
    "reacao": {
        "vender_tudo": 0.0,
        "vender_parte": 0.25,
        "manter": 0.5,
        "comprar": 0.75,
        "comprar_mais": 1.0,
    },
    "perda": {
        "ate_5": 0.0,
        "5_a_10": 0.25,
        "10_a_20": 0.5,
        "20_a_35": 0.75,
        "mais_de_35": 1.0,
    },
    "experiencia": {
        "nenhuma": 0.0,
        "basica": 0.25,
        "intermediaria": 0.5,
        "avancada": 0.75,
        "profissional": 1.0,
    },
    "liquidez": {
        "a_qualquer_momento": 0.0,
        "ate_1_ano": 0.2,
        "1_a_3_anos": 0.5,
        "mais_de_3_anos": 0.8,
        "sem_previsao": 1.0,
    },
    "renda": {
        "ate_3k": 0.0,
        "3_a_8k": 0.25,
        "8_a_20k": 0.5,
        "20_a_50k": 0.75,
        "mais_de_50k": 1.0,
    },
    "patrimonio": {
        "ate_50k": 0.0,
        "50_a_200k": 0.25,
        "200k_a_1m": 0.5,
        "1m_a_5m": 0.75,
        "mais_de_5m": 1.0,
    },
    "concentracao": {
        "ate_10": 1.0,
        "10_a_30": 0.75,
        "30_a_60": 0.5,
        "60_a_80": 0.25,
        "mais_de_80": 0.0,
    },
    "necessidade_futura": {
        "nenhuma": 1.0,
        "ate_10": 0.75,
        "10_a_30": 0.5,
        "30_a_60": 0.25,
        "mais_de_60": 0.0,
    },
    "produtos": {
        "nenhum": 0.0,
        "renda_fixa_fundos": 0.25,
        "etf_fii_acoes": 0.5,
        "cripto": 0.75,
        "complexos": 1.0,
    },
    "operacoes": {
        "nenhuma": 0.0,
        "inicial": 0.25,
        "ocasional": 0.5,
        "regular": 0.75,
        "frequente": 1.0,
    },
    "formacao": {
        "nenhuma": 0.0,
        "autodidata": 0.25,
        "academica": 0.5,
        "profissional_indireta": 0.75,
        "profissional_direta": 1.0,
    },
}

QUESTION_DIMENSIONS = {
    "objetivo": "apetite",
    "horizonte": "liquidez",
    "capacidade": "capacidade",
    "reacao": "apetite",
    "perda": "apetite",
    "experiencia": "conhecimento",
    "liquidez": "liquidez",
    "renda": "capacidade",
    "patrimonio": "capacidade",
    "concentracao": "capacidade",
    "necessidade_futura": "capacidade",
    "produtos": "conhecimento",
    "operacoes": "conhecimento",
    "formacao": "conhecimento",
}

QUESTION_WEIGHTS = {
    "objetivo": 1.2,
    "horizonte": 1.3,
    "capacidade": 1.3,
    "reacao": 2.0,
    "perda": 1.8,
    "experiencia": 1.0,
    "liquidez": 1.4,
    "renda": 1.0,
    "patrimonio": 1.0,
    "concentracao": 1.4,
    "necessidade_futura": 1.4,
    "produtos": 1.0,
    "operacoes": 1.0,
    "formacao": 1.0,
}

RESTRICTION_OPTIONS = frozenset(
    {
        "nenhuma",
        "priorizar_renda",
        "evitar_cripto",
        "evitar_exterior",
        "limitar_concentracao",
        "evitar_illiquidez",
    }
)

REQUIRED_QUESTIONS = frozenset((*QUESTION_OPTIONS, "restricoes"))


class ProfileSubmission(ProfileSchema):
    answers: dict[str, AnswerValue]
    investable_capital_brl: Annotated[float, Field(gt=0)]
    consented: bool

    @field_validator("investable_capital_brl")
    @classmethod
    def validate_capital(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("investable capital must be finite")
        return value

    @model_validator(mode="after")
    def validate_submission(self) -> ProfileSubmission:
        answer_keys = set(self.answers)
        missing = sorted(REQUIRED_QUESTIONS - answer_keys)
        unknown = sorted(answer_keys - REQUIRED_QUESTIONS)
        if missing:
            raise ValueError(f"missing profile answers: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"unknown profile answers: {', '.join(unknown)}")
        if not self.consented:
            raise ValueError("profile consent is required")

        for question, options in QUESTION_OPTIONS.items():
            value = self.answers[question]
            if not isinstance(value, str) or value not in options:
                raise ValueError(f"invalid answer for {question}")

        restrictions = self.answers["restricoes"]
        if not isinstance(restrictions, list) or not restrictions:
            raise ValueError("at least one restriction choice is required")
        if any(value not in RESTRICTION_OPTIONS for value in restrictions):
            raise ValueError("invalid restriction choice")
        return self


class ComputedProfile(ProfileSchema):
    version: int = PROFILE_VERSION
    answers: dict[str, AnswerValue]
    investable_capital_brl: float
    consented: bool
    raw_score: float
    score: float
    dimensions: dict[str, float]
    generic_profile: GenericProfile
    rules: list[str]
    warnings: list[str]


class ProfileResponse(ProfileSchema):
    id: str
    account_id: str
    version: int
    answers: dict[str, AnswerValue]
    dimensions: dict[str, float]
    suitability_score: float
    generic_profile: GenericProfile
    investable_capital_brl: float
    consented_at: datetime
    created_at: datetime
