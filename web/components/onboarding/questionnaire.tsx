"use client";

import { useAuth } from "@clerk/nextjs";
import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiClientError, getProfile, saveProfile } from "@/lib/api-client";
import type { ProfileInput } from "@/lib/api-types";

type Option = {
  value: string;
  label: string;
  help: string;
};

type Question = {
  key: string;
  legend: string;
  help: string;
  options: Option[];
  multiple?: boolean;
};

type Answers = ProfileInput["answers"];
type FieldErrors = Record<string, string>;

const QUESTIONS: Question[] = [
  {
    key: "objetivo",
    legend: "Qual é o principal objetivo desta parcela do seu patrimônio?",
    help: "A finalidade declarada é uma das âncoras do perfil.",
    options: [
      ["preservacao", "Preservar o capital", "Priorizar estabilidade e reduzir oscilações."],
      ["renda", "Gerar renda", "Receber dividendos e manter previsibilidade."],
      ["equilibrio", "Equilibrar renda e crescimento", "Aceitar oscilações moderadas para buscar valorização."],
      ["crescimento", "Crescer o patrimônio", "Priorizar retorno de longo prazo."],
      ["crescimento_agressivo", "Buscar crescimento agressivo", "Aceitar perdas relevantes em troca de maior potencial."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "horizonte",
    legend: "Por quanto tempo você pretende manter o investimento?",
    help: "Considere quando o dinheiro poderá ser necessário, não apenas sua intenção atual.",
    options: [
      ["ate_2_anos", "Até 2 anos", "Posso precisar resgatar em breve."],
      ["2_a_5_anos", "De 2 a 5 anos", "Tenho alguma flexibilidade."],
      ["5_a_10_anos", "De 5 a 10 anos", "Posso atravessar ciclos de mercado."],
      ["mais_de_10_anos", "Mais de 10 anos", "Não há necessidade planejada de resgate."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "capacidade",
    legend: "Quanto desta parcela poderia permanecer investido sem comprometer despesas e reserva de emergência?",
    help: "Esta é uma triagem simples; a análise regulatória deve coletar renda, patrimônio e necessidades futuras separadamente.",
    options: [
      ["nenhuma", "Nenhuma ou quase nenhuma", "O dinheiro pode ser necessário para despesas essenciais."],
      ["menos_de_10", "Menos de 10%", "Tenho pouca margem para perdas."],
      ["10_a_30", "De 10% a 30%", "Tenho alguma reserva separada."],
      ["30_a_60", "De 30% a 60%", "Consigo manter o investimento em uma queda."],
      ["mais_de_60", "Mais de 60%", "Esta parcela não é necessária para o orçamento."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "reacao",
    legend: "Como você reagiria se a carteira caísse 20% em poucos meses?",
    help: "A pergunta mede comportamento diante de uma perda, não uma previsão de retorno.",
    options: [
      ["vender_tudo", "Venderia tudo", "Eu não conseguiria manter a posição."],
      ["vender_parte", "Venderia parte", "Reduziria o risco até me sentir confortável."],
      ["manter", "Manteria a carteira", "Aguardaria uma recuperação."],
      ["comprar", "Manteria e compraria um pouco", "Usaria a queda para ajustar posições."],
      ["comprar_mais", "Compraria mais agressivamente", "Tenho convicção e caixa para aproveitar a queda."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "perda",
    legend: "Qual perda temporária máxima você aceitaria sem mudar seu plano?",
    help: "Escolha a faixa que ainda permitiria manter o horizonte informado acima.",
    options: [
      ["ate_5", "Até 5%", "Oscilações maiores já seriam desconfortáveis."],
      ["5_a_10", "De 5% a 10%", "Aceito uma variação limitada."],
      ["10_a_20", "De 10% a 20%", "Aceito oscilações de um ciclo normal."],
      ["20_a_35", "De 20% a 35%", "Consigo suportar uma queda forte."],
      ["mais_de_35", "Mais de 35%", "Aceito alta volatilidade e perdas expressivas."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "experiencia",
    legend: "Qual é sua experiência com ações e indicadores fundamentalistas?",
    help: "Conhecimento não transforma um investimento arriscado em investimento seguro.",
    options: [
      ["nenhuma", "Nenhuma", "Nunca investi em ações por conta própria."],
      ["basica", "Básica", "Conheço os conceitos, mas tenho pouca prática."],
      ["intermediaria", "Intermediária", "Já acompanho resultados, valuation ou indicadores."],
      ["avancada", "Avançada", "Analiso empresas e acompanho minhas posições."],
      ["profissional", "Profissional", "Tenho formação ou atuação diretamente relacionada."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "liquidez",
    legend: "Quando você poderá precisar acessar esta parcela?",
    help: "Necessidade de liquidez reduz a capacidade de atravessar uma queda.",
    options: [
      ["a_qualquer_momento", "A qualquer momento", "Não posso aceitar bloqueio ou espera."],
      ["ate_1_ano", "Em até 1 ano", "Tenho uma necessidade próxima."],
      ["1_a_3_anos", "Entre 1 e 3 anos", "Consigo planejar o resgate."],
      ["mais_de_3_anos", "Depois de 3 anos", "Tenho flexibilidade para esperar."],
      ["sem_previsao", "Não há previsão", "Esta parcela pode permanecer investida por tempo indeterminado."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "renda",
    legend: "Qual é a faixa da sua renda mensal regular?",
    help: "Use renda recorrente, antes de considerar ganhos eventuais.",
    options: [
      ["ate_3k", "Até R$ 3 mil", "Renda regular mais limitada."],
      ["3_a_8k", "De R$ 3 mil a R$ 8 mil", "Renda regular intermediária."],
      ["8_a_20k", "De R$ 8 mil a R$ 20 mil", "Renda regular com maior margem."],
      ["20_a_50k", "De R$ 20 mil a R$ 50 mil", "Renda regular elevada."],
      ["mais_de_50k", "Mais de R$ 50 mil", "Renda regular muito elevada."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "patrimonio",
    legend: "Qual é a faixa aproximada do seu patrimônio financeiro investível?",
    help: "Considere ativos financeiros disponíveis para investimento, não o imóvel de moradia.",
    options: [
      ["ate_50k", "Até R$ 50 mil", "Patrimônio financeiro inicial."],
      ["50_a_200k", "De R$ 50 mil a R$ 200 mil", "Patrimônio financeiro intermediário."],
      ["200k_a_1m", "De R$ 200 mil a R$ 1 milhão", "Patrimônio financeiro relevante."],
      ["1m_a_5m", "De R$ 1 milhão a R$ 5 milhões", "Patrimônio financeiro elevado."],
      ["mais_de_5m", "Mais de R$ 5 milhões", "Patrimônio financeiro muito elevado."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "concentracao",
    legend: "Que parte do seu patrimônio financeiro representa esta parcela?",
    help: "Quanto maior a concentração nesta carteira, menor a margem para suportar perdas.",
    options: [
      ["ate_10", "Até 10%", "Esta parcela tem impacto limitado no patrimônio."],
      ["10_a_30", "De 10% a 30%", "Tenho diversificação fora desta parcela."],
      ["30_a_60", "De 30% a 60%", "Esta parcela tem peso relevante."],
      ["60_a_80", "De 60% a 80%", "Grande parte do patrimônio está concentrada aqui."],
      ["mais_de_80", "Mais de 80%", "Quase todo o patrimônio depende desta parcela."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "necessidade_futura",
    legend: "Quanto do seu patrimônio financeiro poderá ser necessário para objetivos planejados nos próximos 3 anos?",
    help: "Considere compras, estudos, aposentadoria, dependentes, impostos ou qualquer saque planejado.",
    options: [
      ["nenhuma", "Nenhuma ou quase nenhuma", "Não há saque planejado relevante."],
      ["ate_10", "Até 10%", "Necessidade futura pequena."],
      ["10_a_30", "De 10% a 30%", "Necessidade futura moderada."],
      ["30_a_60", "De 30% a 60%", "Necessidade futura relevante."],
      ["mais_de_60", "Mais de 60%", "Grande parte do patrimônio terá uso planejado."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "produtos",
    legend: "Com quais produtos você tem familiaridade suficiente para explicar riscos e custos?",
    help: "Conhecimento declarado não elimina risco, mas ajuda a verificar se o produto é compreensível.",
    options: [
      ["nenhum", "Nenhum produto além de poupança ou conta remunerada", "Não conheço produtos de mercado de capitais."],
      ["renda_fixa_fundos", "Renda fixa e fundos", "Conheço produtos mais simples."],
      ["etf_fii_acoes", "ETFs, FIIs ou ações", "Conheço produtos negociados em bolsa."],
      ["cripto", "Ações, ETFs, FIIs e criptoativos", "Conheço diferentes classes e seus riscos."],
      ["complexos", "Também derivativos ou produtos estruturados", "Conheço produtos complexos e riscos específicos."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "operacoes",
    legend: "Como foi sua experiência real de operações no mercado?",
    help: "Considere tipo, volume, frequência e período das operações.",
    options: [
      ["nenhuma", "Nunca operei", "Não tenho histórico próprio de operações."],
      ["inicial", "Experiência inicial", "Menos de 1 ano ou poucas operações."],
      ["ocasional", "Experiência ocasional", "De 1 a 3 anos, com operações esporádicas."],
      ["regular", "Experiência regular", "Mais de 3 anos acompanhando e operando posições."],
      ["frequente", "Experiência frequente ou profissional", "Opero com alta frequência ou atuação profissional."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "formacao",
    legend: "Qual formação acadêmica ou experiência profissional financeira você possui?",
    help: "Esta informação complementa, mas não substitui, histórico de operações e familiaridade com produtos.",
    options: [
      ["nenhuma", "Nenhuma relacionada", "Não tive formação ou experiência financeira."],
      ["autodidata", "Estudo autodidata", "Aprendi por cursos, livros ou prática pessoal."],
      ["academica", "Formação acadêmica relacionada", "Cursei disciplinas ou formação na área."],
      ["profissional_indireta", "Experiência profissional próxima", "Atuo em área com contato recorrente com finanças."],
      ["profissional_direta", "Atuação profissional direta", "Trabalho ou tenho certificação diretamente relacionada."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
  {
    key: "restricoes",
    legend: "Quais restrições ou preferências devem ser respeitadas?",
    help: "Selecione pelo menos uma. Estas respostas serão aplicadas explicitamente depois.",
    multiple: true,
    options: [
      ["nenhuma", "Nenhuma restrição específica", "Posso avaliar qualquer alternativa compatível com meu perfil."],
      ["priorizar_renda", "Priorizar geração de renda", "Prefiro receber fluxo recorrente."],
      ["evitar_cripto", "Evitar criptoativos", "Não quero exposição a criptoativos."],
      ["evitar_exterior", "Evitar exposição internacional", "Prefiro manter exposição econômica doméstica."],
      ["limitar_concentracao", "Limitar concentração", "Quero limite adicional por ativo, setor ou classe."],
      ["evitar_illiquidez", "Evitar baixa liquidez", "Preciso conseguir sair sem espera relevante."],
    ].map(([value, label, help]) => ({ value, label, help })),
  },
];

function selectedValue(answers: Answers, key: string): string | string[] | undefined {
  return answers[key];
}

export function OnboardingQuestionnaire() {
  const { getToken } = useAuth();
  const router = useRouter();
  const [answers, setAnswers] = useState<Answers>({ restricoes: [] });
  const [capital, setCapital] = useState("");
  const [consented, setConsented] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [isLoadingProfile, setIsLoadingProfile] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadProfile() {
      try {
        const profile = await getProfile(await getToken());
        if (!active || !profile) return;
        setAnswers(profile.answers);
        setCapital(String(profile.investableCapitalBrl));
        setConsented(true);
      } catch (error) {
        if (active) {
          setFormError(
            error instanceof ApiClientError
              ? error.message
              : "Não foi possível carregar seu perfil. Você pode preenchê-lo novamente.",
          );
        }
      } finally {
        if (active) setIsLoadingProfile(false);
      }
    }

    void loadProfile();
    return () => { active = false; };
  }, [getToken]);

  function updateSingleAnswer(key: string, value: string) {
    setAnswers((current) => ({ ...current, [key]: value }));
    setFieldErrors((current) => ({ ...current, [key]: "" }));
  }

  function updateRestriction(value: string, checked: boolean) {
    setAnswers((current) => {
      const currentValues = Array.isArray(current.restricoes) ? current.restricoes : [];
      const nextValues = checked
        ? [...new Set([...currentValues, value])]
        : currentValues.filter((item) => item !== value);
      return { ...current, restricoes: nextValues };
    });
    setFieldErrors((current) => ({ ...current, restricoes: "" }));
  }

  function validate(): FieldErrors {
    const errors: FieldErrors = {};
    for (const question of QUESTIONS) {
      const answer = selectedValue(answers, question.key);
      const missing = question.multiple
        ? !Array.isArray(answer) || answer.length === 0
        : typeof answer !== "string" || answer.length === 0;
      if (missing) errors[question.key] = "Selecione uma opção.";
    }
    if (!capital || Number(capital) <= 0 || !Number.isFinite(Number(capital))) {
      errors.investableCapitalBrl = "Informe um capital positivo.";
    }
    if (!consented) errors.consented = "Confirme o uso acadêmico do resultado.";
    return errors;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const errors = validate();
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setFormError("Revise os campos destacados antes de continuar.");
      return;
    }

    setIsSubmitting(true);
    setFormError(null);
    try {
      await saveProfile({
        answers,
        investableCapitalBrl: Number(capital),
        consented,
      }, await getToken());
      router.push("/app/recommendation");
    } catch (error) {
      setFormError(
        error instanceof ApiClientError
          ? error.message
          : "Não foi possível salvar seu perfil. Tente novamente.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  if (isLoadingProfile) {
    return <section className="card" aria-live="polite"><p>Carregando seu perfil…</p></section>;
  }

  return (
    <form className="questionnaire" onSubmit={handleSubmit} noValidate>
      <section className="card">
        <p className="eyebrow">Onboarding</p>
        <h1>Mapeie seu perfil</h1>
        <p>
          Responda com base nesta parcela do patrimônio. O resultado é uma
          parametrização inicial, não uma classificação regulatória nem uma
          promessa de retorno.
        </p>
      </section>

      {QUESTIONS.map((question, questionIndex) => {
        const answer = selectedValue(answers, question.key);
        const errorId = `${question.key}-error`;
        return (
          <section className="card question" key={question.key}>
            <fieldset aria-describedby={`${question.key}-help ${errorId}`}>
              <legend>{questionIndex + 1}. {question.legend}</legend>
              <p id={`${question.key}-help`} className="field-hint">{question.help}</p>
              <div className="options">
                {question.options.map((option, optionIndex) => {
                  const inputId = `${question.key}-${option.value}`;
                  const checked = question.multiple
                    ? Array.isArray(answer) && answer.includes(option.value)
                    : answer === option.value;
                  return (
                    <label className="option" htmlFor={inputId} key={option.value}>
                      <input
                        id={inputId}
                        name={question.key}
                        type={question.multiple ? "checkbox" : "radio"}
                        value={option.value}
                        checked={checked}
                        onChange={(event) => question.multiple
                          ? updateRestriction(option.value, event.target.checked)
                          : updateSingleAnswer(question.key, option.value)}
                        required={!question.multiple && optionIndex === 0}
                      />
                      <span><strong>{option.label}</strong><small>{option.help}</small></span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
            {fieldErrors[question.key] ? <p id={errorId} className="field-error">{fieldErrors[question.key]}</p> : null}
          </section>
        );
      })}

      <section className="card">
        <label htmlFor="investableCapitalBrl">
          Capital investível desta parcela (BRL)
          <input
            id="investableCapitalBrl"
            name="investableCapitalBrl"
            type="number"
            min="0.01"
            step="0.01"
            inputMode="decimal"
            value={capital}
            onChange={(event) => setCapital(event.target.value)}
            aria-invalid={Boolean(fieldErrors.investableCapitalBrl)}
            aria-describedby="capital-help capital-error"
          />
        </label>
        <p id="capital-help" className="field-hint">Use apenas o valor que será considerado nesta carteira, não o patrimônio total.</p>
        {fieldErrors.investableCapitalBrl ? <p id="capital-error" className="field-error">{fieldErrors.investableCapitalBrl}</p> : null}
      </section>

      <label className="consent option" htmlFor="consent">
        <input
          id="consent"
          name="consent"
          type="checkbox"
          checked={consented}
          onChange={(event) => {
            setConsented(event.target.checked);
            setFieldErrors((current) => ({ ...current, consented: "" }));
          }}
        />
        <span>Entendo que este resultado é uma parametrização acadêmica e não substitui suitability regulatório ou recomendação de investimento.</span>
      </label>
      {fieldErrors.consented ? <p className="field-error">{fieldErrors.consented}</p> : null}

      {formError ? <p className="form-error" role="alert">{formError}</p> : null}
      <div className="actions">
        <button className="button" type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Salvando…" : "Salvar perfil"}
        </button>
      </div>
    </form>
  );
}
