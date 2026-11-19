"""Aviso do fim do ciclo mensal no Telegram, com o bot do Prumo (issue #88, spec #82).

``scripts/ciclo_mensal.py`` liga este aviso por padrão. Ele lê duas variáveis de ambiente:

    PRUMO_TELEGRAM_BOT_TOKEN   token do bot, dado pelo BotFather ("123456:ABC...")
    PRUMO_TELEGRAM_CHAT_ID     chat que recebe o aviso (id numérico do chat com o bot)

Sem uma delas, o ciclo segue e o stderr diz que o aviso não foi enviado. Uma falha no envio (rede, timeout, token ou
chat inválido) também só vira uma linha no stderr: nunca muda o resultado nem o código de saída do ciclo. O token
nunca aparece em mensagem de erro ou log (ele faz parte da URL da API, por isso o erro é montado sem ela).

Mensagens, em texto puro (sem parse_mode, para tickers e erros não precisarem de escape) e cortadas no limite de
4096 caracteres do sendMessage:

- sucesso: manifesto publicado (ou reaproveitado), run, o que ficou/entrou/saiu da carteira por classe e o teste
  para frente da carteira anterior em uma linha por classe, tudo tirado do CicloResult e do relatório do ciclo;
- falha: a etapa, o código de saída e o erro.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
import sys
from typing import Any
import urllib.request

ENV_TOKEN = "PRUMO_TELEGRAM_BOT_TOKEN"
ENV_CHAT_ID = "PRUMO_TELEGRAM_CHAT_ID"
API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT_SEGUNDOS = 10
LIMITE_MENSAGEM = 4096

ROTULOS_CLASSE = {"brazilian_stocks": "Ações", "fiis": "FIIs"}
CLASSES_CARTEIRA = (("acoes", "Ações"), ("fiis", "FIIs"))


class TelegramError(Exception):
    """Falha no envio, com mensagem que nunca contém o token."""


class TelegramClient:
    """Envia texto para um chat pelo ``sendMessage`` da API de bots, só com a stdlib."""

    def __init__(self, token: str, chat_id: str, *, opener: Callable[..., Any] | None = None,
                 timeout: float = TIMEOUT_SEGUNDOS):
        self._token = token
        self.chat_id = chat_id
        self._opener = opener
        self.timeout = timeout

    def send(self, text: str) -> None:
        opener = self._opener or urllib.request.urlopen
        request = urllib.request.Request(
            API_URL.format(token=self._token),
            data=json.dumps({"chat_id": self.chat_id, "text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener(request, timeout=self.timeout) as response:
                body = json.loads(response.read() or b"{}")
        except Exception as exc:  # noqa: BLE001 - toda falha vira TelegramError sem o token
            raise TelegramError(self._hide_token(_describe(exc))) from None
        if not isinstance(body, Mapping) or body.get("ok") is not True:
            description = body.get("description") if isinstance(body, Mapping) else None
            raise TelegramError(self._hide_token(f"a API respondeu ok=false: {description or 'sem descrição'}"))

    def _hide_token(self, text: str) -> str:
        return _hide_token(text, self._token)


def _hide_token(text: str, token: str) -> str:
    """``text`` sem o token nem a parte secreta dele (depois do ``:``)."""
    for secret in (token, token.partition(":")[2]):
        if secret:
            text = text.replace(secret, "<token>")
    return text


def _describe(exc: BaseException) -> str:
    """O erro sem a URL (que contém o token): status HTTP e descrição da API, ou só o tipo e o motivo."""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        description = None
        try:
            payload = json.loads(exc.read() or b"{}")
            description = payload.get("description") if isinstance(payload, Mapping) else None
        except Exception:  # noqa: BLE001 - o corpo é só um detalhe da mensagem
            pass
        finally:
            exc.close()
        return f"HTTP {exc.code}" + (f": {description}" if description else "")
    if isinstance(exc, urllib.error.URLError):
        return f"{type(exc).__name__}: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


# --- mensagens -------------------------------------------------------------------------------------------------------


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def _tickers(items) -> str:
    return ", ".join(items) if items else "—"


def _sucesso(result, report: Mapping) -> list[str]:
    lines = [f"Run {result.run_id} {'reaproveitado' if result.run_reaproveitado else 'criado e concluído'}", "",
             "Carteira:"]
    for label, title in CLASSES_CARTEIRA:
        change = report["carteira"][label]
        count = change["contagem"]
        lines.append(f"{title}: {count['ficaram']} ficaram, {count['entraram']} entraram, {count['sairam']} saíram "
                     f"(entraram {_tickers(change['entraram'])}; saíram {_tickers(change['sairam'])})")
    lines.append("")
    forward = report.get("teste_para_frente") or {}
    measured = forward.get("resultado") if forward.get("status") == "medido" else None
    if not measured:
        lines.append("Teste para frente: sem carteira anterior")
        return lines
    lines.append(f"Teste para frente do run {measured.get('run_id')} ({measured.get('inicio')} a {measured.get('fim')}):")
    for key, item in (measured.get("classes") or {}).items():
        title = ROTULOS_CLASSE.get(key, key)
        if item.get("referencia"):
            lines.append(f"{title}: {_pct(item.get('retorno_ponderado'))} "
                         f"({item['referencia']} {_pct(item.get('retorno_referencia'))})")
        else:
            lines.append(f"{title}: {_pct(item.get('retorno_ponderado'))} ({item.get('fonte', '—')})")
    portfolio = measured.get("carteira") or {}
    cdi = ((measured.get("referencias") or {}).get("CDI") or {}).get("retorno")
    lines.append(f"Carteira: {_pct(portfolio.get('retorno_ponderado'))} (CDI {_pct(cdi)})")
    return lines


def formatar_mensagem(result) -> str:
    """O texto do aviso para um ``CicloResult`` de ``scripts/ciclo_mensal.py``; não recalcula nada."""
    day = result.data.isoformat() if result.data else "(sem data válida)"
    lines = [f"Prumo: ciclo mensal {day} {'concluído' if result.ok else 'falhou'}"]
    if result.manifest_id:
        origin = "reaproveitado" if result.manifesto_reaproveitado else "publicado"
        lines.append(f"Manifesto {result.manifest_id} (corte {result.corte}) {origin}")
    if result.ok and result.relatorio:
        lines += _sucesso(result, result.relatorio)
    elif not result.ok:
        lines.append(f"Etapa: {result.etapa} (saída {result.exit_code})")
        if result.etapa == "publicacao":
            lines.append("Nenhum manifesto novo publicado; o anterior continua valendo.")
        lines.append(f"Erro: {result.erro}")
    text = "\n".join(lines)
    return text if len(text) <= LIMITE_MENSAGEM else text[: LIMITE_MENSAGEM - 1] + "…"


# --- notificador ------------------------------------------------------------------------------------------------------


def notificador_telegram(
    env: Mapping[str, str] | None = None,
    *,
    client_factory: Callable[[str, str], Any] = TelegramClient,
) -> Callable[[Any], None]:
    """O ``notificar`` do ``ciclo_mensal.main``: manda o aviso e nunca levanta; o que houve vai para o stderr."""
    env = os.environ if env is None else env
    token, chat_id = (env.get(ENV_TOKEN) or "").strip(), (env.get(ENV_CHAT_ID) or "").strip()

    def notificar(result) -> None:
        if not token or not chat_id:
            print(f"ciclo_mensal: aviso no Telegram não enviado: defina {ENV_TOKEN} e {ENV_CHAT_ID}", file=sys.stderr)
            return
        try:
            client_factory(token, chat_id).send(formatar_mensagem(result))
        except Exception as exc:  # noqa: BLE001 - o aviso nunca muda o resultado do ciclo
            message = _hide_token(f"{type(exc).__name__}: {exc}", token)
            print(f"ciclo_mensal: aviso no Telegram não enviado: {message}", file=sys.stderr)
            return
        print("ciclo_mensal: aviso enviado no Telegram", file=sys.stderr)

    return notificar
