"""Issue #88: o ciclo-mensal avisa no Telegram como terminou (resumo no sucesso, etapa e erro na falha)."""

import contextlib
import io
import json
import os
import unittest
from unittest import mock
import urllib.error

from tests.test_ciclo_mensal import CicloMensalCase, ciclo

import aviso_telegram  # noqa: E402 - scripts/ entra no sys.path ao carregar scripts/ciclo_mensal.py
from teste_para_frente import TesteParaFrenteError  # noqa: E402

TOKEN = "123456:segredo-do-bot"
ENV = {aviso_telegram.ENV_TOKEN: TOKEN, aviso_telegram.ENV_CHAT_ID: "4242"}


class FakeClient:
    """Cliente do Telegram substituído: guarda as mensagens em vez de enviar."""

    def __init__(self, token, chat_id):
        self.token, self.chat_id = token, chat_id
        self.sent = []
        FakeClient.instances.append(self)

    def send(self, text):
        self.sent.append(text)


class TelegramCase(CicloMensalCase):
    def setUp(self):
        super().setUp()
        self.cli_registry = self.root / "cli-registry"
        from tests.test_ciclo_mensal import NOVEMBRO, build_manifest

        build_manifest(self.cli_registry, "mensal-2026-10-31", NOVEMBRO)
        FakeClient.instances = []

    def main(self, *argv, env=ENV, runner=None, medir=None, notificar="telegram", client_factory=FakeClient):
        argv = list(argv) or ["--date", "2026-10-31", "--registry", str(self.cli_registry),
                              "--relatorio-dir", str(self.reports)]
        if notificar == "telegram":
            notificar = aviso_telegram.notificador_telegram(env, client_factory=client_factory)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ciclo.main(argv, session_factory=self.sessions, executor=self.executor(runner),
                              medir=medir or self.forward, notificar=notificar)
        return code, stdout.getvalue(), stderr.getvalue()

    def sent(self):
        return [text for client in FakeClient.instances for text in client.sent]


class MensagensTests(TelegramCase):
    def test_ciclo_concluido_envia_manifesto_mudanca_da_carteira_e_teste_para_frente_por_classe(self):
        october = self.run_october()

        code, _out, err = self.main()

        self.assertEqual(code, 0)
        (client,) = FakeClient.instances
        self.assertEqual((client.token, client.chat_id), (TOKEN, "4242"))
        (text,) = client.sent
        report = self.report()
        lines = text.splitlines()
        self.assertIn("concluído", lines[0])
        # O registro da CLI já tem o manifesto de novembro publicado: a publicação o reaproveita.
        self.assertIn("Manifesto mensal-2026-10-31 (corte 2026-10-30) reaproveitado", text)
        stocks = report["carteira"]["acoes"]["contagem"]
        self.assertIn(f"Ações: {stocks['ficaram']} ficaram, {stocks['entraram']} entraram, {stocks['sairam']} saíram",
                      text)
        self.assertIn(f"Teste para frente do run {october.run_id} (2026-10-29 a 2026-10-30):", text)
        self.assertIn("Ações: +2.00% (IBOV +1.00%)", lines)
        self.assertIn("FIIs: -1.00% (IFIX -0.50%)", lines)
        self.assertIn("Carteira: +1.23% (CDI +0.09%)", lines)
        self.assertIn("aviso enviado no Telegram", err)

    def test_ciclo_sem_carteira_anterior_diz_que_nao_houve_teste_para_frente(self):
        code, _out, _err = self.main()

        self.assertEqual(code, 0)
        (text,) = self.sent()
        self.assertIn("Teste para frente: sem carteira anterior", text)

    def test_ciclo_que_falha_envia_a_etapa_e_o_erro(self):
        self.run_october()

        def unavailable(_run, _manifest):
            raise TesteParaFrenteError("COTAHIST desatualizado")

        code, _out, _err = self.main(medir=unavailable)

        self.assertEqual(code, 6)
        (text,) = self.sent()
        self.assertIn("falhou", text.splitlines()[0])
        self.assertIn("Etapa: teste_para_frente (saída 6)", text)
        self.assertIn("Erro: COTAHIST desatualizado", text)
        self.assertIn("Manifesto mensal-2026-10-31", text)

    def test_publicacao_falhada_avisa_que_o_manifesto_anterior_continua_valendo(self):
        empty = self.root / "registro-que-nao-existe"
        code, _out, _err = self.main("--date", "2026-10-31", "--registry", str(empty),
                                     "--relatorio-dir", str(self.reports))

        self.assertEqual(code, 4)
        (text,) = self.sent()
        self.assertIn("Etapa: publicacao (saída 4)", text)
        self.assertIn("o anterior continua valendo", text)

    def test_mensagem_longa_cabe_no_limite_do_telegram(self):
        result = ciclo.CicloResult(data=ciclo.date(2026, 10, 31), etapa="run", erro="x" * 10000, exit_code=5)

        text = aviso_telegram.formatar_mensagem(result)

        self.assertLessEqual(len(text), aviso_telegram.LIMITE_MENSAGEM)
        self.assertIn("Etapa: run", text)


class SemConfiguracaoTests(TelegramCase):
    def test_sem_token_ou_chat_id_o_ciclo_segue_e_avisa_no_terminal(self):
        for name, env in {"sem nada": {}, "sem chat id": {aviso_telegram.ENV_TOKEN: TOKEN},
                          "sem token": {aviso_telegram.ENV_CHAT_ID: "4242"}}.items():
            with self.subTest(name):
                FakeClient.instances = []
                code, _out, err = self.main(env=env)
                self.assertEqual(code, 0)
                self.assertEqual(FakeClient.instances, [])
                self.assertIn("aviso no Telegram não enviado", err)
                self.assertIn(aviso_telegram.ENV_TOKEN, err)
                self.assertIn(aviso_telegram.ENV_CHAT_ID, err)

    def test_cli_liga_o_telegram_por_padrao_lendo_o_ambiente(self):
        stderr = io.StringIO()
        argv = ["--date", "2026-10-31", "--registry", str(self.cli_registry), "--relatorio-dir", str(self.reports)]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(aviso_telegram.ENV_TOKEN, None)
            os.environ.pop(aviso_telegram.ENV_CHAT_ID, None)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = ciclo.main(argv, session_factory=self.sessions, executor=self.executor(), medir=self.forward)
        self.assertEqual(code, 0)
        self.assertIn("aviso no Telegram não enviado", stderr.getvalue())


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class EnvioTests(TelegramCase):
    def test_cli_padrao_faz_o_post_do_send_message_com_timeout_curto(self):
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            return FakeResponse(b'{"ok": true, "result": {}}')

        argv = ["--date", "2026-10-31", "--registry", str(self.cli_registry), "--relatorio-dir", str(self.reports)]
        with mock.patch.dict(os.environ, ENV, clear=False), mock.patch("urllib.request.urlopen", opener), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = ciclo.main(argv, session_factory=self.sessions, executor=self.executor(), medir=self.forward)

        self.assertEqual(code, 0)
        ((request, timeout),) = calls
        self.assertEqual(request.full_url, f"https://api.telegram.org/bot{TOKEN}/sendMessage")
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data)
        self.assertEqual(payload["chat_id"], "4242")
        self.assertIn("concluído", payload["text"])
        self.assertNotIn("parse_mode", payload)
        self.assertLessEqual(timeout, 15)
        self.assertNotIn(TOKEN, err.getvalue())

    def test_falha_no_envio_nao_muda_o_resultado_e_nunca_mostra_o_token(self):
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        failures = {
            "token inválido": urllib.error.HTTPError(
                url, 401, f"Unauthorized {url}", {}, io.BytesIO(b'{"ok": false, "description": "Unauthorized"}')),
            "rede": urllib.error.URLError(f"sem rota para {url}"),
            "timeout": TimeoutError(f"timed out {url}"),
            "resposta ok false": None,
        }
        for name, failure in failures.items():
            with self.subTest(name):
                def opener(request, timeout, failure=failure):
                    if failure is None:
                        return FakeResponse(f'{{"ok": false, "description": "chat {TOKEN} not found"}}'.encode())
                    raise failure

                client_factory = lambda token, chat_id: aviso_telegram.TelegramClient(  # noqa: E731
                    token, chat_id, opener=opener)
                code, out, err = self.main(client_factory=client_factory)
                self.assertEqual(code, 0)
                self.assertIn("aviso no Telegram não enviado", err)
                self.assertNotIn(TOKEN, err + out)
                self.assertNotIn("segredo-do-bot", err + out)

    def test_falha_no_envio_de_ciclo_que_falhou_mantem_o_codigo_da_etapa(self):
        def opener(_request, timeout):
            raise urllib.error.URLError("rede fora")

        client_factory = lambda token, chat_id: aviso_telegram.TelegramClient(  # noqa: E731
            token, chat_id, opener=opener)
        self.add_account("outra@example.com", plan="premium")
        code, _out, err = self.main(client_factory=client_factory)
        self.assertEqual(code, 3)
        self.assertIn("aviso no Telegram não enviado", err)


if __name__ == "__main__":
    unittest.main()
