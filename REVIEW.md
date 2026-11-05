# Instruções de revisão

## O que é Important aqui
Este repositório tem o otimizador de carteira (GA de ações e FIIs, alocação por classe), a validação científica congelada e a API/web Prumo que entrega recomendações a contas Basic e Premium. Aqui, uma falha é:
- Um número errado sem erro visível: ação sem score entrando no GA, NaN virando 0 sem decisão registrada, peso de classe errado, recomendação lida de um cache gerado pela regra antiga.
- Um documento que afirma mais do que a prova commitada mostra. Vale para `docs/validation/`, `baselines/*/README.md`, ADRs e specs: denominador, contagem de linhas, datas, "out-of-sample". Nesses arquivos o texto é o resultado, então texto falso é Important.
- Reprodutibilidade quebrada: bytes de dado hasheado mudando, uma ferramenta gravando dentro de uma árvore congelada, um relatório que aceita o veredito de uma etapa anterior sem recalcular.
- Uma conta vendo dado de outra, recurso Premium sem `require_premium`, `result` ou provenance de uma run que não está `completed` aparecendo na resposta.
- Um CLI de pesquisa saindo com código fora da tabela documentada no README ou na docstring da ferramenta.

Conta no máximo como Nit: helper duplicado, parâmetro sem uso, nome, arquivo grande em ferramenta de pesquisa, wrap de parágrafo.

## Limite de Nits
Reporte no máximo 5 Nits por revisão.

## Não reportar
- Estilo e formatação: o repo não tem formatter, lint nem CI.
- Arquivos gerados ou de ambiente: `**/__pycache__/**`, `*.pyc`, `.idea/**`, `py/.idea/**`, `*.db`, `identifier.sqlite`, `web/package-lock.json`, `.cache/**`, `outputs*/**`.
- Skills de terceiros em `.agents/skills/**`.
- O conteúdo linha a linha de CSV, parquet e JSON de prova (`data/**`, `baselines/*/reference/**`, `**/proofs/**`). Reporte apenas se a mudança no arquivo for indevida (veja "Sempre verificar").

## Sempre verificar
- Se uma mudança em `py/core/` ou `py/pipelines/` altera o que vai para os CSVs: os `data/processed/fundamentals_clean_*.csv` versionados, lidos com `use_cache=True`, foram regenerados ou ainda carregam a regra antiga?
- Se o método de alocação ou de seleção mudou: `model_versions` (`api/app/services/premium_policy.py`) foi atualizado, já que esse valor fica persistido em cada run?
- `data/allocation*/**` e todo arquivo cujo hash é registrado são commitados byte a byte (`.gitattributes` com `-text`). Confira com `git ls-files --eol`.
- Ferramentas de replay, validação e snapshot registry não gravam em `baselines/*/{code,data,reference}` nem no registry, e recalculam hash e proveniência em vez de confiar no `status` de um artefato anterior.
- Todo código de saída novo de CLI está na tabela documentada e tem um teste que passa pelo caminho real (não só por `--compare-only`).
- Todo número ou afirmação em README, `docs/validation/`, ADR ou `baselines/*/README.md` bate com o artefato commitado que cita.
- Um teste novo falha quando o comportamento que ele nomeia é removido. Se uma mutação simples passa, aponte qual.
- Strings ou enums copiados entre `api/` e `web/lib/api-types.ts` têm um teste que os compara ou vêm de uma fonte única.
- Rotas em `api/app/routers/` buscam dados escopados por conta (`get_owned_run` ou filtro por `account_id`) e protegem recurso Premium com `require_premium`.

## Evidência
Afirmações sobre comportamento citam `arquivo:linha` do código; não deduza comportamento pelo nome. Afirmações sobre resultado citam o artefato commitado (CSV, JSON de prova) e o número exato.

## Caminhos sensíveis
- `api/app/auth/**`: identifica a conta; um erro aqui expõe dados de outra conta.
- `api/app/entitlements/**`: separa Basic de Premium (o que o cliente paga).
- `api/app/routers/**`: escopo por conta e gate Premium em cada rota.
- `api/migrations/**`: schema do banco; uma migration errada não tem rollback simples.
- `web/middleware.ts`: guarda das rotas logadas.
- `py/core/**`: score, preprocessing e otimizador que geram a recomendação entregue ao usuário.
- `api/app/adapters/**`: ligação entre a API e o otimizador; decide o que vira recomendação.
- `py/snapshot_registry.py`: escolhe o snapshot de dados que alimenta a recomendação.
- `py/snapshot_manifest.py`: escolhe o snapshot de dados que alimenta a recomendação.
- `data/allocation*/**`: a auditoria point-in-time hasheia esses arquivos byte a byte.
- `baselines/**`: baseline científico congelado; qualquer byte alterado invalida o replay.
