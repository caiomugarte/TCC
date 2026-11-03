# Artefatos da pesquisa de alocação entre classes (2026-11)

> **Validação point-in-time:** os números históricos deste documento não têm fonte point-in-time auditada. Veredito da alocação entre classes: VALIDATED; sobreposição com o período da escolha do método: full (ver [docs/validation/class-allocation-oos.md](../../validation/class-allocation-oos.md)).

Scripts e saídas brutas por trás de
[`../class-allocation-methods-2026-11.md`](../class-allocation-methods-2026-11.md).

| Arquivo | O que é |
|---|---|
| `anchors.py` | Alocação por score × janela de treino × conjunto de âncoras, usando a busca em grade de produção (`optimize_window`) e a política (`_allocation_policy`). Hoje lê o IBOV para ações brasileiras; a primeira execução da rodada 0 usou a cesta `caio_stocks`. |
| `compare.py` | Rodada 1: janela única de 3 anos (hoje) × reamostragem por janelas × paridade de risco, mais o walk-forward do score 0,58. |
| `compare2.py` | Rodada 2: acrescenta 1/N + renda fixa, bootstrap estacionário por blocos com a busca em grade e o `MeanRisk` do skfolio (pior queda, CDaR, CDaR + bootstrap). Importa `compare.py`. |
| `round0-anchors-caio-stocks-ifix.txt` | Rodada 0, `caio_stocks` + IFIX, âncoras atuais e amplas. |
| `round0-anchors-ibov-ifix.txt` | Rodada 0, IBOV + IFIX, âncoras atuais. |
| `round1-resampling-vs-risk-parity.txt` | Saída da rodada 1. |
| `round2-all-methods.txt` | Saída da rodada 2. |

## Como reproduzir

Os scripts leem o snapshot `snapshots/data/snap-2026-10-29/` (corte em
2026-10-28), gerado com `py/fetch_allocation_snapshot.py` e publicado como o
manifest `dev-2026-10-29`. A pasta `snapshots/` não é versionada, então rodar
de novo com outro snapshot dá números diferentes.

```bash
python3 -m venv .venv-research
.venv-research/bin/pip install -r api/requirements.txt -r py/requirements.txt skfolio arch
cd docs/research/class-allocation-2026-11
../../../.venv-research/bin/python -I compare.py    # rodada 1, ~10 min em 8 núcleos
../../../.venv-research/bin/python -I compare2.py   # rodada 2, ~35 min em 10 núcleos
```

O `compare.py` grava `grid_cache.pkl` ao lado dos scripts para reaproveitar
as buscas em execuções seguintes; apague o arquivo ao trocar de snapshot.
Versões usadas: skfolio 1.12.5, arch 8.0.0, Python 3.14.
