# Sistema Modularizado de Otimização de Carteiras

Sistema refatorado para otimização de carteiras de investimento usando Algoritmo Genético, com foco em performance, manutenibilidade e facilidade de uso.

## 🎯 Principais Melhorias

### ✅ Resolvido
- **Pipeline único**: Um único comando executa tudo
- **Cache inteligente**: Evita reprocessamento desnecessário
- **Paralelização**: Execução simultânea de múltiplos perfis
- **CLI interativa**: Menu amigável para escolher operações
- **Código modular**: Organização clara em pacotes
- **Configuração centralizada**: Todas as configs em um único arquivo

### 🚀 Performance

- **Cache automático**: Dados pré-processados são salvos e reutilizados
- **Paralelização**: Múltiplas execuções do GA em paralelo (usa todos os cores)
- **Execução seletiva**: Rode apenas as etapas necessárias

## 📁 Nova Estrutura

```
py/
├── main.py                    # 🎯 PONTO DE ENTRADA PRINCIPAL
├── config.py                  # ⚙️  Todas as configurações
├── requirements.txt           # 📦 Dependências
│
├── core/                      # 🧠 Módulos principais
│   ├── __init__.py
│   ├── preprocessing.py       # Limpeza e padronização de dados
│   ├── scoring.py             # Cálculo de scores fundamentalistas
│   ├── optimizer.py           # Algoritmo Genético
│   └── metrics.py             # HHI, Jaccard, métricas compartilhadas
│
├── pipelines/                 # 🔄 Pipelines de execução
│   ├── __init__.py
│   ├── single_run.py          # Execução única do GA
│   ├── multi_run.py           # Múltiplas execuções (robustez)
│   └── backtest.py            # Backtest (em desenvolvimento)
│
├── utils/                     # 🛠️ Utilitários
│   ├── __init__.py
│   └── cache.py               # Sistema de cache inteligente
│
└── [arquivos antigos]         # 📜 Mantidos para referência
    ├── ga.py
    ├── profiles.py
    ├── data_preprocessing.py
    └── ...
```

## 🚀 Uso Rápido

### Modo 1: CLI Interativa (Recomendado)

```bash
cd py
python main.py
```

Você verá um menu interativo:
```
[1] 🔧 Pré-processar dados
[2] 🚀 Execução única do GA
[3] 📊 Múltiplas execuções
[4] 📈 Backtest de carteiras
[5] 🗑️  Limpar cache
[6] ⚙️  Configurações
[0] 🚪 Sair
```

### Modo 2: Linha de Comando

```bash
# Executa tudo (recomendado para primeira execução)
python main.py --all

# Apenas pré-processamento
python main.py --preprocess

# Apenas execução única
python main.py --single

# Apenas múltiplas execuções (30 rodadas, padrão)
python main.py --multi

# Múltiplas execuções com 50 rodadas
python main.py --multi --n-runs 50

# Execução sem cache (reprocessa tudo)
python main.py --single --no-cache

# Limpa cache
python main.py --clear-cache
```

## 📊 Fluxo de Execução

### Pipeline Completo

```
1. PRÉ-PROCESSAMENTO
   ├─ Carrega dados brutos
   ├─ Aplica filtros de elegibilidade
   ├─ Winsoriza outliers
   ├─ Normaliza (z-score)
   └─ Salva em cache ✅

2. EXECUÇÃO ÚNICA
   ├─ Carrega dados do cache ⚡
   ├─ Calcula scores por perfil
   ├─ Executa GA
   └─ Gera relatórios

3. MÚLTIPLAS EXECUÇÕES
   ├─ Executa GA N vezes em paralelo 🚀
   ├─ Analisa estabilidade
   ├─ Calcula Jaccard médio
   └─ Gera carteira consenso
```

## 🎛️ Configurações

Todas as configurações estão centralizadas em `config.py`:

```python
# Exemplo: Alterar parâmetros do GA para perfil conservador
GA_CONFIG = {
    "conservador": {
        "n_assets": 10,        # Número de ativos
        "lambda": 0.50,        # Penalização HHI
        "generations": 300,    # Gerações do GA
        "pop_size": 200        # Tamanho da população
    },
    ...
}
```

## 🔧 Exemplos de Uso Programático

### Executar para um único perfil

```python
from pipelines.single_run import run_single_portfolio

portfolio = run_single_portfolio(
    profile="conservador",
    use_cache=True,
    robustness_filter=True,
    random_seed=42
)

print(portfolio[["TICKER", "SCORE"]])
```

### Múltiplas execuções customizadas

```python
from pipelines.multi_run import run_multi_execution_profile

results = run_multi_execution_profile(
    profile="moderado",
    n_runs=50,
    parallel=True
)

print(f"Fitness médio: {results['stability_metrics']['fitness']['mean']:.2f}")
print(f"Jaccard médio: {results['stability_metrics']['portfolio_similarity']['jaccard_mean']:.3f}")
```

## 📈 Saídas Geradas

### Diretório `outputs/`

```
outputs/
├── carteira_conservador_ga.json       # Carteira única
├── carteira_conservador_consensus.json # Carteira consenso
├── metrics_stability_conservador.csv  # Métricas de cada run
├── summary_ga.json                    # Summary consolidado
└── multiple_runs_summary.json         # Análise de robustez
```

### Diretório `.cache/`

```
.cache/
├── preprocessing_conservador.csv      # Dados pré-processados
├── preprocessing_moderado.csv
├── preprocessing_arrojado.csv
└── metadata.json                      # Metadados do cache
```

## 🧹 Gerenciamento de Cache

### Via CLI
```bash
python main.py --clear-cache
```

### Via Menu Interativo
```
[5] 🗑️ Limpar cache
```

### Programaticamente
```python
from utils.cache import CacheManager

cache = CacheManager()
cache.clear()  # Limpa tudo

# Ou limpar apenas um item
cache.clear("preprocessing_conservador")
```

## ⚡ Dicas de Performance

1. **Primeira execução**: Use `--all` para processar e cachear tudo
2. **Iterações rápidas**: Com cache, execuções subsequentes são 10x+ mais rápidas
3. **Múltiplos perfis**: A paralelização automática usa todos os cores
4. **Múltiplas rodadas**: Use `--parallel` para análise de robustez

## 🔄 Migrando do Código Antigo

### Antes (código antigo)
```bash
# Tinha que executar manualmente em ordem:
python data_preprocessing.py
python profiles.py conservador
python ga.py conservador
python ga_multiple_runs.py
python build_portfolios_summary.py
```

### Agora (código novo)
```bash
# Um único comando:
python main.py --all
```

## 🆘 Troubleshooting

### Cache desatualizado
```bash
python main.py --clear-cache
python main.py --all
```

### Erro de import
```bash
# Certifique-se de estar no diretório py/
cd py
python main.py
```

### Performance lenta
```bash
# Use cache e paralelização
python main.py --multi --parallel
```

## 📚 Módulos Principais

### `core.preprocessing`
- `load_raw_data()`: Carrega CSV bruto
- `preprocess_profile()`: Pipeline de limpeza
- `load_processed_data()`: Carrega dados do cache

### `core.scoring`
- `build_scores()`: Calcula scores ponderados
- `get_top_stocks()`: Retorna top N ativos

### `core.optimizer`
- `optimize_portfolio()`: Executa GA
- `GeneticAlgorithm`: Classe principal do otimizador

### `core.metrics`
- `hhi_sector()`: Calcula HHI
- `jaccard_similarity()`: Similaridade entre carteiras
- `coefficient_of_variation()`: Análise de estabilidade

## 🎓 Benefícios para o TCC

1. **Reprodutibilidade**: Seeds fixos + cache = resultados consistentes
2. **Análise de robustez**: 30+ execuções em minutos com paralelização
3. **Documentação**: Código bem estruturado e documentado
4. **Manutenibilidade**: Fácil adicionar novos perfis ou métricas
5. **Performance**: Cache evita reprocessamento desnecessário

## 📝 Próximos Passos

- [ ] Implementar pipeline de backtest completo
- [ ] Adicionar testes unitários
- [ ] Adicionar visualizações automáticas
- [ ] Integração com outros frameworks

## 🤝 Contribuindo

Para adicionar novos perfis, edite `config.py`:

```python
GA_CONFIG["novo_perfil"] = {
    "n_assets": 12,
    "lambda": 0.30,
    "generations": 350,
    "pop_size": 225
}

FILTERS["novo_perfil"] = {
    "cap_min": 2_000_000_000,
    "liq_min": 1_000_000
}

PROFILE_WEIGHTS["novo_perfil"] = {
    "liquidez": 0.25,
    "rent": 0.30,
    "value": 0.20,
    "growth": 0.15,
    "div": 0.10
}
```

Depois execute:
```bash
python main.py --all
```

---

**Desenvolvido para TCC 2025**
