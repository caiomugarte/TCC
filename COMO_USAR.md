# 🎯 Como Usar o Código Modularizado

## Opção 1: Menu Interativo (Mais Fácil)

```bash
cd py
python main.py
```

Você verá:
```
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║      Sistema de Otimização de Carteiras - Algoritmo Genético         ║
║                           TCC - 2025                                  ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

Escolha uma opção:

  [1] 🔧 Pré-processar dados (todos os perfis)
  [2] 🚀 Execução única do GA (todos os perfis)
  [3] 📊 Múltiplas execuções (análise de robustez)
  [4] 📈 Backtest de carteiras (em desenvolvimento)
  [5] 🗑️  Limpar cache
  [6] ⚙️  Configurações
  [0] 🚪 Sair
```

## Opção 2: Linha de Comando (Mais Rápido)

```bash
# PRIMEIRA EXECUÇÃO (faz tudo)
cd py
python main.py --all

# PRÓXIMAS EXECUÇÕES (usa cache, super rápido!)
python main.py --single          # Apenas GA
python main.py --multi           # Análise de robustez
python main.py --multi --n-runs 50  # 50 rodadas
```

## 📊 Resultados

Tudo fica em `outputs/`:
```
outputs/
├── carteira_conservador_ga.json          # Carteira única
├── carteira_conservador_consensus.json   # Carteira consenso
├── metrics_stability_conservador.csv     # Métricas detalhadas
├── summary_ga.json                       # Summary consolidado
└── multiple_runs_summary.json            # Análise de robustez
```

## ⚙️  Personalizar Perfis

Edite `py/config.py`:
```python
GA_CONFIG["conservador"]["generations"] = 500  # Aumenta gerações
FILTERS["conservador"]["cap_min"] = 10_000_000_000  # Empresas maiores
```

Depois:
```bash
python main.py --clear-cache  # Limpa cache antigo
python main.py --all          # Roda com novas configs
```

## 📚 Documentação Completa

- `py/README_MODULAR.md` - Documentação técnica completa
- `py/GUIA_RAPIDO.md` - Guia rápido de uso

---

**Dúvidas?** Leia os arquivos de documentação ou teste com `python main.py`
