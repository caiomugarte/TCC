● Perfeito! Agora entendi o fluxo completo. Deixe-me criar um diagrama para você:

  📊 Fluxo de Processamento de Dados

  1. Arquivos brutos por setor (PRÉ-CARREGADOS):
     ├── data/bens_industriais_raw.csv
     ├── data/consumo_ciclico_raw.csv
     ├── data/consumo_nao_ciclico_raw.csv
     ├── data/utilidade_publica_raw.csv
     ├── data/materiais_basicos_raw.csv
     ├── data/financeiros_outros_raw.csv
     ├── data/tecnologia_informacao_raw.csv
     ├── data/sauda_raw.csv
     ├── data/petroleo_gas_biocombustivel_raw.csv
     └── data/comunicacoes_raw.csv

  2. cleaner.py (GERA O ARQUIVO CONSOLIDADO):
     ↓
     status_invest_fundamentals.csv (COM coluna SETOR)

  3. data_preprocessing.py:
     ↓
     data/processed/fundamentals_clean_{perfil}.csv

  Resposta à sua pergunta

  Os dados dos setores são PRÉ-CARREGADOS nos arquivos data/*_raw.csv. O script cleaner.py é responsável por:

  1. Ler os 10 arquivos CSV separados por setor
  2. Adicionar a coluna SETOR a cada arquivo (linha 35 do cleaner.py)
  3. Consolidar todos em um único arquivo status_invest_fundamentals.csv
  4. Limpar dados (remover outliers, converter formatos)

  Então você precisa executar python cleaner.py antes de rodar o preprocessing se os dados brutos mudarem. O arquivo gerado (status_invest_fundamentals.csv) já contém a coluna SETOR que é usada em todo o pipeline.

  Quer que eu verifique se há alguma inconsistência entre os arquivos atuais?
