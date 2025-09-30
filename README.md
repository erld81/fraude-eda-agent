# Fraude EDA Agent

Projeto para a disciplina da I2A2: agente autônomo + análise exploratória (EDA) do dataset de fraude em cartão de crédito.

---

## 📂 Estrutura do projeto

fraude-eda-agent/  
│  
├── src/                  # Códigos-fonte dos membros  
│   ├── lopes.py          # Código do Erlon Lopes  
│   ├── rag.py            # Código do Rafael Fagundes (agente principal)  
│   ├── silva.py          # Código do membro Silva  
│   └── ...               # Adicionar outros membros  
│  
├── data/                 # Base de dados  
│   └── creditcard.csv    # Dataset de fraudes em cartão de crédito  
│  
├── notebooks/            # Notebooks Jupyter para EDA e testes  
│   └── exploracao.ipynb  
│  
├── requirements.txt      # Dependências do projeto  
└── README.md             # Este arquivo  

---

## 🎯 Objetivo

Criar um agente que permita responder perguntas sobre o dataset de fraudes em cartão de crédito, com suporte a:  
- Estatísticas descritivas;  
- Visualizações gráficas;  
- Conclusões automatizadas.  

---

## 🚀 Como rodar o projeto

```bash
# 1. Clonar o repositório
git clone https://github.com/erld81/fraude-eda-agent.git
cd fraude-eda-agent

# 2. Criar ambiente Conda
conda create -n fraude python=3.11 -y
conda activate fraude

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Rodar o agente (Streamlit)
streamlit run src/rag.py

---

## 📊 Dataset (resumo)

- **Time:** segundos desde a primeira transação registrada.
- **Amount:** valor da transação.
- **V1...V28:** componentes PCA (anonimizados).
- **Class:** 0 = normal, 1 = fraude.
**Observação:** dataset altamente desbalanceado (~0,17% de fraudes).

---

## 🧑‍💻 Equipe

- Erlon Lopes Dias → src/lopes.py
- Rafael Fagundes → src/rag.py
- Silva → src/silva.py
(adicione outros membros aqui)

---

## 📌 Próximos Passos

Implementar EDA com gráficos e estatísticas.
Integrar agentes individuais.
Produzir relatório final em PDF conforme atividade da disciplina.
