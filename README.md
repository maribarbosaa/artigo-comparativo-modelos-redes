# Comparativo de Modelos de Redes para Cibersegurança (Dataset UNSW-NB15)

Este repositório contém a implementação, os experimentos e os resultados de um estudo comparativo entre diferentes modelos de aprendizado de máquina aplicados à detecção de intrusão e classificação de tráfego de rede utilizando o dataset **UNSW-NB15**.

---

## Estrutura do Repositório

O projeto está organizado da seguinte forma:

* `data/`: Contém os conjuntos de dados utilizados (como o `UNSW_NB15_testing-set.csv`) e notebooks de exploração e prototipagem (`experimento_preditor_redes.ipynb`).
* `experiment/`: Contém os scripts Python dedicados à execução e automação dos experimentos comparativos (`experiment.py`).
* `results/`: Armazena os dados brutos e consolidados dos testes, incluindo métricas em formato tabular (`metrics.csv`), arquivos JSON estruturados (`results.json`) e as matrizes de confusão (`confusion_matrices.json`).
* `css/` e `js/`: Arquivos de folha de estilos (`style.css`) e scripts interativos (`script.js`) para a interface web.
* `index.html`: Dashboard interativo e página de apresentação visual dos resultados comparativos do artigo.
* `requirements.txt`: Lista de dependências e bibliotecas Python necessárias para reproduzir integralmente o ambiente experimental.

---

## Resultados do Experimento

Os modelos avaliados neste estudo compreendem abordagens baseadas em árvores de decisão e métodos de *ensemble* (Árvore de Decisão, Random Forest e XGBoost). A tabela abaixo consolida as métricas de desempenho obtidas no conjunto de testes:

| Modelo | Acurácia Global | Precision Média | Recall Médio | F1-Score Médio | Custo de Treinamento |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Árvore de Decisão** | aprox 0.89 | aprox 0.89 | aprox 0.89 | aprox 0.89 | Baixo |
| **Random Forest** | aprox 0.92 | aprox 0.92 | aprox 0.91 | aprox 0.91 | Moderado |
| **XGBoost** | **aprox 0.93** | **aprox 0.93** | **aprox 0.92** | **aprox 0.92** | Moderado/Alto |

*Nota: Os valores granulares por classe de ataque encontram-se estruturados nos arquivos em `results/metrics.csv` e `results/results.json`.*

---

## Conclusões e Decisões Técnicas

1. **Desempenho de Generalização:** O **XGBoost** obteve o melhor desempenho geral em termos de F1-Score e capacidade de discriminação nas classes de tráfego malicioso e benigno do UNSW-NB15, seguido muito de perto pelo **Random Forest**.
2. **Custo Computacional vs. Robustez:** Embora a **Árvore de Decisão** apresente o menor tempo de processamento, ela demonstrou maior propensão ao *overfitting* em cenários de alta dimensionalidade em comparação aos métodos de *ensemble*.
3. **Viabilidade Prática em Cibersegurança:** Os modelos baseados em *Ensemble* (XGBoost/Random Forest) provaram ser altamente adequados para ambientes reais de detecção de intrusão (IDS), equilibrando excelente acurácia com tempos de inferência viáveis para monitoramento de tráfego.

---

## Como Executar o Projeto

Siga os passos abaixo para reproduzir os experimentos e visualizar os resultados:

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o script de experimentos:**
   ```bash
   python experiment/experiment.py
   ```

3. **Visualize o Dashboard:**
   Abra o arquivo `index.html` diretamente em seu navegador web para explorar os gráficos dinâmicos e as métricas comparativas detalhadas.

---

## Licença

Este projeto está licenciado sob os termos da licença informada no arquivo [LICENSE](LICENSE).

---

## Integrantes do Grupo

- **Guilherme Trajane da Silva**
- **Leandro do Nascimento Lemes**
- **Mariana Moreira Barbosa**
- **Samara Fernandes Soares**
- **Vitor Julião Diogo dos Santos**

---

### Informações Acadêmicas
- **Curso / Disciplina:** Estrutura de Dados II
- **Orientadora:** Profª. Andrea Ono Sakai

```
