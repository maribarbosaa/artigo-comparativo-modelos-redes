# Árvore, Floresta ou Boosting? — Preditor de Redes

Projeto acadêmico para comparar **Árvore de Decisão, Random Forest e XGBoost** em uma tarefa de predição relacionada a redes.

Link do Drive com os arquivos: https://drive.google.com/drive/folders/1NlhPmCbzKNPe6_2UlmU4TY_vadcHdCw5

## Estrutura

- `index.html` — página pública.
- `css/style.css` — estilos e responsividade.
- `js/script.js` — pequenas interações.
- `experiment/experiment.py` — experimento reproduzível.
- `results/` — métricas e matrizes geradas pelo experimento.
- `data/` — instruções para baixar a base pública.
- `requirements.txt` — versões mínimas das bibliotecas.

## Experimento

O experimento deve usar a mesma divisão treino/teste, semente fixa e esforço de ajuste comparável entre os três modelos.

### Regra importante

O conjunto de teste não pode ser usado para escolher hiperparâmetros. Se houver tuning, separe treino e validação ou use validação cruzada somente no conjunto de desenvolvimento.

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Coloque a base pública em `data/`, adapte `load_dataset()` em `experiment/experiment.py` e execute:

```bash
python experiment/experiment.py
```

Os resultados serão gravados em `results/metrics.csv` e `results/confusion_matrices.json`.

## Deploy

O projeto é estático e pode ser publicado no GitHub Pages.

1. Crie um repositório público.
2. Envie os arquivos.
3. Vá em Settings → Pages.
4. Escolha a branch principal e a pasta `/root`.
5. Aguarde a URL pública.
6. Teste a URL em outro dispositivo e no celular.

## Privacidade

Publique somente dados pessoais autorizados pelos integrantes. Não publique telefone, e-mail pessoal, endereço, perfis privados ou capturas de tráfego reais.

## Referências iniciais

- Scikit-learn — documentação de árvores e ensembles.
- XGBoost — documentação oficial.
- Chen & Guestrin — artigo original do XGBoost.
- Base pública escolhida pelo grupo e sua licença.

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

