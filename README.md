# 📊 E-commerce Data Analysis and Demand Forecasting

Este projeto realiza uma análise completa de dados de e-commerce, incluindo análise exploratória e previsão de demanda usando diferentes modelos de machine learning.

## 🎯 Objetivos

- Analisar padrões de comportamento de usuários em um e-commerce
- Identificar categorias e produtos mais populares
- Entender padrões temporais de compras
- Desenvolver modelos preditivos de demanda
- Criar visualizações interativas dos insights

## 📋 Estrutura do Projeto

```
ecommerce-data-analysis/
├── data/
│   ├── raw/           # Dados brutos do Kaggle
│   └── processed/     # Dados processados
├── src/
│   ├── data/          # Scripts de carregamento e processamento
│   ├── analysis/      # Análise exploratória
│   ├── models/        # Modelos de previsão
│   └── main.py        # Script principal
├── notebooks/         # Jupyter notebooks para análise
├── output/
│   ├── plots/         # Visualizações geradas
│   └── models/        # Modelos treinados
└── requirements.txt   # Dependências do projeto
```

## 🚀 Tecnologias Utilizadas

- **Python 3.8+**
  - pandas: Manipulação de dados
  - numpy: Computação numérica
  - scikit-learn: Machine Learning
  - prophet: Previsão de séries temporais
  - matplotlib/seaborn: Visualização
  - jupyter: Análise interativa

## 📥 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/ecommerce-data-analysis.git
cd ecommerce-data-analysis
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Como Usar

1. Baixe o dataset do Kaggle e coloque-o em `data/raw/`

2. Execute a análise completa:
```bash
python src/main.py
```

3. Para análise interativa, inicie o Jupyter Notebook:
```bash
jupyter notebook notebooks/
```

## 📊 Funcionalidades

### Análise Exploratória
- Distribuição de eventos (visualização, carrinho, compra)
- Performance por categoria
- Padrões temporais de compras
- Análise de usuários e produtos

### Previsão de Demanda
- Modelo Random Forest para previsão de vendas
- Modelo Prophet para análise de tendências
- Avaliação de importância de features
- Métricas de performance dos modelos

## 📈 Outputs

- Visualizações salvas em `output/plots/`
- Modelos treinados em `output/models/`
- Relatórios de análise em `notebooks/`

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Autores

- Eduardo Germano de Oliveira - [@EduardoGermanoOliveira](https://github.com/EduardoGermanoOliveira)

## 🙏 Agradecimentos

- Dataset fornecido pelo Kaggle
- Comunidade open source por suas ferramentas e bibliotecas
