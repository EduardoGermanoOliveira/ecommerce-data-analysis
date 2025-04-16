# 📊 E-commerce Data Analysis and Demand Forecasting

Este projeto realiza uma análise completa de dados de e-commerce, incluindo análise exploratória e previsão de demanda usando diferentes modelos de machine learning.

## 🎯 Objetivos

- Analisar padrões de comportamento de usuários em um e-commerce
- Identificar categorias e produtos mais populares
- Entender padrões temporais de compras
- Desenvolver modelos preditivos de demanda
- Criar visualizações interativas dos insights

## 📊 Resultados Recentes

### Distribuição de Eventos
- Views: 96.11% (23,917,519 eventos)
- Cart: 2.12% (526,431 eventos)
- Purchase: 1.77% (441,703 eventos)

### Top Categorias por Receita
1. electronics.smartphone
2. computers.notebook
3. electronics.video.tv
4. electronics.clocks
5. appliances.kitchen.washer

### Padrões Temporais
- Análise por hora do dia
- Análise por dia da semana
- Identificação de picos de atividade

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
python src/main.py [caminho_do_arquivo]  # Opcional: especifique o arquivo CSV
```

3. Para análise interativa, inicie o Jupyter Notebook:
```bash
jupyter notebook notebooks/
```

## 📊 Funcionalidades

### Análise Exploratória
- Distribuição de eventos (visualização, carrinho, compra)
- Performance por categoria e marca
- Padrões temporais de compras
  - Distribuição por hora do dia
  - Distribuição por dia da semana
- Análise de usuários e produtos
- Cache inteligente de dados para melhor performance

### Visualizações
- Gráficos de distribuição de eventos
- Performance por categoria
- Padrões temporais
  - Gráficos por hora do dia
  - Gráficos por dia da semana

## 📈 Outputs

- Visualizações salvas em `output/plots/`
  - Distribuição de eventos
  - Performance por categoria
  - Padrões temporais
    - Distribuição por hora
    - Distribuição por dia da semana
- Modelos treinados em `output/models/`
- Relatórios de análise em `notebooks/`

## 🔄 Últimas Melhorias

- Implementação de cache de chunks em memória
- Otimização do processamento de grandes datasets
- Melhor tratamento de erros e logging
- Visualizações mais informativas
- Suporte a argumentos de linha de comando
- Novas análises temporais

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
