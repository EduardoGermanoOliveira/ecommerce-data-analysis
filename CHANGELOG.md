# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [0.1.0] - 2024-03-XX

### Adicionado
- Implementação inicial do projeto com estrutura básica
- Classe `EcommerceDataLoader` para carregamento de dados em chunks
- Classe `EcommerceAnalyzer` para análise exploratória
- Classe `DemandForecaster` para previsão de demanda
- Visualizações de distribuição de eventos e performance por categoria
- Sistema de cache de chunks em memória para melhor performance

### Melhorado
- Otimização do processamento de chunks para evitar esgotamento do iterator
- Melhor tratamento de erros em todas as classes
- Sistema de logging mais detalhado
- Visualizações mais informativas com matplotlib/seaborn

### Corrigido
- Problema de chunks esgotados após primeira iteração
- Tratamento de dados ausentes em análises de categoria e marca
- Gestão de memória com chunks grandes

### Técnico
- Implementação de cache de DataFrame completo
- Adição de tipos estáticos com Python type hints
- Melhor organização do código em módulos
- Documentação de classes e métodos 